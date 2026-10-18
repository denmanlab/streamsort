"""Compare near-pulse mitigations on real trials vs the KS2.5 reference.

Variants are raw-domain transforms applied by a source wrapper before the
unchanged decoder (so they are causal and live-compatible):
  V0 baseline (decoder's own clip blank + sample-and-hold)
  V1 subtract causal per-condition median artifact (previous N trials), tapered
  V2 V1 + bridge the clipped span by linear interpolation (no step at release)
  V3 bridge only
Trials HIST..HIST+NEVAL-1 of every condition are evaluated; earlier trials only
feed the template history.  Window: pulse offset +1 ms .. +50 ms.
"""
import os, sys, dataclasses, collections, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from streamsort.config import StreamConfig
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window
from streamsort.memguard import MemGuard, preflight
ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
N_HIST, START, NEVAL = int(os.environ.get("N_HIST", 20)), 20, int(os.environ.get("NEVAL", 20))
L_MS, TAPER_MS = float(os.environ.get("L_MS", 20)), 5.0
BINS = [1, 2, 3, 5, 10, 20, 50]

class Wrapped:
    def __init__(self, src, on, span, tpl, bridge, S):
        self.src, self.on, self.span, self.tpl, self.bridge, self.S = src, on, span, tpl, bridge, S
    def read(self, s0, n):
        x = np.asarray(self.src.read(s0, n)).astype(np.float32)
        o = self.on - s0
        if self.tpl is not None:
            a, b = max(o, 0), min(o + len(self.tpl), n)
            if b > a: x[a:b] -= self.tpl[a - o:b - o]
        if self.bridge:
            b0, b1 = o - self.S(0.2), o + self.span + self.S(0.5)
            if b0 >= 3 and b1 + 3 <= n:
                l, r = x[b0 - 3:b0].mean(0), x[b1:b1 + 3].mean(0)
                w = np.linspace(0, 1, b1 - b0, dtype=np.float32)[:, None]
                x[b0:b1] = (1 - w) * l + w * r
        return np.clip(np.rint(x), -32768, 32767).astype(np.int16)

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "variants")
    sess = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1")
    st = sess.stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0)
    src = FileStreamSource(st, 200); fs = cfg.sample_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    usable = pd.read_csv("results/sorting_report.csv").usable.to_numpy(bool)
    dec = OnlineDecoder(bank, cfg, Whitening.load("results/whitening_probeA.npz"),
                        DecoderParams(min_energy=50.0), units_mask=usable, check=False)
    tr = pd.read_csv("results/jlh32_probeA_trials.csv")
    on_np = np.load("results/online_spikes_all_trials.npz"); off_all = on_np["meta_pulse_offset"]
    rf = dict(np.load("results/reference_spikes_all_trials.npz")); units = np.unique(rf["ks4_to_nwb"][:, 0])
    L = S(L_MS); taper = np.ones(L, np.float32); k = S(TAPER_MS)
    taper[-k:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
    rows = []
    for cond, g in tr.groupby(["amplitude", "polarity", "pulse_duration"], sort=False):
        g = g.sort_values("trial").iloc[:START + NEVAL]
        hist = collections.deque(maxlen=N_HIST)
        for j, (_, t) in enumerate(g.iterrows()):
            tk = int(t.trial); off = int(off_all[tk]); nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0])
            raw = np.asarray(src.read(nom - S(5), S(5) + L + S(2))).astype(np.float32)
            onr = detect_artifact_onset(raw, S(5), S(1.5))
            if onr is None: continue
            on = nom - S(5) + onr; span = clipped_span(raw, onr, S(5))
            snip = raw[onr:onr + L] - raw[onr - S(1):onr].mean(0)
            if j >= START and len(hist) >= 5:
                tpl = (np.median(np.stack(hist), 0) * taper[:, None]).astype(np.float32)
                w0, nW = off + S(1), S(49)
                rm = (rf["trial"] == tk) & np.isin(rf["unit"], units)
                rfr, rur = rf["frame"][rm], rf["unit"][rm]
                for vname, tp_, br in [("V0", None, False), ("V1", tpl, False), ("V2", tpl, True), ("V3", None, True)]:
                    s_ = src if vname == "V0" else Wrapped(src, on, span, tp_, br, S)
                    b = decode_window(dec, s_, w0, nW); b = b.select(np.isin(b.unit, units))
                    for lo, hi in zip(BINS[:-1], BINS[1:]):
                        mo = ((b.frame - off) >= S(lo)) & ((b.frame - off) < S(hi))
                        mr = ((rfr - off) >= S(lo)) & ((rfr - off) < S(hi))
                        fo, uo, fr, ur = b.frame[mo], b.unit[mo], rfr[mr], rur[mr]
                        tp = 0
                        for u in np.unique(uo):
                            a_ = fo[uo == u]; r_ = fr[ur == u]
                            if len(r_): tp += int((np.abs(a_[:, None] - r_[None]).min(1) <= 12).sum())
                        rows.append(dict(amp=cond[0], pol=cond[1], dur=cond[2], trial=tk, var=vname, lo=lo,
                                         n_on=len(fo), n_ref=len(fr), tp=tp))
            hist.append(snip)
        guard.check(str(cond)); print(f"[var] {cond} | {guard.report()}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"results/exp_artifact_variants_N{N_HIST}_L{int(L_MS)}.csv", index=False)
    nU = len(units)
    df["hi_charge"] = (df.amp.abs() * df.dur) >= 5000
    for hc, d in df.groupby("hi_charge"):
        a = d.groupby(["var", "lo"])[["n_on", "n_ref", "tp"]].sum()
        ntr = d.trial.nunique(); bw = pd.Series(np.diff(BINS), index=BINS[:-1])
        a["rate_on"] = a.n_on / (ntr * nU * a.index.get_level_values("lo").map(bw).to_numpy() / 1e3)
        a["P"] = a.tp / a.n_on; a["R"] = a.tp / a.n_ref
        print(f"\n== {'high' if hc else 'low'} charge (|I|*dur >= 5000 uA*us: {hc}), {ntr} trials; ref rate shown as V0 n_ref")
        print(a[["rate_on", "P", "R"]].unstack("var").round(2).to_string())

if __name__ == "__main__":
    main()
