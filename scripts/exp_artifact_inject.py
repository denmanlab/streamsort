"""Artifact-injection test: is the near-pulse excess artifact or real spiking?

Per condition: median raw artifact (onset-aligned, baseline-referenced, 60 ms,
cosine taper over the last 10 ms) from the first 30 trials.  Inject it into
quiet data 1.0 s after real pulses (clip to the measured rails), decode
[inj_offset+1 ms, +300 ms) with and without injection, and match spikes
(same unit, +/-12 samples).  New spikes = artifact-induced FPs; lost spikes =
artifact-induced misses.  Independent of the KS2.5 reference.
"""
import os, sys, dataclasses, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from streamsort.config import StreamConfig
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.artifact import detect_artifact_onset
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window
from streamsort.memguard import MemGuard, preflight
sys.path.insert(0, os.path.join(os.getcwd(), "scripts")); import ctx
ROOT = ctx.ROOT
RAIL_HI, RAIL_LO = 6369, -6206
N_TPL, N_INJ = 30, int(os.environ.get("N_INJ", "12"))
BINS = [1, 3, 5, 10, 20, 50, 300]

class InjectedSource:
    def __init__(self, src, at, tpl):
        self.src, self.at, self.tpl = src, int(at), tpl
    def read(self, s0, n):
        x = np.asarray(self.src.read(s0, n)).astype(np.float32)
        a, b = max(self.at, s0), min(self.at + len(self.tpl), s0 + n)
        if b > a:
            x[a - s0:b - s0] += self.tpl[a - self.at:b - self.at]
        return np.clip(np.rint(x), RAIL_LO, RAIL_HI).astype(np.int16)

def matched(fa, ua, fb, ub, tol=12):
    """bool per spike in a: has a same-unit partner in b within tol."""
    ok = np.zeros(len(fa), bool)
    for u in np.unique(ua):
        ia = np.flatnonzero(ua == u); r = np.sort(fb[ub == u])
        if not len(r): continue
        i = np.clip(np.searchsorted(r, fa[ia]), 1, max(len(r) - 1, 1))
        d = np.minimum(np.abs(fa[ia] - r[i - 1]), np.abs(fa[ia] - r[np.minimum(i, len(r) - 1)]))
        ok[ia] = d <= tol
    return ok

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "inject")
    sess = OpenEphysSession(ctx.REC_DIR)
    st = sess.stream(ctx.STREAM)
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=ctx.NCH)
    src = FileStreamSource(st, ctx.NCH); fs = cfg.sample_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
    bank = TemplateBank.load(ctx.R("templates.npz"))
    usable = pd.read_csv(ctx.R("sorting_report.csv")).usable.to_numpy(bool)
    dec = OnlineDecoder(bank, cfg, Whitening.load(ctx.R("whitening.npz")),
                        DecoderParams(min_energy=50.0), units_mask=usable, check=False)
    tr = pd.read_csv(ctx.R("trials.csv"))
    L = S(60); taper = np.ones(L, np.float32); k = S(10)
    taper[-k:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
    rows = []
    for cond_lab, g in tr.groupby("cond", sort=False):
        cond = (int(g.amplitude.iloc[0]), str(g.polarity.iloc[0]), int(g.pulse_duration.iloc[0]))
        g = g.sort_values("trial"); sn = []
        for _, t in g.iloc[:N_TPL].iterrows():
            nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0])
            raw = np.asarray(src.read(nom - S(5), S(70))).astype(np.float32)
            on = detect_artifact_onset(raw, S(5), S(1.5))
            if on is None: continue
            sn.append(raw[on:on + L] - raw[on - S(1):on].mean(0))
        tpl = (np.median(np.stack(sn), 0) * taper[:, None]).astype(np.float32); del sn
        pw = S(2 * t.pulse_duration * 1e-3)
        for _, t in g.iloc[N_TPL - N_INJ:N_TPL].iterrows() if len(g) >= N_TPL else g.iloc[:N_INJ].iterrows():
            off = int(st.time_to_sample(np.array([t.ttl_offset]))[0])
            inj = off + S(1000); w0 = inj + pw + S(1); nW = S(299)
            bc = decode_window(dec, src, w0, nW); bc = bc.select(bc.unit >= 0)
            bi = decode_window(dec, InjectedSource(src, inj, tpl), w0, nW); bi = bi.select(bi.unit >= 0)
            new = ~matched(bi.frame, bi.unit, bc.frame, bc.unit)
            lost = ~matched(bc.frame, bc.unit, bi.frame, bi.unit)
            li = (bi.frame - (inj + pw)) / fs * 1e3; lc = (bc.frame - (inj + pw)) / fs * 1e3
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                mi = (li >= lo) & (li < hi); mc = (lc >= lo) & (lc < hi)
                rows.append(dict(cond=cond_lab, amp=cond[0], pol=cond[1], dur=cond[2], trial=int(t.trial), lo=lo, hi=hi,
                                 n_clean=int(mc.sum()), n_inj=int(mi.sum()), n_new=int((mi & new).sum()),
                                 n_lost=int((mc & lost).sum()), new_score_med=float(np.median(bi.score[mi & new])) if (mi & new).any() else np.nan))
        guard.check(str(cond)); print(f"[inj] {cond} | {guard.report()}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(ctx.R("exp_artifact_inject.csv"), index=False)
    nU = int(usable.sum())
    s = df.groupby(["cond", "lo"]).agg(n=("trial", "size"), clean=("n_clean", "sum"), new=("n_new", "sum"), lost=("n_lost", "sum")).reset_index()
    s["w"] = s.n * (s.lo.map(dict(zip(BINS[:-1], np.diff(BINS)))) / 1e3) * nU
    s["clean_hz"] = s.clean / s.w; s["fp_hz"] = s.new / s.w; s["miss_frac"] = s.lost / s.clean.clip(lower=1)
    p = s.pivot_table(index=["cond"], columns="lo", values=["fp_hz", "miss_frac"])
    pd.set_option("display.width", 250); print(p.round(2).to_string())
    print("overall clean rate Hz/unit:", round(s.clean.sum() / s.w.sum(), 2))

if __name__ == "__main__":
    main()
