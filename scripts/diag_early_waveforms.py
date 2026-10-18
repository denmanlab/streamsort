"""Are 2-5 ms detections spikes?  Single-detection and STA waveform fidelity vs
late-window detections of the same units, on V2-cleaned, causally filtered data
(high-charge conditions, evaluation trials).  Also: mean residual at random
times (latency-matched) = what an artifact-only STA would look like."""
import os, sys, dataclasses, collections, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from exp_artifact_variants import Wrapped
import ctx
ROOT = ctx.ROOT
from streamsort.config import StreamConfig
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span
from streamsort.preprocess.filters import CausalPreprocessor
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window
sess = OpenEphysSession(ctx.REC_DIR)
st = sess.stream(ctx.STREAM)
cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=ctx.NCH)
src = FileStreamSource(st, ctx.NCH); fs = cfg.sample_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
bank = TemplateBank.load(ctx.R("templates.npz"))
usable = pd.read_csv(ctx.R("sorting_report.csv")).usable.to_numpy(bool)
dec = OnlineDecoder(bank, cfg, Whitening.load(ctx.R("whitening.npz")), DecoderParams(min_energy=50.0), units_mask=usable, check=False)
pre = CausalPreprocessor(cfg, ctx.NCH)
D = bank.dense(ctx.NCH); T = bank.n_samples; tro = bank.trough_offset
tr = pd.read_csv(ctx.R("trials.csv")); tr = tr[(tr.amplitude.abs() * tr.pulse_duration) >= 5000]
off_all = np.load(ctx.R("online_spikes_all_trials_sup.npz"))["meta_pulse_offset"]
L = S(20); taper = np.ones(L, np.float32); k = S(5); taper[-k:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
rng = np.random.default_rng(0)
snips = collections.defaultdict(list)   # (unit, cls) -> list of snippets on unit's 8 channels
chans = {u: np.argsort(-(D[u] ** 2).sum(0))[:8] for u in range(len(D))}
for cond, g in tr.groupby("cond", sort=False):
    hist = collections.deque(maxlen=20)
    for j, (_, t) in enumerate(g.sort_values("trial").iloc[:40].iterrows()):
        tk = int(t.trial); off = int(off_all[tk]); nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0])
        raw = np.asarray(src.read(nom - S(5), S(5) + L + S(2))).astype(np.float32)
        onr = detect_artifact_onset(raw, S(5), S(1.5))
        if onr is None: continue
        on = nom - S(5) + onr; span = clipped_span(raw, onr, S(5))
        if j >= 20 and len(hist) >= 5:
            tpl = (np.median(np.stack(hist), 0) * taper[:, None]).astype(np.float32)
            ws = Wrapped(src, on, span, tpl, True, S)
            b = decode_window(dec, ws, off + S(1), S(149)); b = b.select(b.unit >= 0)
            s0 = on - S(16); x = np.asarray(ws.read(s0, off + S(152) - s0)).astype(np.float32)
            pre.reset(); pre.prime(x[0]); y = pre.process(x)
            for f, u in zip(b.frame, b.unit):
                lat = (f - off) / fs * 1e3
                cls = "1-2" if lat < 2 else "2-5" if lat < 5 else "5-10" if lat < 10 else "late" if lat >= 20 else None
                if cls is None: continue
                i = f - tro[u] - s0
                if i < 0 or i + T > len(y): continue
                snips[(u, cls)].append(y[i:i + T][:, chans[u]])
                # latency-matched random-time control on the same channels, other trial-time
                if cls in ("1-2", "2-5"):
                    fr = off + S(rng.uniform(1, 5)) if cls == "2-5" else off + S(rng.uniform(1, 2))
                    ir = fr - tro[u] - s0
                    if ir >= 0 and ir + T <= len(y):
                        snips[(u, cls + "_rand")].append(y[ir:ir + T][:, chans[u]])
        hist.append(raw[onr:onr + L] - raw[onr - S(1):onr].mean(0))
    print("done", cond, flush=True)
rows = []
for (u, cls), v in snips.items():
    X = np.stack(v); tp = D[u][:, chans[u]]
    sta = X.mean(0); cos_sta = float((sta * tp).sum() / (np.linalg.norm(sta) * np.linalg.norm(tp) + 1e-9))
    gain = float((sta * tp).sum() / (tp ** 2).sum())
    cs = (X * tp).sum((1, 2)) / (np.linalg.norm(X.reshape(len(X), -1), axis=1) * np.linalg.norm(tp) + 1e-9)
    rows.append(dict(unit=u, cls=cls, n=len(X), cos_sta=cos_sta, gain_sta=gain, cos_single_med=float(np.median(cs))))
df = pd.DataFrame(rows); df.to_csv(ctx.R("diag_early_waveforms.csv"), index=False)
ok = df.groupby("unit").filter(lambda d: (d.set_index("cls").n.reindex(["2-5", "late"]).fillna(0) >= 15).all())
print("units with >=15 detections in both 2-5 ms and late:", ok.unit.nunique())
w = ok.groupby("cls").agg(units=("unit", "nunique"), n=("n", "sum"), cos_sta=("cos_sta", "median"), gain=("gain_sta", "median"), cos_single=("cos_single_med", "median"))
print(w.round(3).to_string())
