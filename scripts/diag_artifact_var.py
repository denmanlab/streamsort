import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from streamsort.config import StreamConfig
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span
ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
sess = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1")
st = sess.stream("Neuropix-PXI-104.ProbeA-AP"); src = FileStreamSource(st, 200)
fs = st.effective_rate; S = lambda ms: int(round(ms * 1e-3 * fs)); bv = st.bit_volts
tr = pd.read_csv("results/jlh32_probeA_trials.csv")
out = []
for cond, g in tr.groupby(["amplitude", "polarity", "pulse_duration"], sort=False):
    sn, spans, d_on = [], [], []
    for _, t in g.sort_values("trial").iloc[:30].iterrows():
        nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0])
        raw = np.asarray(src.read(nom - S(5), S(30))).astype(np.float32)
        on = detect_artifact_onset(raw, S(5), S(1.5))
        if on is None: continue
        sp = clipped_span(raw, on, S(5)); spans.append(sp); d_on.append(on - S(5))
        sn.append((raw[on - S(1):on + S(20)] - raw[on - S(1):on].mean(0)) * bv)
    X = np.stack(sn); t0 = S(1)
    med = np.median(X, 0)
    # deviations at latencies after onset, on the 20 channels with the largest tail at 3 ms
    ch = np.argsort(-np.abs(med[t0 + S(3)]))[:20]
    row = dict(cond=str(cond), n=len(X), span_med=np.median(spans), span_rng=f"{min(spans)}-{max(spans)}",
               onset_jit_sd=np.std(d_on))
    for ms in [1, 2, 3, 5, 10, 15]:
        i = t0 + S(ms)
        row[f"tail{ms}"] = float(np.median(np.abs(med[i, ch])))
        row[f"sd{ms}"] = float(np.median(X[:, i, ch].std(0)))
    # scale model: per-trial per-channel least squares gain on 2-8 ms
    sl = slice(t0 + S(2), t0 + S(8))
    gains = (X[:, sl, :][:, :, ch] * med[sl][:, ch]).sum(1) / (med[sl][:, ch] ** 2).sum(0)
    row["gain_sd"] = float(np.median(gains.std(0)))
    res = X[:, sl, :][:, :, ch] - gains[:, None, :] * med[sl][None][:, :, ch]
    row["res_sd_scaled"] = float(np.median(res.std(0)))
    row["res_sd_fixed"] = float(np.median((X[:, sl, :][:, :, ch] - med[sl][None][:, :, ch]).std(0)))
    out.append(row)
df = pd.DataFrame(out); pd.set_option("display.width", 250)
print(df.round(2).to_string(index=False))
