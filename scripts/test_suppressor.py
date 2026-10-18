"""Chunking invariance + agreement with the pull-mode experiment wrapper."""
import os, sys, collections, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from exp_artifact_variants import Wrapped, ROOT
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.suppressor import ArtifactSuppressor
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span
sess = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1")
st = sess.stream("Neuropix-PXI-104.ProbeA-AP"); src = FileStreamSource(st, 200)
fs = st.effective_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
tr = pd.read_csv("results/jlh32_probeA_trials.csv")
g = tr[(tr.amplitude == -100) & (tr.polarity == "monopolar")].sort_values("trial").iloc[:10]
rng = np.random.default_rng(1)
def run(chunker):
    sup = ArtifactSuppressor(fs, 200); outs = []
    for _, t in g.iterrows():
        nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0]); s0 = nom - S(15)
        x = np.asarray(src.read(s0, S(60)))
        sup.reset(s0); sup.set_condition(("m", -100, 100))
        o = [sup.push(c) for c in chunker(x)] + [sup.flush()]
        outs.append(np.concatenate(o))
    return outs, sup
a, supA = run(lambda x: [x])
b, _ = run(lambda x: np.array_split(x, np.cumsum(rng.integers(1, 200, 100)))[:-1] + [x[np.cumsum(rng.integers(1,200,100))[-1]:]] if False else np.split(x, np.sort(rng.choice(np.arange(1, len(x)), 40, replace=False))))
print("lengths ok:", all(len(u) == S(60) for u in a), "| chunk-invariant (max abs diff):", max(float(np.abs(u - v).max()) for u, v in zip(a, b)))
print("events:", [(e["span"], e["n_history"], e["subtracted"]) for e in supA.events])
# compare trial 9 with the pull-mode wrapper using the same history (first 9 trials)
t = g.iloc[9]; nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0]); s0 = nom - S(15)
L = S(20); taper = supA.taper
hist = []
for _, tt in g.iloc[:9].iterrows():
    n2 = int(st.time_to_sample(np.array([tt.ttl_onset]))[0]); raw = np.asarray(src.read(n2 - S(5), S(5) + L + S(2))).astype(np.float32)
    o = detect_artifact_onset(raw, S(5), S(1.5)); hist.append(raw[o:o + L] - raw[o - S(1):o].mean(0))
tpl = (np.median(np.stack(hist), 0) * taper[:, None]).astype(np.float32)
raw = np.asarray(src.read(nom - S(5), S(5) + L + S(2))).astype(np.float32); o = detect_artifact_onset(raw, S(5), S(1.5))
w = Wrapped(src, nom - S(5) + o, clipped_span(raw, o, S(5)), tpl, True, S).read(s0, S(60)).astype(np.float32)
d = np.abs(w - a[9]); print("vs pull wrapper: max abs diff (counts)", float(d.max()), "| frac > 0.5:", float((d > 0.5).mean()))
print("span stream vs offline:", supA.events[9]["span"], clipped_span(raw, o, S(5)))
for k, (u, v) in enumerate(zip(a, b)):
    d = np.abs(u - v).max(1); bad = np.flatnonzero(d > 1e-3)
    if len(bad):
        on_rel = supA.events[k]["onset"] - (int(st.time_to_sample(np.array([g.iloc[k].ttl_onset]))[0]) - S(15))
        print(k, "onset idx", on_rel, "span", supA.events[k]["span"], "bad range", bad.min(), bad.max(), "n", len(bad))
