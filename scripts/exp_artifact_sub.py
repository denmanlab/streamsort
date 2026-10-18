"""Experiment: causal per-condition artifact-template subtraction.

For every trial: read raw [onset-30 ms, onset+60 ms] (probe A, ch 0-199), find the
data-derived artifact onset, baseline each channel on the 1 ms before onset, and
subtract a template = median over the previous N trials of the same condition
(aligned on detected onset, causal: trial k never sees itself or later trials).
Then blank the clipped span exactly as the decoder does (sample-and-hold) and run
the causal filter + median CAR.  Score residual RMS and -4 sigma crossing rate per
post-offset latency bin.  Bounded memory: one condition's history at a time.
"""
import os, sys, dataclasses, collections, json
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from streamsort.config import StreamConfig
from streamsort.io.openephys import OpenEphysSession
from streamsort.io.stream import FileStreamSource
from streamsort.preprocess.filters import CausalPreprocessor
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span, CLIP_COUNTS
from streamsort.memguard import MemGuard, preflight

ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
NS = [0, 3, 10, 25]          # history depth (0 = no subtraction)
LS_MS = [10, 25, 50]         # template length after onset
BINS_MS = [1, 2, 3, 5, 10, 20, 40, 55]
PER_RUN = int(os.environ.get("PER_RUN", "0"))

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "exp_artifact_sub")
    sess = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1")
    st = sess.stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts)
    src = FileStreamSource(st, 200)
    tr = pd.read_csv("results/jlh32_probeA_trials.csv")
    fs = cfg.sample_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
    PRE, POST = S(30), S(60)
    bpre, bpost = S(0.2), S(0.5)
    pre = CausalPreprocessor(cfg, 200)
    # noise per channel from a stim-free stretch (filtered) for the threshold
    quiet = np.asarray(src.read(st.time_to_sample(np.array([tr.ttl_onset.iloc[0] - 5.0]))[0], S(2000))).astype(np.float32)
    pre.reset(); pre.prime(quiet[0]); yq = pre.process(quiet)[S(100):]
    sig = (np.median(np.abs(yq - np.median(yq, 0)), 0) / 0.6745).astype(np.float32)
    thr = -4 * sig
    rows = []
    variants = [(0, 0)] + [(N, L) for N in NS[1:] for L in LS_MS]
    for cond, g in tr.groupby(["amplitude", "polarity", "pulse_duration"], sort=False):
        hist = collections.deque(maxlen=max(NS))
        g = g.sort_values("trial")
        if PER_RUN: g = g.iloc[:PER_RUN]
        for _, t in g.iterrows():
            nom = int(st.time_to_sample(np.array([t.ttl_onset]))[0])
            raw = np.asarray(src.read(nom - PRE, PRE + POST)).astype(np.float32)
            on = detect_artifact_onset(raw, PRE, S(1.5))
            if on is None or on - PRE > S(1.5) or on < S(16):
                continue
            span = clipped_span(raw, on, S(5))
            off_rel = on + S(t.pulse_duration * 2e-3)          # biphasic: 2 x phase width
            base = raw[on - S(1):on].mean(0)
            snip = raw[on:on + S(max(LS_MS))] - base           # this trial's artifact (onset-aligned)
            for (N, L) in variants:
                x = raw.copy()
                if N and len(hist) >= min(N, 3):
                    h = np.stack(list(hist)[-N:]); tpl = np.median(h, 0)[:S(L)]
                    x[on:on + S(L)] -= tpl
                # decoder-style blank: clipped samples +/- pad, hold last clean
                b0, b1 = on - bpre, on + span + bpost
                x[b0:b1] = x[b0 - 1]
                x = x[on - S(16):]                           # 16 ms warm-up for the filter
                pre.reset(); pre.prime(x[0]); y = pre.process(x)
                t_ms = (np.arange(len(y)) + on - S(16) - off_rel) / fs * 1e3
                for lo, hi in zip(BINS_MS[:-1], BINS_MS[1:]):
                    m = (t_ms >= lo) & (t_ms < hi)
                    yy = y[m]
                    cross = ((yy[1:] < thr) & (yy[:-1] >= thr)).sum() / ((hi - lo) * 1e-3) / 200
                    rows.append(dict(trial=int(t.trial), run=int(t.run), amp=t.amplitude, pol=t.polarity,
                                     dur=t.pulse_duration, N=N, L=L, lo=lo, hi=hi, n_hist=len(hist),
                                     rms_uV=float(np.median(np.sqrt((yy ** 2).mean(0)))),
                                     cross_hz=float(cross)))
            hist.append(snip)
        guard.check(f"cond {cond}")
        print(f"[exp] cond {cond} done | {guard.report()}", flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/exp_artifact_sub.csv", index=False)
    print("noise median uV", float(np.median(sig)))
    s = df[df.n_hist >= 3].groupby(["N", "L", "lo"]).agg(rms=("rms_uV", "median"), cross=("cross_hz", "mean")).unstack("lo")
    print(s.round(1).to_string())

if __name__ == "__main__":
    main()
