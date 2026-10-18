#!/usr/bin/env python
"""Decoder latency benchmarks (step 5).

  A. pull-mode latency vs chunk length (0.5, 1, 1.9, 3, 5 s), per stage incl. disk read
  B. push-mode real-time factor at plugin-sized buffers (384, 1024 samples)
  C. scaling with numba thread count (1, 2, 4, 8) at the 1.9 s chunk

Writes results/latency_bench.csv (A, C) and results/latency_push_buffers.csv (B).
Memory-bounded (MemGuard 3.5 GB).
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamsort.memguard import preflight, MemGuard  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import OpenEphysSession, FileStreamSource, SimulatedLiveStream  # noqa: E402
from streamsort.sorting import TemplateBank  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, SpikeBatch, Whitening  # noqa: E402
import numba  # noqa: E402

ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"


def timed_window(dec, src, start, n, warmup_ms=15.0, tail_ms=4.0):
    W = dec.cfg.samples(warmup_ms); Tl = dec.cfg.samples(tail_ms)
    s0 = start - W
    t0 = time.perf_counter()
    raw = np.asarray(src.read(s0, start + n + Tl - s0))
    t_read = time.perf_counter() - t0
    dec.start_at(s0)
    b = SpikeBatch.concat([dec.push(raw), dec.flush()])
    total = time.perf_counter() - t0
    return dict(read=t_read, total=total, events=len(b), **dec.timing)


def main():
    preflight(3.5, margin_gb=0.0, strict=False)
    guard = MemGuard(3.5, "bench")
    st = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1").stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts),
                              detect_threshold=4.0)
    src = FileStreamSource(st, 200)
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    usable = pd.read_csv("results/sorting_report.csv").usable.to_numpy(bool)
    wh = Whitening.load("results/whitening_probeA.npz")
    dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False)
    base = cfg.samples(20_000)
    nthr_default = numba.get_num_threads()
    rows = []

    # A. chunk length sweep
    for L in (0.5, 1.0, 1.9, 3.0, 5.0):
        n = cfg.samples(L * 1e3)
        for rep in range(12):
            start = base + rep * cfg.samples(6_000)
            r = timed_window(dec, src, start, n)
            if rep >= 2:                                  # drop page-cache / warm-up reps
                rows.append(dict(bench="chunk_length", chunk_s=L, threads=nthr_default, rep=rep, **r))
        sub = pd.DataFrame([x for x in rows if x["bench"] == "chunk_length" and x["chunk_s"] == L])
        print(f"[bench] chunk {L:3.1f} s: total median {sub.total.median()*1e3:5.0f} ms, p95 "
              f"{sub.total.quantile(0.95)*1e3:5.0f} ms | read {sub.read.median()*1e3:4.0f} ms | events {sub.events.median():.0f}",
              flush=True)
        guard.check(f"chunk {L}")

    # C. thread scaling at 1.9 s
    n = cfg.samples(1_900)
    for th in (1, 2, 4, 8):
        if th > nthr_default:
            continue
        numba.set_num_threads(th)
        for rep in range(8):
            r = timed_window(dec, src, base + rep * cfg.samples(6_000), n)
            if rep >= 2:
                rows.append(dict(bench="threads", chunk_s=1.9, threads=th, rep=rep, **r))
        sub = pd.DataFrame([x for x in rows if x["bench"] == "threads" and x["threads"] == th])
        print(f"[bench] threads {th}: 1.9 s chunk total median {sub.total.median()*1e3:5.0f} ms", flush=True)
    numba.set_num_threads(nthr_default)
    pd.DataFrame(rows).to_csv("results/latency_bench.csv", index=False)

    # B. push-mode real-time factor
    prow = []
    for B in (384, 1024):
        live = SimulatedLiveStream(src, head=0)
        dec.start_at(base)
        times = []
        for i, (s, buf) in enumerate(live.iter_buffers(B, start=base, stop=base + cfg.samples(30_000))):
            t0 = time.perf_counter(); dec.push(buf); dt = time.perf_counter() - t0
            if i >= 50:
                times.append(dt)
        times = np.array(times); dur = B / cfg.sample_rate
        rtf = times / dur
        for t_ in times:
            prow.append(dict(buffer=B, proc_s=t_, buffer_s=dur, rtf=t_ / dur))
        print(f"[bench] push {B:4d}-sample buffers ({dur*1e3:.1f} ms): proc median {np.median(times)*1e3:.2f} ms, "
              f"p99 {np.percentile(times,99)*1e3:.2f} ms, max {times.max()*1e3:.2f} ms | RTF median {np.median(rtf):.3f}, "
              f"p99 {np.percentile(rtf,99):.3f} | overruns (RTF>1) {np.mean(rtf>1)*100:.2f}%", flush=True)
        guard.check(f"push {B}")
    pd.DataFrame(prow).to_csv("results/latency_push_buffers.csv", index=False)
    print(f"[bench] DONE | numba threads default {nthr_default} | {guard.report()}", flush=True)


if __name__ == "__main__":
    main()
