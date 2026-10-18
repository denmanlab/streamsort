#!/usr/bin/env python
"""Numba decoder path: equivalence with the numpy reference, and latency."""
from __future__ import annotations

import dataclasses
import json
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
from streamsort.io import OpenEphysSession, FileStreamSource  # noqa: E402
from streamsort.sorting import TemplateBank  # noqa: E402
from streamsort.preprocess import CausalPreprocessor  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, SpikeBatch, Whitening, decode_window  # noqa: E402
from streamsort.online import kernels  # noqa: E402

ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"


def main():
    preflight(3.5, margin_gb=0.0, strict=False)
    guard = MemGuard(3.5, "test_numba")
    st = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1").stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts),
                              detect_threshold=4.0)
    src = FileStreamSource(st, 200)
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    usable = pd.read_csv("results/sorting_report.csv").usable.to_numpy(bool)
    wh = Whitening.load("results/whitening_probeA.npz")
    p = DecoderParams(min_energy=50.0)

    t = time.perf_counter()
    dn = OnlineDecoder(bank, cfg, wh, p, units_mask=usable, check=False, use_numba=True)
    print(f"[test] numba decoder built + JIT warm-up in {time.perf_counter()-t:.1f}s", flush=True)
    dp = OnlineDecoder(bank, cfg, wh, p, units_mask=usable, check=False, use_numba=False)

    # filter equivalence on raw kernels
    raw = np.asarray(src.read(cfg.samples(20_000), cfg.samples(1_000)))
    ref = CausalPreprocessor(cfg, 200).process(raw)
    pp = CausalPreprocessor(cfg, 200); x0 = raw[0].astype(np.float64) * cfg.bit_volts
    zi = pp._zi_unit[:, :, None] * x0[None, None, :]
    nbf = kernels.median_reference(kernels.sos_filter(raw, cfg.bit_volts, pp.sos.astype(np.float64), zi))
    print(f"[test] filter+CMR numba vs numpy: max abs diff {np.abs(nbf-ref).max():.2e} uV "
          f"(signal max {np.abs(ref).max():.0f} uV)", flush=True)

    s0, n = cfg.samples(20_000), cfg.samples(5_000)
    raw = np.asarray(src.read(s0, n))
    dn.start_at(s0); A = SpikeBatch.concat([dn.push(raw), dn.flush()])
    dp.start_at(s0); B = SpikeBatch.concat([dp.push(raw), dp.flush()])
    oa, ob = np.lexsort((A.unit, A.frame)), np.lexsort((B.unit, B.frame))
    same_n = len(A) == len(B)
    eq_frames = same_n and np.array_equal(A.frame[oa], B.frame[ob])
    eq_units = same_n and np.array_equal(A.unit[oa], B.unit[ob])
    print(f"[test] numba vs numpy decode (5 s): events {len(A)} vs {len(B)} | frames equal {eq_frames} | "
          f"units equal {eq_units}", flush=True)
    if not (eq_frames and eq_units) and same_n:
        print(f"[test]   unit disagreements: {int((A.unit[oa] != B.unit[ob]).sum())}", flush=True)
    dn.start_at(s0); C_ = SpikeBatch.concat([dn.push(raw[i:i + 1024]) for i in range(0, n, 1024)] + [dn.flush()])
    oc = np.lexsort((C_.unit, C_.frame))
    print(f"[test] numba push(1024) == pull: {len(C_)==len(A) and np.array_equal(C_.frame[oc], A.frame[oa]) and np.array_equal(C_.unit[oc], A.unit[oa])}",
          flush=True)
    guard.check("equivalence")

    lat, stages = [], []
    n19 = cfg.samples(1_900)
    for i in range(14):
        start = cfg.samples(20_000) + i * cfg.samples(2_500)
        t = time.perf_counter()
        b = decode_window(dn, src, start, n19)
        lat.append(time.perf_counter() - t); stages.append(dict(dn.timing, events=len(b)))
    lat = np.array(lat[2:]); sd = pd.DataFrame(stages[2:]).median()
    print(f"[test] numba 1.9 s chunk latency: median {np.median(lat)*1e3:.0f} ms, max {lat.max()*1e3:.0f} ms | "
          + ", ".join(f"{k} {v*1e3:.0f} ms" for k, v in sd.items() if k != "events")
          + f" | events {sd['events']:.0f}", flush=True)
    json.dump(dict(latency_ms_median=float(np.median(lat) * 1e3), latency_ms_max=float(lat.max() * 1e3),
                   stages_ms={k: float(v * 1e3) for k, v in sd.items() if k != "events"}),
              open("results/decoder_latency_numba_quick.json", "w"), indent=2)
    print(f"[test] DONE | {guard.report()}", flush=True)


if __name__ == "__main__":
    main()
