#!/usr/bin/env python
"""Evaluate the whitened online decoder against Kilosort 4's own spikes (in-sample).

1. estimate whitening from stim-free data (60-80 s) and save it
2. push (1024-sample buffers) vs pull equivalence on 5 s
3. accuracy over detection threshold x energy acceptance on 20-40 s:
   pooled correct / neighbour / missed (spatially constrained) and
   per-unit precision / recall / F1 (label + time within 0.4 ms)
4. latency of pull-mode 1.9 s chunks, per stage

Memory-bounded (MemGuard 3.5 GB; user limit < 4 GB), 2 BLAS threads.
"""
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
from streamsort.online import (OnlineDecoder, DecoderParams, SpikeBatch,  # noqa: E402
                               estimate_whitening, Whitening, decode_window)

BUDGET_GB = 3.5
ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
KSOUT = "/Users/danieldenman/stim_scratch/jlh32_probeA/sort_kilosort4_seg600/sorter_output"


def nearest_match(a, b, tol):
    """bool per element of a: is there an element of sorted b within tol."""
    if len(b) == 0 or len(a) == 0:
        return np.zeros(len(a), bool)
    k = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    if len(b) == 1:
        return np.abs(a - b[0]) <= tol
    return np.minimum(np.abs(a - b[k - 1]), np.abs(a - b[k])) <= tol


def main():
    preflight(BUDGET_GB, margin_gb=0.0, strict=False)
    guard = MemGuard(BUDGET_GB, "eval_decoder")
    st = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1").stream("Neuropix-PXI-104.ProbeA-AP")
    cfg0 = StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts)
    src = FileStreamSource(st, 200)
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    rep = pd.read_csv("results/sorting_report.csv")
    usable = rep.usable.to_numpy(bool)
    pos = bank.channel_locations.astype(float)
    uidx = {int(u): k for k, u in enumerate(bank.unit_ids)}

    # 1. whitening ---------------------------------------------------------
    wpath = Path("results/whitening_probeA.npz")
    t = time.perf_counter()
    wh = estimate_whitening(src, cfg0, pos, start=cfg0.samples(60_000), n=cfg0.samples(20_000), k=32)
    wh.save(wpath)
    print(f"[eval] whitening estimated in {time.perf_counter()-t:.1f}s | noise sd median {np.median(wh.noise_sd):.3f} "
          f"| {guard.report()}", flush=True)

    # truth
    st_all = np.load(KSOUT + "/spike_times.npy", mmap_mode="r").ravel()
    cl_all = np.load(KSOUT + "/spike_clusters.npy", mmap_mode="r").ravel()

    # 2. push vs pull ------------------------------------------------------
    dec = OnlineDecoder(bank, cfg0, wh, DecoderParams(), units_mask=usable)
    s0, n = cfg0.samples(20_000), cfg0.samples(5_000)
    raw = np.asarray(src.read(s0, n))
    dec.start_at(s0); A = SpikeBatch.concat([dec.push(raw), dec.flush()])
    dec.start_at(s0); B = SpikeBatch.concat([dec.push(raw[i:i + 1024]) for i in range(0, n, 1024)] + [dec.flush()])
    oa, ob = np.lexsort((A.unit, A.frame)), np.lexsort((B.unit, B.frame))
    same = len(A) == len(B) and np.array_equal(A.frame[oa], B.frame[ob]) and np.array_equal(A.unit[oa], B.unit[ob])
    print(f"[eval] push==pull: {same} ({len(A)} events) | {guard.report()}", flush=True)
    del raw, A, B
    guard.check("push/pull")

    # 3. accuracy ------------------------------------------------------------
    S0, N = cfg0.samples(20_000), cfg0.samples(20_000)
    lo, hi = S0 + 300, S0 + N - 300
    sel = (st_all >= lo) & (st_all < hi)
    tw = np.asarray(st_all[sel]).astype(np.int64); cw = np.asarray(cl_all[sel]).astype(np.int64)
    use_ids = bank.unit_ids[usable]
    truth = {int(u): np.sort(tw[cw == u]) for u in use_ids}
    rows, per_rows = [], []
    CH = cfg0.samples(2_000)
    for thr in (4.0, 4.5, 5.0):
        cfg = dataclasses.replace(cfg0, detect_threshold=thr)
        dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=0.0), units_mask=usable, check=False)
        dec.start_at(S0)
        outs = [dec.push(np.asarray(src.read(S0 + i, min(CH, N - i)))) for i in range(0, N, CH)] + [dec.flush()]
        out = SpikeBatch.concat(outs)
        out = out.select((out.frame >= lo) & (out.frame < hi))
        guard.check(f"decode thr {thr}")
        for me in (25.0, 50.0, 100.0, 200.0):
            lab = np.where(out.score >= me, out.unit, -1)
            o = np.argsort(out.frame); F, L, PC = out.frame[o], lab[o], out.peak_channel[o]
            cnt = dict(correct=0, neighbour=0, missed=0); tp = 0; ndec = 0; nks = 0
            for u in use_ids:
                tr = truth[int(u)]; nks += len(tr)
                de = np.sort(F[L == u]); ndec += len(de)
                m_u = int(nearest_match(tr, de, 12).sum()); tp += m_u
                p = pos[bank.peak_channel[uidx[int(u)]]]
                near = np.linalg.norm(pos[PC] - p, axis=1) <= 75
                Fu, Lu = F[near], L[near]
                if len(tr) == 0:
                    continue
                if len(Fu) == 0:
                    cnt["missed"] += len(tr)
                else:
                    k = np.clip(np.searchsorted(Fu, tr), 1, max(len(Fu) - 1, 1)); k2 = np.minimum(k, len(Fu) - 1)
                    j = np.where(np.abs(tr - Fu[k - 1]) <= np.abs(tr - Fu[k2]), k - 1, k2)
                    hit = np.abs(tr - Fu[j]) <= 12
                    cnt["missed"] += int((~hit).sum())
                    cnt["correct"] += int((hit & (Lu[j] == u)).sum())
                    cnt["neighbour"] += int((hit & (Lu[j] != u)).sum())
                if me == 50.0:
                    per_rows.append(dict(thr=thr, unit=int(u), n_ks=len(tr), n_dec=len(de), tp=m_u))
            tot = sum(cnt.values())
            rows.append(dict(thr=thr, min_energy=me, events_per_s=len(out) / 20.0,
                             assigned_per_s=float((lab >= 0).sum()) / 20.0,
                             correct=cnt["correct"] / tot, neighbour=cnt["neighbour"] / tot,
                             missed=cnt["missed"] / tot,
                             precision=tp / max(ndec, 1), recall=tp / max(nks, 1)))
            r = rows[-1]
            print(f"[eval] thr {thr:.1f} E>={me:5.0f}: assigned/s {r['assigned_per_s']:6.0f} | recall {r['recall']:.3f} "
                  f"precision {r['precision']:.3f} | correct {r['correct']:.3f} neighbour {r['neighbour']:.3f} "
                  f"missed {r['missed']:.3f}", flush=True)
        del out, outs
    pd.DataFrame(rows).to_csv("results/decoder_accuracy_sweep.csv", index=False)
    per = pd.DataFrame(per_rows)
    per["precision"] = per.tp / per.n_dec.clip(lower=1)
    per["recall"] = per.tp / per.n_ks.clip(lower=1)
    per["f1"] = 2 * per.tp / (per.n_dec + per.n_ks).clip(lower=1)
    per.to_csv("results/decoder_per_unit_insample.csv", index=False)
    for thr, g in per.groupby("thr"):
        g = g[g.n_ks >= 20]
        print(f"[eval] per-unit (E>=50, thr {thr}): units {len(g)} | F1 median {g.f1.median():.2f} | "
              f"F1>=0.8: {(g.f1>=0.8).sum()} | F1>=0.9: {(g.f1>=0.9).sum()}", flush=True)

    # 4. latency ----------------------------------------------------------------
    cfg = dataclasses.replace(cfg0, detect_threshold=4.5)
    dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False)
    lat, stages = [], []
    n19 = cfg.samples(1_900)
    for i in range(12):
        start = cfg.samples(20_000) + i * cfg.samples(2_500)
        t = time.perf_counter()
        b = decode_window(dec, src, start, n19)
        lat.append(time.perf_counter() - t); stages.append(dict(dec.timing, events=len(b)))
    lat = np.array(lat[2:])                    # drop warm-up chunks
    sd = pd.DataFrame(stages[2:]).median()
    print(f"[eval] 1.9 s chunk latency: median {np.median(lat)*1e3:.0f} ms, max {lat.max()*1e3:.0f} ms | stages (median s): "
          + ", ".join(f"{k} {v:.3f}" for k, v in sd.items() if k != 'events') + f" | events {sd['events']:.0f}", flush=True)
    json.dump(dict(latency_ms_median=float(np.median(lat) * 1e3), latency_ms_max=float(lat.max() * 1e3),
                   stages_s=sd.to_dict()), open("results/decoder_latency_quick.json", "w"), indent=2)
    print(f"[eval] DONE | {guard.report()}", flush=True)


if __name__ == "__main__":
    main()
