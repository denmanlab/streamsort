#!/usr/bin/env python
"""Offline initial sort with SpikeInterface -- the slow stage, run once.

This never runs in the latency path.  It consumes a leading segment of a
recording and emits the template bank the online decoder matches against.

Order of operations is deliberate and differs from the SpikeInterface default:
blanking happens *before* the highpass, not after.  A stimulus artifact that
reaches the filter rings for several milliseconds, and zeroing a window after
the fact leaves that ringing in the surrounding samples (measured: 22 uV
residual at 3 ms post-pulse versus 13 uV when the transient is removed first).

Usage
-----
    python scripts/fit_initial_sort.py --seconds 600 --sorter spykingcircus2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The scratch volume is a fuse-t NFS mount that fabricates ._* AppleDouble
# entries; SpikeInterface would try to load them as data.  Must run before
# spikeinterface is imported.
from streamsort.io import fsguard  # noqa: E402
fsguard.install()

# Keep BLAS from spawning a thread per core inside worker processes; on the
# Windows acquisition rig this contends with the acquisition thread.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Per-sorter settings.  Two principles apply to every backend:
#  * No motion correction.  The online matcher applies *fixed* templates to
#    raw streamed samples; a drift-corrected sort yields templates in a warped
#    coordinate frame the live stream never shares.  Drift is instead measured
#    explicitly at validation (early vs late session).
#  * No internal re-filtering.  The cache is already highpassed and referenced
#    in exactly the order the online decoder uses; a second filter would put
#    the sorter's view out of step with the data templates are extracted from.
#    Internal whitening is left on -- it only aids clustering, and the final
#    templates are re-extracted from the unwhitened cache by the analyzer.
SORTER_OVERRIDES = {
    # Primary backend.  KS4 keeps its own highpass and whitening because it
    # needs whitened data to cluster; CAR is off because the cache is already
    # median-referenced.  Templates are re-extracted from the unwhitened cache.
    "kilosort4": dict(do_correction=False, nblocks=0, do_CAR=False,
                      torch_device="cpu", keep_good_only=False,
                      delete_recording_dat=True),
    # DEFERRED: the two backends below fail in the sandbox because
    # SpikeInterface hard-codes ProcessPoolExecutor in its clustering code.
    "spykingcircus2": dict(apply_motion_correction=False,
                           apply_preprocessing=False,
                           apply_whitening=True),
    "tridesclous2": dict(apply_motion_correction=False,
                         apply_preprocessing=False),
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx  # noqa: E402  (REC / PROBE from the environment)
ROOT, STREAM, SCRATCH = ctx.ROOT, ctx.SI_STREAM, ctx.SCRATCH



def rmtree_safe(path: Path) -> None:
    """Remove a directory, tolerating the AppleDouble sidecars the NFS scratch
    mount synthesises (``._foo``), which appear in listings but cannot be
    unlinked and otherwise abort shutil.rmtree."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        for sub in sorted(path.rglob("*"), reverse=True):
            try:
                sub.unlink() if sub.is_file() else sub.rmdir()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass


def build_blank_periods(src, trials, seg_end_frame, cfg, artifact_mod):
    """Data-derived blank windows for every trial inside the segment.

    The TTL index is off by -0.40 to +0.60 ms against the true artifact onset,
    so each window is located by finding the clipping in the raw data.
    """
    sub = trials[trials["ttl_onset"] < src.sample_to_time(seg_end_frame)]
    periods, missed = [], 0
    npre = cfg.samples(3.0)
    for _, row in sub.iterrows():
        on = int(src.time_to_sample(row["ttl_onset"]))
        raw = np.asarray(src.read(on - npre, npre + cfg.samples(12)))
        det = artifact_mod.detect_artifact_onset(raw, npre, cfg.samples(2.0))
        if det is None:
            missed += 1
            det = npre
        span = artifact_mod.clipped_span(raw, det, cfg.samples(10))
        a = on - npre + det - cfg.samples(0.2)
        b = on - npre + det + span + cfg.samples(0.5)
        periods.append([int(a), int(b)])
    arr = (np.asarray(periods, dtype=np.int64) if periods
           else np.zeros((0, 2), dtype=np.int64))
    return arr, len(sub), missed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--sorter", default="kilosort4")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume-analyzer", action="store_true",
                    help="reuse an existing sort + analyzer; only (re)compute metrics")
    args = ap.parse_args()
    tag = args.tag or f"seg{int(args.seconds)}"

    import spikeinterface.full as si
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    import spikeinterface.sorters as ss

    # The sandbox forbids os.sysconf("SC_SEM_NSEMS_MAX"), so ProcessPoolExecutor
    # cannot start.  A thread pool is fine here: the heavy inner loops are numpy
    # and scipy calls that release the GIL.
    si.set_global_job_kwargs(pool_engine="thread", n_jobs=args.n_jobs,
                             chunk_duration="2s", progress_bar=False)

    from streamsort.config import StreamConfig
    from streamsort.io import (OpenEphysSession, FileStreamSource)
    import streamsort.preprocess.artifact as artifact_mod

    import dataclasses
    sess = OpenEphysSession(ctx.REC_DIR)
    stream = sess.stream(ctx.STREAM)
    cfg = dataclasses.replace(StreamConfig(sample_rate=stream.effective_rate, bit_volts=stream.bit_volts),
                              n_channels_used=ctx.NCH)
    print(f"[fit] {ctx.describe()}", flush=True)
    src = FileStreamSource(stream, n_channels_used=cfg.n_channels_used)
    trials = ctx.load_trials(sess)

    seg_end = int(round(args.seconds / stream.slope))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache = SCRATCH / f"preproc_{tag}"
    report = {"tag": tag, "seconds": args.seconds, "sorter": args.sorter,
              "effective_rate_hz": stream.effective_rate,
              "si_nominal_rate_hz": 30000.0,
              "rate_note": "SpikeInterface reports the nominal 30 kHz; convert "
                           "sorter frames to global time with the fitted map, "
                           "not with the SI sampling frequency"}

    # ---- preprocess + cache -------------------------------------------
    if args.reuse_cache and cache.exists():
        rec = si.load(str(cache))
        print(f"[fit] reusing cache {cache}", flush=True)
    else:
        rec = se.read_openephys(ROOT, stream_name=STREAM)
        rec = rec.select_channels(rec.channel_ids[:cfg.n_channels_used])
        rec = rec.frame_slice(0, seg_end)
        periods, n_tr, missed = build_blank_periods(
            src, trials, seg_end, cfg, artifact_mod)
        from spikeinterface.core.base import base_period_dtype
        parr = np.zeros(len(periods), dtype=base_period_dtype)
        parr["segment_index"] = 0
        parr["start_sample_index"] = periods[:, 0]
        parr["end_sample_index"] = periods[:, 1]
        blank_frac = float((periods[:, 1] - periods[:, 0]).sum() / seg_end)
        report.update(n_trials_in_segment=int(n_tr), n_onsets_missed=int(missed),
                      blanked_fraction=blank_frac)
        print(f"[fit] {n_tr} trials, {missed} onsets undetected, "
              f"{blank_frac*100:.3f}% of samples blanked", flush=True)

        rec = spre.astype(rec, np.float32)
        rec = spre.center(rec, mode="median")
        rec = spre.silence_periods(rec, periods=parr, mode="noise", seed=0)
        rec = spre.highpass_filter(rec, freq_min=cfg.highpass_hz,
                                   filter_order=cfg.filter_order)
        rec = spre.common_reference(rec, operator="median", reference="global")

        rmtree_safe(cache)
        t0 = time.perf_counter()
        rec = rec.save(folder=str(cache), overwrite=True,
                       pool_engine="thread", n_jobs=args.n_jobs,
                       chunk_duration="2s", progress_bar=False)
        report["cache_seconds"] = time.perf_counter() - t0
        print(f"[fit] cached in {report['cache_seconds']:.1f} s -> {cache}",
              flush=True)

    sort_dir = SCRATCH / f"sort_{args.sorter}_{tag}"
    an_dir = SCRATCH / f"analyzer_{args.sorter}_{tag}"
    if args.resume_analyzer and an_dir.exists():
        analyzer = si.load_sorting_analyzer(str(an_dir))
        print(f"[fit] resuming analyzer {an_dir}", flush=True)
        return _finish(analyzer, args, tag, report, an_dir)

    # ---- sort -----------------------------------------------------------
    rmtree_safe(sort_dir)
    t0 = time.perf_counter()
    sorter_params = SORTER_OVERRIDES.get(args.sorter, {}).copy()
    defaults = ss.get_default_sorter_params(args.sorter)
    if "job_kwargs" in defaults:
        sorter_params["job_kwargs"] = dict(pool_engine="thread",
                                           n_jobs=args.n_jobs)
    elif "pool_engine" in defaults:
        sorter_params.update(pool_engine="thread", n_jobs=args.n_jobs)
    report["sorter_params"] = sorter_params
    sorting = ss.run_sorter(args.sorter, rec, folder=str(sort_dir),
                            remove_existing_folder=True, verbose=True,
                            **sorter_params)
    report["sort_seconds"] = time.perf_counter() - t0
    report["n_units_raw"] = int(len(sorting.unit_ids))
    print(f"[fit] {args.sorter}: {report['n_units_raw']} units in "
          f"{report['sort_seconds']:.1f} s", flush=True)

    # ---- analyzer + templates -------------------------------------------
    rmtree_safe(an_dir)
    analyzer = si.create_sorting_analyzer(
        sorting, rec, format="binary_folder", folder=str(an_dir),
        sparse=True, method="radius", radius_um=cfg.sparsity_radius_um,
    )
    analyzer.compute(
        ["random_spikes", "waveforms", "templates", "noise_levels",
         "spike_amplitudes", "unit_locations", "template_similarity"],
        pool_engine="thread", n_jobs=args.n_jobs, progress_bar=False,
    )
    return _finish(analyzer, args, tag, report, an_dir)


def _finish(analyzer, args, tag, report, an_dir):
    qm = analyzer.compute("quality_metrics",
                          metric_names=["snr", "firing_rate",
                                        "isi_violation",
                                        "presence_ratio", "amplitude_cutoff"],
                          pool_engine="thread", n_jobs=args.n_jobs, progress_bar=False)
    metrics = qm.get_data()
    metrics.to_csv(SCRATCH / f"quality_metrics_{args.sorter}_{tag}.csv")
    report["analyzer"] = str(an_dir)
    print("[fit] analyzer extensions computed", flush=True)
    print(metrics.describe().round(3).to_string(), flush=True)

    (SCRATCH / f"fit_report_{args.sorter}_{tag}.json").write_text(
        json.dumps(report, indent=2, default=float))
    print("[fit] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
