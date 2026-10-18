#!/usr/bin/env python
"""Gate: does the quick initial sort reproduce the offline reference sort?

Reference = the lab's full-session Kilosort 2.5 units in jlh32.nwb (probe A),
restricted to the time span the initial sort saw.  Tested = the new sort.
Unit ids are arbitrary on both sides, so pairing is done by spike-time
coincidence (0.4 ms tolerance) and a one-to-one Hungarian assignment.

Independent evidence that a match is physically real: templates for BOTH
sorts are estimated from the same preprocessed cache with identical code, so
waveform correlation and peak-channel distance compare like with like.

Outputs (in results/):
  initial_sort_agreement_matrix.csv   unit x unit agreement scores, Hungarian-ordered
  initial_sort_confusion_matrix.csv   event-level counts incl. FN row / FP column
  sort_agreement_report.csv           per reference unit: match, counts, acc/prec/rec,
                                      waveform corr, peak-channel distance
  gate_summary.json                   headline numbers + pass/fail per criterion
  gate_templates.npz                  dense templates of both sorts (for the figure)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from streamsort.io import fsguard  # noqa: E402
fsguard.install()

import spikeinterface.full as si  # noqa: E402
import spikeinterface.comparison as sc  # noqa: E402
from streamsort.io import OpenEphysSession, load_nwb_units  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx  # noqa: E402
ROOT, REC, SCRATCH, OUT = ctx.ROOT, ctx.REC_DIR, ctx.SCRATCH, ctx.OUT

# Pass criteria, fixed before looking at the result (from the approved plan).
CRIT_FRAC_GOOD_MATCHED = 0.70   # of active NWB 'good' units matched at agreement > 0.5
CRIT_MEDIAN_ACCURACY = 0.60     # median accuracy over matched pairs
MIN_SPIKES_ACTIVE = 50          # a reference unit must fire this often in-segment to be matchable


def dense_templates(sorting, rec, n_jobs):
    """Mean waveform per unit on all channels, from the same cache, same code."""
    an = si.create_sorting_analyzer(sorting, rec, format="memory", sparse=False)
    an.compute("random_spikes", max_spikes_per_unit=300, seed=0)
    an.compute("templates", operators=["average"], pool_engine="thread",
               n_jobs=n_jobs, progress_bar=False)
    return an.get_extension("templates").get_data(operator="average")  # (u, t, ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seg600")
    ap.add_argument("--sorter", default="kilosort4")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--no-templates", action="store_true")
    ap.add_argument("--delta-ms", type=float, default=0.4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.tag == "seg600" else f"_{args.tag}"

    st = OpenEphysSession(REC).stream(ctx.STREAM)
    fs = st.effective_rate
    an_new = si.load_sorting_analyzer(str(SCRATCH / f"analyzer_{args.sorter}_{args.tag}"))
    rec = an_new.recording
    n_frames = an_new.get_num_samples()
    # Spike times on both sides are integer sample indices into the same
    # continuous.dat, which is what carries the timing.  The sampling-rate label
    # only converts delta_time to samples (0.4 ms -> 12.00 at the nominal
    # 30 kHz vs 11.99 at the fitted 29970.08 Hz), so use the recording's own
    # label for SpikeInterface consistency.  Seconds <-> samples conversion
    # always goes through the fitted map, never through this label.
    fs = rec.get_sampling_frequency()
    t_lo, t_hi = st.sample_to_time(0), st.sample_to_time(n_frames)

    # tested sorting: frames are already sample indices into continuous.dat
    new_dict = {int(u): an_new.sorting.get_unit_spike_train(u).astype(np.int64)
                for u in an_new.sorting.unit_ids}
    tested = si.NumpySorting.from_unit_dict([new_dict], sampling_frequency=fs)

    # reference: NWB seconds -> sample index via the fitted map (same clock)
    ref_df = load_nwb_units(ctx.NWB, probe=ctx.PROBE)
    ref_dict, ref_meta = {}, []
    for i, r in ref_df.iterrows():
        t = np.asarray(r["spike_times"], dtype=np.float64)
        t = t[(t >= t_lo) & (t < t_hi)]
        fr = st.time_to_sample(t)
        fr = fr[(fr >= 0) & (fr < n_frames)]
        uid = int(i)                      # row index as a stable integer id
        ref_dict[uid] = np.unique(fr)
        ref_meta.append(dict(ref_id=uid, nwb_unit_id=r["unit_id"], group=r["group"],
                             nwb_ch=int(r["ch"]), nwb_depth=float(r["depth"]),
                             nwb_amp=float(r["KSamplitude"]),
                             n_ref_in_segment=int(len(ref_dict[uid]))))
    ref_meta = pd.DataFrame(ref_meta).set_index("ref_id")
    # Units silent in the segment cannot be matched, and SpikeInterface 0.105
    # mis-types the label array of an empty unit (float instead of str), which
    # crashes compute_labels.  Leave them out of the comparison; they stay in
    # the report flagged as inactive.
    ref_dict = {u: t for u, t in ref_dict.items() if len(t) > 0}
    gt = si.NumpySorting.from_unit_dict([ref_dict], sampling_frequency=fs)

    cmp = sc.compare_sorter_to_ground_truth(
        gt, tested, gt_name="KS2.5_NWB", tested_name=f"{args.sorter}_{args.tag}",
        delta_time=args.delta_ms, match_score=0.5, exhaustive_gt=False,
        match_mode="hungarian", compute_labels=True)

    # --- unit-level agreement matrix, Hungarian-ordered ------------------
    agr = cmp.get_ordered_agreement_scores()
    agr.to_csv(OUT / f"initial_sort_agreement_matrix{sfx}.csv")
    # --- event-level confusion matrix ------------------------------------
    conf = cmp.get_confusion_matrix()
    conf.to_csv(OUT / f"initial_sort_confusion_matrix{sfx}.csv")

    perf = cmp.get_performance(method="by_unit")      # accuracy/recall/precision per GT unit
    match = cmp.hungarian_match_12                     # ref -> new (-1 if none)
    scores = cmp.agreement_scores
    rows = []
    for rid in ref_meta.index:
        if rid not in ref_dict:
            rows.append(dict(ref_id=rid, new_id=-1, agreement=0.0, n_matched=0,
                             n_miss=0, n_false_pos=0, accuracy=np.nan,
                             precision=np.nan, recall=np.nan))
            continue
        nid = int(match.get(rid, -1)) if rid in match.index else -1
        a = float(scores.at[rid, nid]) if nid != -1 else 0.0
        n_ref = int(ref_meta.at[rid, "n_ref_in_segment"])
        n_new = int(len(new_dict[nid])) if nid != -1 else 0
        # matched events: invert agreement = m / (n1 + n2 - m)
        m = int(round(a * (n_ref + n_new) / (1 + a))) if nid != -1 else 0
        rows.append(dict(ref_id=rid, new_id=nid, agreement=a,
                         n_matched=m, n_miss=n_ref - m, n_false_pos=n_new - m,
                         accuracy=float(perf.at[rid, "accuracy"]) if rid in perf.index else np.nan,
                         precision=float(perf.at[rid, "precision"]) if rid in perf.index else np.nan,
                         recall=float(perf.at[rid, "recall"]) if rid in perf.index else np.nan))
    rep = ref_meta.join(pd.DataFrame(rows).set_index("ref_id"))

    # --- independent waveform evidence -----------------------------------
    if not args.no_templates:
        active = rep[rep.n_ref_in_segment >= MIN_SPIKES_ACTIVE].index.tolist()
        gt_active = gt.select_units(active)
        T_ref = dense_templates(gt_active, rec, args.n_jobs)
        T_new = dense_templates(tested, rec, args.n_jobs)
        pos = rec.get_channel_locations()
        ref_idx = {u: k for k, u in enumerate(gt_active.unit_ids)}
        new_idx = {u: k for k, u in enumerate(tested.unit_ids)}
        corr, dist, pk_ref, pk_new = {}, {}, {}, {}
        for rid in active:
            tr = T_ref[ref_idx[rid]]
            pk_ref[rid] = int(np.argmax(np.ptp(tr, axis=0)))
            nid = rep.at[rid, "new_id"]
            if nid == -1:
                continue
            tn = T_new[new_idx[nid]]
            pk_new[rid] = int(np.argmax(np.ptp(tn, axis=0)))
            # correlation over channels where either template is substantial
            sup = (np.ptp(tr, 0) > 0.2 * np.ptp(tr, 0).max()) | (np.ptp(tn, 0) > 0.2 * np.ptp(tn, 0).max())
            a, b = tr[:, sup].ravel(), tn[:, sup].ravel()
            corr[rid] = float(np.corrcoef(a, b)[0, 1])
            dist[rid] = float(np.linalg.norm(pos[pk_ref[rid]] - pos[pk_new[rid]]))
        rep["waveform_corr"] = pd.Series(corr)
        rep["peak_ch_dist_um"] = pd.Series(dist)
        rep["ref_peak_ch"] = pd.Series(pk_ref)
        rep["ref_ptp_uV"] = pd.Series({r: float(np.ptp(T_ref[ref_idx[r]], 0).max() * st.bit_volts)
                                       for r in active})
        np.savez_compressed(OUT / f"gate_templates{sfx}.npz",
                            T_ref=T_ref, ref_ids=np.array(gt_active.unit_ids),
                            T_new=T_new, new_ids=np.array(tested.unit_ids),
                            channel_locations=pos, bit_volts=st.bit_volts)
    rep.to_csv(OUT / f"sort_agreement_report{sfx}.csv")

    # --- headline numbers -------------------------------------------------
    good = rep[rep.group == "good"]
    good_active = good[good.n_ref_in_segment >= MIN_SPIKES_ACTIVE]
    matched_pairs = rep[rep.agreement > 0.5]
    frac_good_active = float((good_active.agreement > 0.5).mean()) if len(good_active) else float("nan")
    frac_good_all = float((good.agreement > 0.5).mean())
    med_acc = float(matched_pairs.accuracy.median()) if len(matched_pairs) else float("nan")
    summary = {
        "tag": args.tag, "sorter": args.sorter, "delta_ms": args.delta_ms,
        "segment_s": [float(t_lo), float(t_hi)],
        "n_new_units": len(new_dict), "n_ref_units_probeA": len(rep),
        "n_ref_good": int(len(good)), "n_ref_good_active": int(len(good_active)),
        "n_ref_active_any": int((rep.n_ref_in_segment >= MIN_SPIKES_ACTIVE).sum()),
        "good_active_matched_gt0.5": int((good_active.agreement > 0.5).sum()),
        "good_active_matched_gt0.8": int((good_active.agreement > 0.8).sum()),
        "frac_good_active_matched_gt0.5": frac_good_active,
        "frac_good_active_matched_gt0.8": float((good_active.agreement > 0.8).mean()) if len(good_active) else float("nan"),
        "frac_good_all_matched_gt0.5": frac_good_all,
        "n_matched_pairs_gt0.5": int(len(matched_pairs)),
        "median_accuracy_matched": med_acc,
        "median_precision_matched": float(matched_pairs.precision.median()) if len(matched_pairs) else float("nan"),
        "median_recall_matched": float(matched_pairs.recall.median()) if len(matched_pairs) else float("nan"),
        "unit_categories": {k: int(v) for k, v in cmp.count_units_categories().items()},
        "criteria": {
            "frac_good_active_matched_gt0.5 >= %.2f" % CRIT_FRAC_GOOD_MATCHED: frac_good_active >= CRIT_FRAC_GOOD_MATCHED,
            "median_accuracy_matched > %.2f" % CRIT_MEDIAN_ACCURACY: med_acc > CRIT_MEDIAN_ACCURACY,
        },
    }
    if "waveform_corr" in rep:
        mp = rep[rep.agreement > 0.5]
        summary["median_waveform_corr_matched"] = float(mp.waveform_corr.median())
        summary["median_peak_ch_dist_um_matched"] = float(mp.peak_ch_dist_um.median())
        summary["frac_matched_peak_within_50um"] = float((mp.peak_ch_dist_um <= 50).mean())
    summary["PASS"] = bool(all(summary["criteria"].values()))
    (OUT / f"gate_summary{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
