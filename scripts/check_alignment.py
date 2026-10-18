#!/usr/bin/env python
"""Sanity-check the clock link between a new sort and the NWB reference sort.

Before any agreement number means anything, the two spike trains must be on the
same clock.  This finds, for the strongest few unit pairs, the lag that
maximises coincident spikes.  A lag near zero confirms the sample->time map; a
consistent non-zero lag means a conversion bug that would otherwise masquerade
as poor sorting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamsort.io import fsguard  # noqa: E402
fsguard.install()

import spikeinterface.full as si  # noqa: E402
from streamsort.io import OpenEphysSession, load_nwb_units  # noqa: E402

ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
REC = f"{ROOT}/Record Node 105/experiment1/recording1"
SCRATCH = Path("/Users/danieldenman/stim_scratch/jlh32_probeA")


def main(tag: str = "smoke150", sorter: str = "kilosort4") -> None:
    st = OpenEphysSession(REC).stream("Neuropix-PXI-104.ProbeA-AP")
    an = si.load_sorting_analyzer(str(SCRATCH / f"analyzer_{sorter}_{tag}"))
    sorting = an.sorting
    n_frames = an.get_num_samples()
    t_lo, t_hi = st.sample_to_time(0), st.sample_to_time(n_frames)
    print(f"segment {t_lo:.3f}..{t_hi:.3f} s ({n_frames} frames)")

    new = {u: st.sample_to_time(sorting.get_unit_spike_train(u)) for u in sorting.unit_ids}
    ref_df = load_nwb_units(f"{ROOT}/jlh32.nwb", probe="A")
    ref = {}
    for _, r in ref_df.iterrows():
        t = np.asarray(r["spike_times"])
        t = t[(t >= t_lo) & (t < t_hi)]
        if len(t) >= 50:
            ref[r["unit_id"]] = t
    print(f"new units {len(new)} | NWB probe-A units with >=50 spikes in segment {len(ref)}")
    nwb_all = np.concatenate([np.asarray(x) for x in ref_df["spike_times"]])
    print(f"NWB probe-A spike times overall: {nwb_all.min():.3f}..{nwb_all.max():.3f} s")

    # coarse search: for each ref unit, best new unit by coincidences at +-0.5 ms,
    # scanning lags -20..+20 ms so a constant offset is detectable
    lags = np.arange(-20.0, 20.01, 0.1) * 1e-3
    tol = 0.5e-3
    best = []
    new_items = list(new.items())
    for rid, rt in list(ref.items()):
        for nid, nt in new_items:
            # quick prefilter at zero lag with a wide window
            j = np.searchsorted(nt, rt)
            j = np.clip(j, 1, len(nt) - 1) if len(nt) > 1 else np.zeros_like(j)
            if len(nt) < 2:
                continue
            d = np.minimum(np.abs(rt - nt[j - 1]), np.abs(rt - nt[j]))
            if (d < 25e-3).mean() < 0.3:
                continue
            counts = []
            for L in lags:
                s = rt + L
                k = np.clip(np.searchsorted(nt, s), 1, len(nt) - 1)
                dd = np.minimum(np.abs(s - nt[k - 1]), np.abs(s - nt[k]))
                counts.append((dd <= tol).sum())
            counts = np.asarray(counts)
            i = int(counts.argmax())
            score = counts[i] / (len(rt) + len(nt) - counts[i])
            best.append((score, rid, nid, lags[i] * 1e3, counts[i], len(rt), len(nt)))
    best.sort(reverse=True)
    print("\ntop pairs (agreement, ref, new, best lag ms, matched, n_ref, n_new):")
    for b in best[:15]:
        print("  %.3f  ref=%-10s new=%-5s lag=%+5.1f ms  %5d  %5d  %5d" % b)
    if best:
        top = [b for b in best[:15] if b[0] > 0.3]
        L = np.array([b[3] for b in top])
        print(f"\nlag over pairs with agreement>0.3 (n={len(top)}): "
              f"median {np.median(L) if len(L) else float('nan'):+.2f} ms, "
              f"range {L.min() if len(L) else float('nan'):+.1f}..{L.max() if len(L) else float('nan'):+.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
