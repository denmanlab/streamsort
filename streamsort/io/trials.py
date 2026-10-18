"""Stimulus trial table, reconciled against the hardware TTL record.

The NWB trial table is the authoritative description of *what* was delivered
(amplitude, polarity, contacts, pulse duration).  The NI-DAQ TTL line is the
authoritative record of *when*.  This module loads both and checks they agree
before either is used, because a silent mismatch between them would misplace
every analysis window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRIAL_FIELDS = (
    "start_time", "stop_time", "amplitude", "contact_negative",
    "contact_positive", "pulse_duration", "pulse_number", "run",
    "event_period", "train_period", "train_quantity",
)
TRIAL_STR_FIELDS = ("polarity", "shape", "contacts", "notes")


def load_nwb_trials(nwb_path) -> pd.DataFrame:
    """Read ``intervals/trials`` from an NWB file with h5py (no pynwb needed)."""
    import h5py

    with h5py.File(Path(nwb_path), "r") as f:
        tr = f["intervals/trials"]
        data = {k: tr[k][:] for k in TRIAL_FIELDS if k in tr}
        for k in TRIAL_STR_FIELDS:
            if k in tr:
                data[k] = [x.decode() if isinstance(x, bytes) else x
                           for x in tr[k][:]]
    df = pd.DataFrame(data)
    df.index.name = "trial"
    return df


def load_nwb_units(nwb_path, probe: str | None = None) -> pd.DataFrame:
    """Read the ``units`` table, returning spike times as per-unit arrays.

    ``spike_times`` in NWB is one ragged concatenation indexed by
    ``spike_times_index``; this unpacks it.
    """
    import h5py

    with h5py.File(Path(nwb_path), "r") as f:
        u = f["units"]
        idx = u["spike_times_index"][:]
        starts = np.concatenate([[0], idx[:-1]])
        rows = {
            "unit_id": [x.decode() if isinstance(x, bytes) else x
                        for x in u["unit_id"][:]],
            "probe": [x.decode() if isinstance(x, bytes) else x
                      for x in u["probe"][:]],
            "group": [x.decode() if isinstance(x, bytes) else x
                      for x in u["group"][:]],
            "ch": u["ch"][:], "depth": u["depth"][:],
            "xpos": u["xpos"][:], "ypos": u["ypos"][:],
            "firing_rate": u["firing_rate"][:],
            "no_spikes": u["no_spikes"][:],
            "KSamplitude": u["KSamplitude"][:],
            "KScontamination": u["KScontamination"][:],
        }
        df = pd.DataFrame(rows)
        df["_start"] = starts
        df["_stop"] = idx
        keep = df if probe is None else df[df["probe"] == probe]
        spikes = u["spike_times"]
        df = keep.copy()
        df["spike_times"] = [
            spikes[int(a):int(b)] for a, b in zip(df["_start"], df["_stop"])
        ]
    return df.drop(columns=["_start", "_stop"]).reset_index(drop=True)


def reconcile_with_ttl(trials: pd.DataFrame, ttl_rising: np.ndarray,
                       ttl_falling: np.ndarray,
                       tolerance_ms: float = 1.0) -> dict:
    """Match trial start times to TTL rising edges.

    Returns a report plus, in ``trials_out``, the trial table augmented with
    the matched TTL onset/offset times and the measured pulse width.  Extra
    TTL pulses not present in the trial table are reported, not discarded
    silently -- in this dataset there are 16 of them.
    """
    t = trials["start_time"].to_numpy(dtype=np.float64)
    order = np.argsort(ttl_rising)
    rising = np.asarray(ttl_rising, dtype=np.float64)[order]
    j = np.searchsorted(rising, t)
    j = np.clip(j, 1, len(rising) - 1)
    left, right = rising[j - 1], rising[j]
    pick = np.where(np.abs(t - left) <= np.abs(t - right), j - 1, j)
    dt = t - rising[pick]

    falling = np.sort(np.asarray(ttl_falling, dtype=np.float64))
    k = np.searchsorted(falling, rising[pick], side="left")
    k = np.clip(k, 0, len(falling) - 1)
    width_us = (falling[k] - rising[pick]) * 1e6

    out = trials.copy()
    out["ttl_onset"] = rising[pick]
    out["ttl_offset"] = falling[k]
    out["ttl_width_us"] = width_us
    out["ttl_index"] = pick

    matched = np.abs(dt) <= tolerance_ms * 1e-3
    unmatched_ttl = sorted(set(range(len(rising))) - set(pick.tolist()))
    report = {
        "n_trials": int(len(trials)),
        "n_ttl_rising": int(len(rising)),
        "n_matched": int(matched.sum()),
        "n_unmatched_trials": int((~matched).sum()),
        "n_extra_ttl": len(unmatched_ttl),
        "max_abs_dt_ms": float(np.abs(dt).max() * 1e3),
        "median_abs_dt_ms": float(np.median(np.abs(dt)) * 1e3),
        "duplicate_assignments": int(len(pick) - len(set(pick.tolist()))),
        "width_us_pct": [float(x) for x in np.percentile(width_us, [1, 50, 99])],
        "trials_out": out,
    }
    return report


def ttl_trials(ttl_rising, ttl_falling) -> "pd.DataFrame":
    """Trial table from TTL edges only (no NWB): one row per rising edge, offset = next falling edge."""
    up = np.sort(np.asarray(ttl_rising, float)); dn = np.sort(np.asarray(ttl_falling, float))
    j = np.searchsorted(dn, up, side="right"); off = np.where(j < len(dn), dn[np.minimum(j, len(dn) - 1)], np.nan)
    return pd.DataFrame(dict(ttl_onset=up, ttl_offset=off, pulse_width_us=(off - up) * 1e6))
