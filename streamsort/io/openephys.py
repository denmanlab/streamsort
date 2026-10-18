"""Adapter for an Open Ephys binary-format recording.

Parses ``structure.oebin``, the TTL event folders and ``sync_messages.txt``,
and resolves the two clock problems that otherwise corrupt stimulus-locked
analysis:

1. The AP stream does not run at its nominal rate.  Open Ephys writes a
   sync-corrected ``timestamps.npy`` for each continuous stream; fitting global
   seconds against sample index recovers the *effective* rate, which for this
   rig is ~29970.08 Hz rather than 30000 Hz.  Over an 84 min session the
   nominal rate is wrong by about 5 s.

2. Event ``timestamps`` are already expressed on a common clock across
   streams, but that correction is estimated, not exact.  :meth:`barcode_check`
   quantifies the residual against the shared 1 Hz barcode line so the
   uncertainty on window placement is measured rather than assumed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ContinuousStream:
    """One continuous data stream on disk."""

    folder_name: str
    path: Path
    sample_rate: float          # nominal, from the oebin
    num_channels: int
    bit_volts: float
    n_samples: int
    # fitted map: global_seconds = slope * sample_index + intercept
    slope: float
    intercept: float
    fit_residual_us_rms: float
    fit_residual_us_max: float

    @property
    def effective_rate(self) -> float:
        """Sample rate implied by the fitted time map (Hz)."""
        return 1.0 / self.slope

    @property
    def t_start(self) -> float:
        return self.intercept

    @property
    def t_end(self) -> float:
        return self.sample_to_time(self.n_samples - 1)

    # -- time <-> sample ----------------------------------------------------
    # Two maps are provided, and the difference matters.
    #
    # EXACT (default): inverts the sync-corrected ``timestamps.npy`` that Open
    # Ephys writes beside continuous.dat.  Every global time recorded anywhere
    # in the session (TTL events on other streams, NWB spike times) was put on
    # that clock, so inverting it recovers the true sample to within half a
    # sample (measured: 15 us max on jlh32 NWB spike times).
    #
    # LINEAR: a single least-squares line through timestamps.npy.  Convenient
    # (no file access) but the sync correction is piecewise, so the line is
    # off by up to ~13 samples (-334..+434 us measured on jlh32).  That error
    # is the same size as a 0.4 ms spike-matching tolerance and must never be
    # used for spike-level comparison.  It remains fine for coarse work such as
    # durations and seeking to within a millisecond.
    def _timestamps(self):
        ts = getattr(self, "_ts_mm", None)
        if ts is None:
            ts = np.load(self.path / "timestamps.npy", mmap_mode="r")
            object.__setattr__(self, "_ts_mm", ts)
        return ts

    def sample_to_time(self, i):
        """Exact global time of sample index ``i`` (reads timestamps.npy)."""
        i = np.asarray(i, dtype=np.int64)
        return np.asarray(self._timestamps()[np.clip(i, 0, self.n_samples - 1)],
                          dtype=np.float64)

    def time_to_sample(self, t):
        """Nearest sample index for a global time, by inverting timestamps.npy."""
        t = np.asarray(t, dtype=np.float64)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        ts = self._timestamps()
        # Vectorised: sort the queries, then for each block of queries whose
        # linear-map guesses span <= BLOCK samples, load only that span of
        # timestamps.npy (plus padding for the <=13-sample linear error) and
        # resolve the whole block with one searchsorted.
        order = np.argsort(t, kind="stable")
        ts_sorted_q = t[order]
        guess = self.time_to_sample_linear(ts_sorted_q)
        pad, BLOCK = 256, 4_000_000
        out_sorted = np.empty(t.shape, dtype=np.int64)
        i = 0
        n = len(ts_sorted_q)
        while i < n:
            g0 = guess[i]
            j = int(np.searchsorted(guess, g0 + BLOCK, side="right"))
            j = max(j, i + 1)
            lo = int(np.clip(guess[i] - pad, 0, self.n_samples - 1))
            hi = int(np.clip(guess[j - 1] + pad + 1, lo + 1, self.n_samples))
            w = np.asarray(ts[lo:hi], dtype=np.float64)
            q = ts_sorted_q[i:j]
            k = np.searchsorted(w, q)
            k = np.clip(k, 1, len(w) - 1) if len(w) > 1 else np.zeros_like(k)
            left_closer = np.abs(q - w[k - 1]) <= np.abs(q - w[k]) if len(w) > 1 else np.ones_like(k, bool)
            idx = np.where(left_closer, k - 1, k) + lo
            # queries outside the loaded span (should not happen given pad)
            bad = (q < w[0]) | (q > w[-1])
            if bad.any():
                full = np.searchsorted(ts, q[bad])
                idx[bad] = np.clip(full, 0, self.n_samples - 1)
            out_sorted[i:j] = idx
            i = j
        out = np.empty_like(out_sorted)
        out[order] = out_sorted
        return out[0] if scalar else out

    def sample_to_time_linear(self, i):
        return self.slope * np.asarray(i, dtype=np.float64) + self.intercept

    def time_to_sample_linear(self, t):
        """Approximate sample index from the linear fit (error up to ~13 samples)."""
        idx = (np.asarray(t, dtype=np.float64) - self.intercept) / self.slope
        return np.rint(idx).astype(np.int64)

    @property
    def dat_path(self) -> Path:
        return self.path / "continuous.dat"


class OpenEphysSession:
    """A single Open Ephys ``recordingN`` folder."""

    def __init__(self, recording_dir, time_map_samples: int = 20001):
        self.root = Path(recording_dir)
        if not (self.root / "structure.oebin").exists():
            raise FileNotFoundError(
                f"no structure.oebin under {self.root}; point at a recordingN folder"
            )
        self._oebin = json.loads((self.root / "structure.oebin").read_text())
        self._time_map_samples = time_map_samples
        self._streams: dict[str, ContinuousStream] = {}
        self._ttl_cache: dict[str, dict] = {}

    # -- continuous streams ------------------------------------------------
    @property
    def stream_names(self) -> list[str]:
        return [c["folder_name"].rstrip("/") for c in self._oebin["continuous"]]

    def stream(self, name: str) -> ContinuousStream:
        """Load (and cache) a continuous stream, fitting its time map."""
        name = name.rstrip("/")
        if name in self._streams:
            return self._streams[name]
        entry = next(
            (c for c in self._oebin["continuous"]
             if c["folder_name"].rstrip("/") == name), None
        )
        if entry is None:
            raise KeyError(f"{name!r} not in {self.stream_names}")
        path = self.root / "continuous" / name
        n_ch = int(entry["num_channels"])
        n_samples = (path / "continuous.dat").stat().st_size // (2 * n_ch)

        ts = np.load(path / "timestamps.npy", mmap_mode="r")
        if len(ts) != n_samples:
            raise ValueError(
                f"{name}: timestamps has {len(ts)} entries but continuous.dat "
                f"implies {n_samples} samples"
            )
        k = min(self._time_map_samples, len(ts))
        idx = np.linspace(0, len(ts) - 1, k).astype(np.int64)
        tsel = np.asarray(ts[idx], dtype=np.float64)
        slope, intercept = np.polyfit(idx.astype(np.float64), tsel, 1)
        resid = (tsel - (slope * idx + intercept)) * 1e6

        bit_volts = float(entry["channels"][0]["bit_volts"])
        st = ContinuousStream(
            folder_name=name, path=path,
            sample_rate=float(entry["sample_rate"]), num_channels=n_ch,
            bit_volts=bit_volts, n_samples=int(n_samples),
            slope=float(slope), intercept=float(intercept),
            fit_residual_us_rms=float(resid.std()),
            fit_residual_us_max=float(np.abs(resid).max()),
        )
        self._streams[name] = st
        return st

    # -- events ------------------------------------------------------------
    def ttl(self, source: str) -> dict:
        """Load a TTL event folder: sample_numbers, states, timestamps."""
        source = source.rstrip("/")
        if source in self._ttl_cache:
            return self._ttl_cache[source]
        d = self.root / "events" / source / "TTL"
        out = {k: np.load(d / f"{k}.npy")
               for k in ("sample_numbers", "states", "timestamps")}
        self._ttl_cache[source] = out
        return out

    def ttl_edges(self, source: str, line: int, rising: bool = True) -> np.ndarray:
        """Global-clock times of edges on one TTL line.

        Open Ephys encodes line N as state ``+N`` for rising and ``-N`` for
        falling.
        """
        t = self.ttl(source)
        state = line if rising else -line
        return t["timestamps"][t["states"] == state]

    def barcode_check(self, source_a: str, line_a: int,
                      source_b: str, line_b: int) -> dict:
        """Residual between a sync line seen on two streams.

        Returns matched edge times and the residual in milliseconds.  A
        *bounded, non-drifting* residual means the streams are already on a
        common clock and event timestamps can be compared directly.
        """
        a = self.ttl_edges(source_a, line_a)
        b = self.ttl_edges(source_b, line_b)
        n = min(len(a), len(b))
        resid_ms = (a[:n] - b[:n]) * 1e3
        drift = np.polyfit(a[:n], resid_ms, 1)[0] if n > 2 else np.nan
        return {
            "n_a": int(len(a)), "n_b": int(len(b)), "n_matched": int(n),
            "t": a[:n], "residual_ms": resid_ms,
            "mean_ms": float(resid_ms.mean()), "sd_ms": float(resid_ms.std()),
            "max_abs_ms": float(np.abs(resid_ms).max()),
            "drift_ms_per_s": float(drift),
        }

    def sync_messages(self) -> str:
        p = self.root / "sync_messages.txt"
        return p.read_text() if p.exists() else ""
