"""Shared configuration for streamsort.

The single most important invariant in this package: the offline stage that
builds templates and the online stage that matches them must see *identically*
processed signal.  Both read their parameters from one :class:`StreamConfig`
instance, which is serialised alongside the template bank so an online decoder
can refuse to run against templates fitted under different settings.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class StreamConfig:
    """Signal-processing parameters shared by the offline and online paths."""

    sample_rate: float = 29970.075737
    """Effective sample rate (Hz).

    Note this is *not* the nominal 30000 Hz: the Open Ephys sync correction
    yields an effective AP rate near 29970 Hz for this rig.  Using the nominal
    value accumulates ~5 s of error over an 84 min session.
    """

    # --- filtering -------------------------------------------------------
    highpass_hz: float = 300.0
    filter_order: int = 3
    common_reference: str = "median"  # "median" | "average" | "none"

    # --- channels --------------------------------------------------------
    n_channels_total: int = 384
    """Channels present in continuous.dat (the on-disk record width)."""
    n_channels_used: int = 200
    """Leading channels actually processed; matches the lab's use200 map."""

    # --- detection -------------------------------------------------------
    detect_threshold: float = 5.0
    """Detection threshold in robust standard deviations (negative-going)."""
    detect_sign: int = -1
    refractory_samples: int = 30
    """Per-channel dead time after a detection (~1 ms)."""
    dedup_radius_um: float = 50.0
    """Events within this radius and the dedup window collapse to one."""
    dedup_window_samples: int = 10

    # --- waveform extraction --------------------------------------------
    n_before: int = 30
    n_after: int = 52
    """Snippet spans n_before + n_after samples (82 total, matching KS2.5)."""
    sparsity_radius_um: float = 75.0

    # --- stimulus artifact -----------------------------------------------
    blank_pre_ms: float = 0.2
    """Blank this long *before* the TTL rising edge (artifact anticipation)."""
    blank_post_ms: float = 1.0
    """Blank until this long after the end of the stimulus pulse."""

    # --- analysis window --------------------------------------------------
    window_s: float = 1.9
    """Post-stimulus analysis window. Must be < the inter-trial interval."""
    bin_ms: float = 10.0
    smooth_sigma_ms: float = 20.0

    # --- scaling ----------------------------------------------------------
    bit_volts: float = 0.1949999928474426

    @property
    def n_samples_waveform(self) -> int:
        return self.n_before + self.n_after

    def samples(self, ms: float) -> int:
        """Convert milliseconds to whole samples at this config's rate."""
        return int(round(ms * 1e-3 * self.sample_rate))

    def fingerprint(self) -> str:
        """Stable hash of the signal-path parameters.

        Only fields that change the processed signal or the detected events
        participate, so re-binning for geometry does not invalidate templates.
        """
        signal_fields = (
            "sample_rate", "highpass_hz", "filter_order", "common_reference",
            "n_channels_total", "n_channels_used", "detect_threshold",
            "detect_sign", "refractory_samples", "n_before", "n_after",
            "bit_volts",
        )
        payload = json.dumps(
            {k: getattr(self, k) for k in signal_fields}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_fingerprint"] = self.fingerprint()
        return d

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path) -> "StreamConfig":
        d = json.loads(Path(path).read_text())
        d.pop("_fingerprint", None)
        return cls(**d)
