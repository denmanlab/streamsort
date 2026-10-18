"""Causal preprocessing with state carried across chunk boundaries.

The offline sort and the online decoder must produce the same numbers from the
same samples.  The only way to guarantee that while processing a stream in
small pieces is a *causal* filter whose internal state persists between calls --
``scipy.signal.sosfilt`` with an explicit ``zi``.  A zero-phase filter
(``sosfiltfilt``) cannot be used online at all: it reads backwards in time.

Operation order is fixed and matches the SpikeInterface chain used offline:
scale to microvolts, highpass, then common reference.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ..config import StreamConfig


class CausalPreprocessor:
    """Stateful highpass + common-reference for streamed int16 data.

    Parameters
    ----------
    cfg : StreamConfig
        Supplies the filter design, reference operator and microvolt scaling.
    n_channels : int
        Channel count of the buffers that will be pushed.
    """

    def __init__(self, cfg: StreamConfig, n_channels: int):
        self.cfg = cfg
        self.n_channels = int(n_channels)
        self.sos = butter(
            cfg.filter_order, cfg.highpass_hz / (cfg.sample_rate / 2.0),
            btype="highpass", output="sos",
        ).astype(np.float64)
        self._zi_unit = sosfilt_zi(self.sos)          # (n_sections, 2)
        self.zi = np.zeros((self.sos.shape[0], 2, self.n_channels), dtype=np.float64)
        self._primed = False

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        """Clear filter state. The next chunk will carry a startup transient."""
        self.zi[:] = 0.0
        self._primed = False

    def prime(self, first_sample: np.ndarray) -> None:
        """Initialise state to the steady-state response for a constant input.

        Cuts the startup transient from tens of milliseconds to essentially
        nothing, which matters because an un-primed filter's first samples are
        dominated by the DC offset of the probe (hundreds of microvolts).
        """
        x0 = np.asarray(first_sample, dtype=np.float64).reshape(1, -1)
        self.zi = self._zi_unit[:, :, None] * x0[None, :, :]
        self._primed = True

    # -- processing ---------------------------------------------------------
    def process(self, chunk: np.ndarray, scale: bool = True) -> np.ndarray:
        """Filter and reference one chunk, advancing the state.

        Parameters
        ----------
        chunk : (n_samples, n_channels) array
            Raw int16 samples, or float if ``scale=False``.

        Returns
        -------
        (n_samples, n_channels) float32
        """
        x = np.asarray(chunk)
        if x.ndim != 2 or x.shape[1] != self.n_channels:
            raise ValueError(
                f"expected (n, {self.n_channels}) chunk, got {x.shape}"
            )
        x = x.astype(np.float64, copy=True)
        if scale:
            x *= self.cfg.bit_volts
        if not self._primed:
            self.prime(x[0])
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        y = self._reference(y)
        return y.astype(np.float32, copy=False)

    def _reference(self, y: np.ndarray) -> np.ndarray:
        op = self.cfg.common_reference
        if op == "none":
            return y
        if op == "median":
            return y - np.median(y, axis=1, keepdims=True)
        if op == "average":
            return y - y.mean(axis=1, keepdims=True)
        raise ValueError(f"unknown common_reference {op!r}")


def robust_sigma(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median-absolute-deviation estimate of noise sd, per channel.

    Uses the MAD rather than the standard deviation because spikes and
    stimulus artifacts are exactly the outliers an sd would absorb.
    """
    med = np.median(x, axis=axis, keepdims=True)
    return (np.median(np.abs(x - med), axis=axis) / 0.6744897501960817).astype(np.float32)
