"""Stimulus-artifact suppression.

An electrical stimulus drives the amplifier far outside its linear range.  If
the raw transient is allowed to reach the highpass filter it rings for many
milliseconds, and that ringing is indistinguishable from spiking to any
threshold detector.  The fix is to remove the transient *before* filtering.

Blanking is applied on the raw trace, so the only thing the filter ever sees is
a smooth substitution.  Three substitution modes are provided; ``hold`` is the
default because it is strictly causal -- it needs no sample from after the
blank window -- and so behaves identically offline and in a live stream.
"""
from __future__ import annotations

import numpy as np

from ..config import StreamConfig


def blank_windows(chunk: np.ndarray, starts: np.ndarray, stops: np.ndarray,
                  mode: str = "hold") -> np.ndarray:
    """Replace sample ranges ``[start, stop)`` in a raw chunk.

    Parameters
    ----------
    chunk : (n_samples, n_channels)
        Raw data. Modified on a copy; the input is untouched.
    starts, stops : int arrays
        Sample indices *relative to this chunk*. Ranges are clipped to the
        chunk and may be partially outside it.
    mode : {"hold", "zeros", "linear"}
        ``hold`` repeats the last sample before the window (causal).
        ``zeros`` substitutes zero. ``linear`` interpolates between the
        samples bracketing the window and is **not** causal.
    """
    out = np.array(chunk, copy=True)
    n = out.shape[0]
    for a, b in zip(np.atleast_1d(starts), np.atleast_1d(stops)):
        a = int(a); b = int(b)
        if b <= 0 or a >= n:
            continue
        lo, hi = max(a, 0), min(b, n)
        if hi <= lo:
            continue
        if mode == "zeros":
            out[lo:hi] = 0
        elif mode == "hold":
            fill = out[lo - 1] if lo > 0 else (out[hi] if hi < n else 0)
            out[lo:hi] = fill
        elif mode == "linear":
            left = out[lo - 1] if lo > 0 else out[hi % n]
            right = out[hi] if hi < n else left
            w = np.linspace(0.0, 1.0, hi - lo + 2)[1:-1][:, None]
            out[lo:hi] = (1 - w) * left[None, :] + w * right[None, :]
        else:
            raise ValueError(f"unknown blank mode {mode!r}")
    return out


class StimBlanker:
    """Tracks stimulus windows and blanks them inside streamed chunks.

    Holds the stimulus onset/offset times once, then answers "which samples of
    this chunk are contaminated?" for any chunk.  Because the windows are known
    in absolute sample indices, a window straddling a buffer boundary is
    handled correctly without special-casing.
    """

    def __init__(self, cfg: StreamConfig, onset_samples: np.ndarray,
                 offset_samples: np.ndarray, mode: str = "hold"):
        self.cfg = cfg
        self.mode = mode
        self.pre = cfg.samples(cfg.blank_pre_ms)
        self.post = cfg.samples(cfg.blank_post_ms)
        self.starts = np.asarray(onset_samples, dtype=np.int64) - self.pre
        self.stops = np.asarray(offset_samples, dtype=np.int64) + self.post
        order = np.argsort(self.starts)
        self.starts, self.stops = self.starts[order], self.stops[order]

    def windows_in(self, start: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Blank windows overlapping ``[start, start+n)``, chunk-relative."""
        lo, hi = int(start), int(start) + int(n)
        sel = (self.stops > lo) & (self.starts < hi)
        return self.starts[sel] - lo, self.stops[sel] - lo

    def apply(self, chunk: np.ndarray, start: int) -> np.ndarray:
        a, b = self.windows_in(start, chunk.shape[0])
        if len(a) == 0:
            return chunk
        return blank_windows(chunk, a, b, mode=self.mode)

    def clean_mask(self, start: int, n: int) -> np.ndarray:
        """Boolean mask of samples *not* inside a blank window."""
        m = np.ones(int(n), dtype=bool)
        a, b = self.windows_in(start, n)
        for x, y in zip(a, b):
            m[max(int(x), 0):max(int(y), 0)] = False
        return m


#: Raw-count magnitude at or beyond which an AP-band sample is treated as
#: clipped.  Measured on jlh32: the stim-free maximum over 120 M samples is
#: 4092 counts, while every stimulus trial pegs at +6369 / -6206 with a hard
#: cliff (99 samples at 6369, none at 6368).  6190 sits just inside both rails.
CLIP_COUNTS = 6190


def detect_artifact_onset(raw: np.ndarray, nominal: int, search_samples: int,
                          clip_counts: int = CLIP_COUNTS,
                          min_channels: int = 3) -> int | None:
    """Find the true artifact onset near a nominal (TTL-derived) sample.

    The TTL index is not reliable to better than about a millisecond once the
    NI-DAQ-to-probe clock residual is folded in -- on jlh32 the artifact is
    sometimes seen up to 0.3 ms *before* the TTL edge.  The artifact itself is
    unmissable (it clips tens of channels), so locating it in the data is both
    more accurate and available live.

    Returns the first sample at which at least ``min_channels`` are clipped,
    or ``None`` if no clipping is found in the search window.
    """
    lo = max(int(nominal) - int(search_samples), 0)
    hi = min(int(nominal) + int(search_samples), raw.shape[0])
    if hi <= lo:
        return None
    n_clipped = (np.abs(raw[lo:hi]) >= clip_counts).sum(axis=1)
    hit = np.flatnonzero(n_clipped >= min_channels)
    return int(lo + hit[0]) if hit.size else None


def clipped_span(raw: np.ndarray, onset: int, max_samples: int,
                 clip_counts: int = CLIP_COUNTS) -> int:
    """Samples from ``onset`` until no channel is clipped any longer.

    This is the only part of the record that is genuinely unrecoverable.
    Everything after it is a large but linear settling tail, which template
    subtraction can remove -- so the blank window should cover this span and
    stop.  Blanking well past it is actively harmful: a sample-and-hold that
    releases into a still-large signal produces a step the highpass rings on
    (measured at 42 uV for an 8 ms blank, versus 9 uV for a 2.5 ms blank).
    """
    hi = min(int(onset) + int(max_samples), raw.shape[0])
    seg = np.abs(raw[int(onset):hi]) >= clip_counts
    if not seg.any():
        return 0
    any_clipped = seg.any(axis=1)
    return int(np.flatnonzero(any_clipped)[-1] + 1)


def artifact_template(snippets: np.ndarray, robust: bool = True) -> np.ndarray:
    """Average artifact waveform across repeats of one stimulus condition.

    Parameters
    ----------
    snippets : (n_trials, n_samples, n_channels)
    robust : bool
        Median across trials (default) rather than mean, so that a trial
        carrying a large spike does not leak into the artifact estimate.
    """
    return (np.median(snippets, axis=0) if robust
            else np.mean(snippets, axis=0)).astype(np.float32)
