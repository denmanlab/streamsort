"""Stream sources: a memory-mapped file reader and a simulated live stream.

:class:`FileStreamSource` is a random-access reader over an Open Ephys
``continuous.dat``.  :class:`SimulatedLiveStream` wraps it and exposes only the
samples "already written", so downstream code cannot accidentally read into the
future.  Any component that works against the simulated stream is causal by
construction, which is the property we need for the live use case.
"""
from __future__ import annotations

import time

import numpy as np

from .openephys import ContinuousStream


class NotYetWritten(Exception):
    """Raised when a read extends past the data currently on disk."""

    def __init__(self, requested_end: int, head: int):
        self.requested_end = requested_end
        self.head = head
        super().__init__(
            f"requested samples up to {requested_end} but only {head} written "
            f"({requested_end - head} short)"
        )


class FileStreamSource:
    """Memory-mapped random access to one continuous.dat.

    Samples are interleaved by channel on disk: shape ``(n_samples, n_channels)``
    int16.  Reads return a view where possible and never copy the full file.
    """

    def __init__(self, stream: ContinuousStream, n_channels_used: int | None = None):
        self.stream = stream
        self.n_channels = stream.num_channels
        self.n_channels_used = n_channels_used or stream.num_channels
        if self.n_channels_used > self.n_channels:
            raise ValueError(
                f"n_channels_used={self.n_channels_used} exceeds the "
                f"{self.n_channels} channels on disk"
            )
        self._mm = np.memmap(
            stream.dat_path, dtype=np.int16, mode="r",
            shape=(stream.n_samples, self.n_channels),
        )

    # -- geometry of the recording ----------------------------------------
    @property
    def n_samples(self) -> int:
        return self.stream.n_samples

    @property
    def t_start(self) -> float:
        return self.stream.t_start

    @property
    def t_end(self) -> float:
        return self.stream.t_end

    def time_to_sample(self, t):
        return self.stream.time_to_sample(t)

    def sample_to_time(self, i):
        return self.stream.sample_to_time(i)

    # -- reading ------------------------------------------------------------
    def read(self, start: int, n: int) -> np.ndarray:
        """Read ``n`` samples starting at sample index ``start``.

        Returns an ``(n, n_channels_used)`` int16 view.  Raises on an
        out-of-range request rather than silently truncating, because a short
        read in a streaming context is a bug, not a condition to paper over.
        """
        start = int(start)
        n = int(n)
        if start < 0:
            raise ValueError(f"negative start sample {start}")
        if start + n > self.n_samples:
            raise ValueError(
                f"read of {n} samples at {start} runs past the file "
                f"({self.n_samples} samples)"
            )
        return self._mm[start:start + n, :self.n_channels_used]

    def read_time(self, t0: float, duration_s: float) -> tuple[np.ndarray, int]:
        """Read ``duration_s`` of data starting at global time ``t0``.

        Returns ``(data, start_sample)``.  The start sample is returned so
        callers can report spike times as exact sample indices rather than
        round-tripping through floating-point seconds.
        """
        start = int(self.time_to_sample(t0))
        n = int(round(duration_s / self.stream.slope))
        return self.read(start, n), start


class SimulatedLiveStream:
    """A :class:`FileStreamSource` that pretends data is still being written.

    The stream exposes a write head.  Reads beyond it raise
    :class:`NotYetWritten`.  The head advances either manually
    (:meth:`advance_to`, :meth:`advance_to_time`) or in wall-clock real time
    (``realtime=True``), which is what the end-to-end demo uses.
    """

    def __init__(self, source: FileStreamSource, head: int = 0,
                 realtime: bool = False, speed: float = 1.0):
        self.source = source
        self._head = int(head)
        self.realtime = realtime
        self.speed = float(speed)
        self._t0_wall: float | None = None
        self._head_at_start = int(head)

    # -- write head ---------------------------------------------------------
    @property
    def head(self) -> int:
        """Samples currently 'written to disk'."""
        if self.realtime and self._t0_wall is not None:
            elapsed = (time.perf_counter() - self._t0_wall) * self.speed
            grown = int(elapsed / self.source.stream.slope)
            return min(self._head_at_start + grown, self.source.n_samples)
        return self._head

    @property
    def head_time(self) -> float:
        return float(self.source.sample_to_time(self.head))

    def start_realtime(self) -> None:
        """Begin advancing the head in wall-clock time from its current value."""
        self.realtime = True
        self._head_at_start = self._head
        self._t0_wall = time.perf_counter()

    def advance_to(self, sample: int) -> None:
        if self.realtime:
            raise RuntimeError("cannot advance manually while in realtime mode")
        self._head = min(int(sample), self.source.n_samples)

    def advance_to_time(self, t: float) -> None:
        self.advance_to(int(self.source.time_to_sample(t)))

    def advance_by(self, n: int) -> None:
        self.advance_to(self._head + int(n))

    @property
    def exhausted(self) -> bool:
        return self.head >= self.source.n_samples

    # -- reading ------------------------------------------------------------
    def read(self, start: int, n: int) -> np.ndarray:
        end = int(start) + int(n)
        head = self.head
        if end > head:
            raise NotYetWritten(end, head)
        return self.source.read(start, n)

    def read_time(self, t0: float, duration_s: float) -> tuple[np.ndarray, int]:
        start = int(self.source.time_to_sample(t0))
        n = int(round(duration_s / self.source.stream.slope))
        return self.read(start, n), start

    def wait_for(self, end_sample: int, timeout_s: float = 30.0,
                 poll_s: float = 0.002) -> float:
        """Block until ``end_sample`` has been written. Returns seconds waited.

        Uses a short polling loop rather than ``sleep`` of a computed duration:
        on Windows the default timer granularity is ~15.6 ms, which would
        dominate any sub-millisecond timing we are trying to measure.
        """
        t_begin = time.perf_counter()
        while self.head < end_sample:
            if time.perf_counter() - t_begin > timeout_s:
                raise TimeoutError(
                    f"waited {timeout_s}s for sample {end_sample}; head={self.head}"
                )
            time.sleep(poll_s)
        return time.perf_counter() - t_begin

    def iter_buffers(self, buffer_size: int, start: int | None = None,
                     stop: int | None = None):
        """Yield ``(start_sample, buffer)`` in fixed-size blocks.

        This is the push-mode driver: it reproduces how the Open Ephys GUI
        hands small buffers to a Filter plugin's ``process()`` callback, so the
        same decoder code path is exercised here and on the rig.
        """
        i = self._head if start is None else int(start)
        stop = self.source.n_samples if stop is None else int(stop)
        while i + buffer_size <= stop:
            if self.realtime:
                self.wait_for(i + buffer_size)
            elif self.head < i + buffer_size:
                self.advance_to(i + buffer_size)
            yield i, self.source.read(i, buffer_size)
            i += buffer_size
