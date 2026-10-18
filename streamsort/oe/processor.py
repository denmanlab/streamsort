"""GUI-independent core of the Open Ephys decoder processor (one instance per probe stream).

Input: the GUI buffer, float32 (n_channels, n_samples) in microvolts.  It is converted back to
raw int16 counts (the domain of the artifact suppressor and of the template bank), transposed and
cut to the first ``nch`` channels, then template-matched by the Phase 1 OnlineDecoder.
Spikes (frame, unit) and pulse onsets go to the worker; nothing is written back to the GUI buffer.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from ..config import StreamConfig
from ..online.decoder import OnlineDecoder, DecoderParams, SpikeBatch
from ..online.whitening import Whitening
from ..preprocess.suppressor import ArtifactSuppressor
from ..sorting.templates import TemplateBank

_CFG_FIELDS = {f.name for f in dataclasses.fields(StreamConfig)}


def _usable(report: Path | None, n_units: int) -> np.ndarray | None:
    """'usable' column of sorting_report.csv (stdlib csv: no pandas import in the GUI)."""
    if report is None or not Path(report).exists():
        return None
    import csv
    with open(report, newline="") as f:
        rows = list(csv.DictReader(f))
    m = np.array([r["usable"].strip().lower() == "true" for r in rows], bool)
    return m if len(m) == n_units else None


class DecoderProcessor:
    """cfg keys: probe, templates, whitening, sorting_report (optional), nch, bit_volts,
    input='uV'|'counts', detect_threshold (4.0), min_energy (50.0), ttl_line (stim TTL line on this
    stream, or None), worker_address ([host, port] or None), authkey, swap_check_s (0.5),
    swap_quiet_s (0.6: no swap within this long after a pulse), warm_ms (15)."""

    def __init__(self, cfg: dict, num_channels: int, sample_rate: float, conn=None):
        self.cfg = dict(cfg)
        self.probe = str(cfg.get("probe", "A"))
        self.nch = int(cfg["nch"])
        if num_channels < self.nch:
            raise ValueError(f"stream has {num_channels} channels, config needs {self.nch}")
        self.bit_volts = float(cfg.get("bit_volts", 0.195))
        self.in_uV = cfg.get("input", "uV") == "uV"
        self.gui_rate = float(sample_rate)
        self.paths = {k: Path(cfg[k]) for k in ("templates", "whitening") if cfg.get(k)}
        rep = cfg.get("sorting_report")
        self.report = Path(rep) if rep else Path(cfg["templates"]).with_name("sorting_report.csv")
        self.ttl_line = cfg.get("ttl_line")
        self.swap_check = float(cfg.get("swap_check_s", 0.5))
        self.swap_quiet = float(cfg.get("swap_quiet_s", 0.6))
        self.conn = conn
        self.n_swaps, self.timing = 0, []
        self._build()

    # -- model ----------------------------------------------------------------
    def _mtimes(self):
        return tuple(p.stat().st_mtime_ns for p in self.paths.values())

    def _build(self, sup: ArtifactSuppressor | None = None) -> None:
        bank = TemplateBank.load(self.paths["templates"])
        wh = Whitening.load(self.paths["whitening"]) if "whitening" in self.paths else None
        base = StreamConfig(**{k: v for k, v in (bank.config or {}).items() if k in _CFG_FIELDS})
        self.scfg = dataclasses.replace(base, detect_threshold=float(self.cfg.get("detect_threshold", 4.0)),
                                        n_channels_used=self.nch)
        self.fs = self.scfg.sample_rate
        self.sup = sup or ArtifactSuppressor(self.fs, self.nch)
        mask = _usable(self.report, bank.n_units)
        self.dec = OnlineDecoder(bank, self.scfg, wh, DecoderParams(min_energy=float(self.cfg.get("min_energy", 50.0))),
                                 units_mask=mask, check=False, suppressor=self.sup)
        self.unit_ids = bank.unit_ids[mask] if mask is not None else bank.unit_ids
        self.bank_mtime = self._mtimes()
        self.warm = self.scfg.samples(float(self.cfg.get("warm_ms", 15.0)))

    # -- acquisition ------------------------------------------------------------
    def start(self, sample0: int = 0) -> None:
        """JIT warm-up on synthetic noise (every kernel and the classify path), then reset."""
        rng = np.random.default_rng(0)
        self.dec.start_at(0)
        self.dec.push((rng.standard_normal((self.scfg.samples(50.0), self.nch)) * 20).astype(np.int16))
        self.dec.flush()
        self.sup.events.clear()
        self.dec.start_at(int(sample0))
        self.head = int(sample0)
        self._ring = np.zeros((0, self.nch), np.int16)
        self._n_ev, self._ttl_k, self._pending_ttl, self._cand = 0, -1, [], []
        self._last_pulse = -10**12
        self._arrivals = []           # (block end sample, perf_counter at block arrival)
        self._next_swap_check = self.head + int(self.swap_check * self.fs)
        self.timing = []
        if self.conn is not None:
            self.conn.send(("hello", self.probe, self.fs, self.unit_ids, self.bank_mtime))

    def ttl(self, line: int, state: bool) -> None:
        """Stim TTL rising edge on this stream: gives the next detected pulse a trial index."""
        if self.ttl_line is not None and int(line) == int(self.ttl_line) and state:
            self._ttl_k += 1
            self._pending_ttl.append((self._ttl_k, self.head))

    def process(self, data: np.ndarray) -> SpikeBatch:
        t0 = time.perf_counter()
        self._arrivals.append((self.head + np.shape(data)[1], t0)); self._arrivals = self._arrivals[-64:]
        x = np.asarray(data)[: self.nch]
        raw = np.rint(x / self.bit_volts) if self.in_uV else x
        raw = np.ascontiguousarray(raw.T).astype(np.int16)
        b = self.dec.push(raw)
        self.head += len(raw)
        self._ring = np.concatenate([self._ring, raw])[-self.warm:]
        self._poll_and_send(b)
        if self.head >= self._next_swap_check:
            self._next_swap_check = self.head + int(self.swap_check * self.fs)
            self._maybe_swap()
        self.timing.append(time.perf_counter() - t0)
        return b

    def stop(self) -> SpikeBatch:
        b = self.dec.flush()
        self._poll_and_send(b)
        if self.conn is not None:
            self.conn.send(("bye", self.probe))
        return b

    # -- worker link --------------------------------------------------------------
    def _pair(self) -> list:
        """Artifact onsets (exact) paired with stim TTLs (trial index) when a TTL line is routed.
        Either may arrive one buffer before the other; a TTL with no artifact within ``tol``
        is sent at its block start (approximate), a clip with no TTL within ``tol`` is dropped."""
        ev = self.sup.events
        self._cand += [int(e["onset"]) for e in ev[self._n_ev:]]
        self._n_ev = len(ev)
        if self.ttl_line is None:
            out, self._cand = [(on, None) for on in self._cand], []
            return out
        tol, out = int(float(self.cfg.get("ttl_tol_ms", 50.0)) * 1e-3 * self.fs), []
        while self._cand and self._pending_ttl:
            on = self._cand[0]; k, th = self._pending_ttl[0]
            if on < th - tol:                      # clip with no TTL: not a stimulus
                self._cand.pop(0)
            elif on > th + tol:                    # TTL with no detected artifact
                self._pending_ttl.pop(0); out.append((th, k))
            else:
                self._cand.pop(0); self._pending_ttl.pop(0); out.append((on, k))
        self._cand = [on for on in self._cand if self.head - on <= tol]
        while self._pending_ttl and self.head - self._pending_ttl[0][1] > tol:
            k, th = self._pending_ttl.pop(0); out.append((th, k))
        return out

    def _poll_and_send(self, b: SpikeBatch) -> None:
        pulses = self._pair()
        for on, _ in pulses:
            self._last_pulse = max(self._last_pulse, on)
        if self.conn is None:
            self.pulses_out = getattr(self, "pulses_out", []) + pulses
            return
        m = b.unit >= 0
        self.conn.send(("buf", self.probe, self.head, b.frame[m], b.unit[m]))
        for on, k in pulses:
            self.conn.send(("pulse", self.probe, on, k, self._arrival_of(on)))
        while self.conn.poll():
            msg = self.conn.recv()
            if msg[0] == "condition":
                self.sup.set_condition(msg[1])

    def _arrival_of(self, sample: int) -> float:
        """Wall time (perf_counter, system-wide monotonic) at which the block holding ``sample`` arrived."""
        for end, t in self._arrivals:
            if sample < end:
                return t
        return time.perf_counter()

    def set_condition(self, key) -> None:
        self.sup.set_condition(key)

    # -- template hot-swap ------------------------------------------------------------
    def _maybe_swap(self) -> bool:
        """Rebuild the decoder when templates.npz / whitening.npz change (watcher writes are atomic).
        Only between pulses; the new decoder is warmed on the last ``warm_ms`` of raw data."""
        try:
            mt = self._mtimes()
        except FileNotFoundError:
            return False
        if mt == self.bank_mtime or self.head - self._last_pulse < self.swap_quiet * self.fs:
            return False
        tail = self.dec.flush()                    # finalise everything up to head
        self._poll_and_send(tail)
        n_ev, evs = self._n_ev, self.sup.events
        self._build(sup=self.sup)                  # keeps per-condition artifact history
        self.sup.events = evs
        self.dec.start_at(self.head - len(self._ring))
        self._n_ev = n_ev
        self.dec.push(self._ring)                  # warm the causal filter; these spikes were already emitted
        self.n_swaps += 1
        if self.conn is not None:
            self.conn.send(("hello", self.probe, self.fs, self.unit_ids, self.bank_mtime))
        return True
