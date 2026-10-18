"""Python side of the C++ decoder core (module ``streamsort_core``, built from cpp/).

* :func:`tables_from_decoder` derives every table the C++ decoder needs from a Python
  :class:`~streamsort.online.decoder.OnlineDecoder`, so both paths share one set of constants.
* :func:`export_tables` / :func:`load_exported` write/read the table folder the Open Ephys plugin
  loads (``.npy`` arrays + ``decoder_tables.txt`` scalars).
* :class:`CoreDecoder` wraps the C++ decoder with the OnlineDecoder push/flush API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .online.decoder import SpikeBatch

_BUILD = Path(__file__).resolve().parents[1] / "cpp" / "build"


def _module():
    try:
        import streamsort_core  # installed with `pip install ./cpp`
    except ImportError:
        p = os.environ.get("STREAMSORT_CORE_PATH", str(_BUILD))
        if p not in sys.path:
            sys.path.insert(0, p)
        import streamsort_core
    return streamsort_core


ARRAYS = ("sos", "zi_unit", "w_ptr", "w_row", "w_val", "thr", "nbr", "cand_ptr", "cand_units", "tro_cand",
          "Tcat", "ttcat", "sset", "shifts", "unit_ids", "sup_taper")


def tables_from_decoder(dec) -> dict:
    """All C++ tables from a constructed (numba-path) OnlineDecoder."""
    if not getattr(dec, "use_numba", False):
        raise ValueError("build the reference OnlineDecoder with use_numba=True (median referencing)")
    WT = np.asarray(dec.WT, np.float32); C = WT.shape[0]
    ptr, rows, vals = [0], [], []
    for j in range(C):                                  # column j of W^T, ascending input channel
        nz = np.flatnonzero(WT[:, j] != 0)
        rows.append(nz); vals.append(WT[nz, j]); ptr.append(ptr[-1] + len(nz))
    t = dict(C=int(C), bit_volts=float(dec.cfg.bit_volts),
             sos=np.ascontiguousarray(dec.sos, np.float64), zi_unit=np.ascontiguousarray(dec.pre._zi_unit, np.float64),
             w_ptr=np.asarray(ptr, np.int64), w_row=np.concatenate(rows).astype(np.int64), w_val=np.concatenate(vals).astype(np.float32),
             thr=np.asarray(dec.thr, np.float32), nbr=np.ascontiguousarray(dec.nbr, np.int64), K=int(dec.nbr.shape[1]),
             dedup=int(dec.dedup), before_need=int(dec.before_need), after_need=int(dec.after_need), lag=int(dec.lag),
             bpre=int(dec.bpre), bpost=int(dec.bpost), clip_counts=int(dec.p.clip_counts), clip_min_channels=int(dec.p.clip_min_channels),
             T=int(dec.T), Ks=int(dec.p.score_channels), n_before=int(dec.cfg.n_before),
             cand_ptr=np.asarray(dec.cand_ptr, np.int64), cand_units=np.asarray(dec.cand_units, np.int64),
             tro_cand=np.asarray(dec.tro_cand, np.int64), Tcat=np.ascontiguousarray(dec.Tcat, np.float32),
             ttcat=np.asarray(dec.ttcat, np.float32), sset=np.ascontiguousarray(dec.sset64, np.int64),
             shifts=np.asarray(dec.shifts, np.int64), amp_lo=float(dec.p.amp_range[0]), amp_hi=float(dec.p.amp_range[1]),
             min_energy=float(dec.p.min_energy), unit_ids=np.asarray(dec.bank.unit_ids, np.int64),
             has_suppressor=dec.suppressor is not None)
    s = dec.suppressor
    if s is not None:
        t.update(sup_clip=int(s.clip), sup_kmin=int(s.kmin), sup_L=int(s.L), sup_pre=int(s.pre), sup_post=int(s.post),
                 sup_maxspan=int(s.maxspan), sup_nb=int(s.nb), sup_n_history=int(s.n_history),
                 sup_min_history=int(s.min_history), sup_taper=np.asarray(s.taper, np.float32),
                 sup_subtract=bool(s.subtract), sup_bridge=bool(s.bridge))
    else:
        t["sup_taper"] = np.zeros(0, np.float32)
    return t


def export_tables(tables: dict, folder, extra: dict | None = None) -> Path:
    """Write the plugin table folder (atomic per file; decoder_tables.txt written last)."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for k in ARRAYS:
        a = np.ascontiguousarray(tables[k]); tmp = folder / f"{k}.tmp.npy"
        np.save(tmp, a.astype(a.dtype.newbyteorder("<"))); os.replace(tmp, folder / f"{k}.npy")
    lines = [f"{k}={int(v) if isinstance(v, (bool, np.bool_)) else v}" for k, v in tables.items() if k not in ARRAYS]
    lines += [f"{k}={v}" for k, v in (extra or {}).items()]
    tmp = folder / "decoder_tables.tmp"; tmp.write_text("# streamsort decoder tables\n" + "\n".join(lines) + "\n")
    os.replace(tmp, folder / "decoder_tables.txt")
    return folder


def load_exported(folder) -> dict:
    folder = Path(folder); t = {}
    for line in (folder / "decoder_tables.txt").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            try:
                t[k] = int(v)
            except ValueError:
                try:
                    t[k] = float(v)
                except ValueError:
                    t[k] = v
    for k in ARRAYS:
        t[k] = np.load(folder / f"{k}.npy")
    t["has_suppressor"] = bool(t["has_suppressor"])
    for k in ("sup_subtract", "sup_bridge"):
        if k in t:
            t[k] = bool(t[k])
    return t


class CoreDecoder:
    """C++ decoder with the OnlineDecoder streaming API (push/flush/start_at/set_condition)."""

    def __init__(self, tables: dict, n_threads: int = 1):
        self._m = _module()
        self.t = tables
        self.C = int(tables["C"])
        self._dec = self._m.Decoder(tables, int(n_threads))
        self._keys: dict = {}

    def cond_id(self, key) -> int:
        if key is None:
            return -1
        return self._keys.setdefault(key, len(self._keys))

    def set_condition(self, key) -> None:
        self._dec.set_condition(self.cond_id(key))

    def start_at(self, sample: int) -> None:
        self._dec.start_at(int(sample))

    @staticmethod
    def _batch(d) -> SpikeBatch:
        return SpikeBatch(d["frame"], d["unit"], d["score"], d["amp"], d["peak_channel"])

    def push(self, raw) -> SpikeBatch:
        return self._batch(self._dec.push(np.asarray(raw)))

    def flush(self) -> SpikeBatch:
        return self._batch(self._dec.flush())

    @property
    def events(self) -> list:
        inv = {v: k for k, v in self._keys.items()}
        return [dict(onset=o, span=(s if s >= 0 else None), condition=inv.get(c), n_history=h, subtracted=b)
                for o, s, c, h, b in self._dec.events()]

    def timing(self) -> dict:
        return dict(zip(("suppress", "blank", "filter", "median_whiten", "detect", "classify"), self._dec.timing()))

    @property
    def n_blanked(self) -> int:
        return self._dec.n_blanked
