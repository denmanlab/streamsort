"""Fit watcher: wait for the leading segment of a recording that is still being written, freeze it
into a self-consistent snapshot, and fit on the snapshot (``streamsort watch``).

Live-file rules (Open Ephys binary format, GUI still writing):
  * continuous.dat grows; the segment is ready when its size covers ``seconds`` on two
    consecutive polls (size-stable polling: the bytes we need are flushed, not mid-write).
  * .npy files (timestamps, sample_numbers, TTL events) are appended to and their header shape is
    only patched when recording stops, so the entry count is taken from the file size.
  * on Windows the GUI holds the files open: every open/read retries on PermissionError.
  * the fitted templates.npz / whitening.npz are written with os.replace (atomic), so running
    decoder processors hot-swap to them without ever reading a partial file.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

CHUNK = 64 << 20


def _retry(fn, tries: int = 50, wait_s: float = 0.2):
    for i in range(tries):
        try:
            return fn()
        except PermissionError:                 # Windows sharing violation while the GUI writes
            if i == tries - 1:
                raise
            time.sleep(wait_s)


def read_growing_npy(path) -> np.ndarray:
    """Complete entries of an .npy file that may still be appended to (header shape ignored)."""
    path = Path(path)

    def _read():
        with open(path, "rb") as f:
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                     else np.lib.format.read_array_header_2_0(f))
            hdr = f.tell()
            row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            size = path.stat().st_size
            n = (size - hdr) // (dtype.itemsize * row)
            a = np.frombuffer(f.read(n * dtype.itemsize * row), dtype=dtype)
        return a.reshape((n,) + tuple(shape[1:]))
    return _retry(_read)


def _oebin(rec_dir) -> dict:
    return json.loads(_retry(lambda: (Path(rec_dir) / "structure.oebin").read_text()))


def segment_bytes(rec_dir, stream: str, seconds: float) -> tuple[int, int, float]:
    e = next(c for c in _oebin(rec_dir)["continuous"] if c["folder_name"].rstrip("/") == stream)
    n_ch, fs = int(e["num_channels"]), float(e["sample_rate"])
    return int(seconds * fs) * 2 * n_ch, n_ch, fs


def wait_for_segment(rec_dir, stream: str, seconds: float, poll_s: float = 5.0,
                     stable_polls: int = 2, timeout_s: float | None = None, log=print) -> int:
    """Block until continuous.dat holds ``seconds`` of data on ``stable_polls`` consecutive polls."""
    need, _, _ = segment_bytes(rec_dir, stream, seconds)
    dat = Path(rec_dir) / "continuous" / stream / "continuous.dat"
    ok, t0 = 0, time.time()
    while True:
        size = _retry(lambda: dat.stat().st_size) if dat.exists() else 0
        ok = ok + 1 if size >= need else 0
        if ok >= stable_polls:
            return size
        if timeout_s is not None and time.time() - t0 > timeout_s:
            raise TimeoutError(f"{dat}: {size} of {need} bytes after {timeout_s} s")
        log(f"[watch] {stream}: {size / need:.0%} of the {seconds:.0f} s segment")
        time.sleep(poll_s)


def snapshot(rec_dir, stream: str, seconds: float, out_root, event_streams=()) -> Path:
    """Freeze the first ``seconds`` of ``stream`` (plus TTL events up to then) into
    ``out_root/<Record Node>/experiment1/recording1`` with a matching structure.oebin.
    Returns the snapshot recording directory (use as REC_DIR; out_root as REC_ROOT)."""
    rec_dir = Path(rec_dir)
    need, n_ch, fs = segment_bytes(rec_dir, stream, seconds)
    n = need // (2 * n_ch)
    node = rec_dir.parents[1].name
    dst = Path(out_root) / node / rec_dir.parent.name / rec_dir.name
    sd = dst / "continuous" / stream; sd.mkdir(parents=True, exist_ok=True)
    src = rec_dir / "continuous" / stream

    def _copy():
        with open(src / "continuous.dat", "rb") as f, open(sd / "continuous.dat.part", "wb") as g:
            left = need
            while left:
                b = f.read(min(CHUNK, left))
                if not b:
                    raise IOError("continuous.dat shorter than the segment")
                g.write(b); left -= len(b)
    _retry(_copy); (sd / "continuous.dat.part").replace(sd / "continuous.dat")
    ts = read_growing_npy(src / "timestamps.npy")
    if len(ts) < n:          # timestamps lag the data on disk: extend at the nominal rate
        t0 = float(ts[0]) if len(ts) else 0.0
        ts = np.concatenate([ts, t0 + np.arange(len(ts), n) / fs]).astype(ts.dtype if len(ts) else np.float64)
    np.save(sd / "timestamps.npy", ts[:n])
    if (src / "sample_numbers.npy").exists():
        sn = read_growing_npy(src / "sample_numbers.npy")
        if len(sn) < n:
            s0 = int(sn[0]) if len(sn) else 0
            sn = np.concatenate([sn, s0 + np.arange(len(sn), n)]).astype(np.int64)
        np.save(sd / "sample_numbers.npy", sn[:n])
    for extra in ("channel_positions.npy", "channel_map.npy"):
        if (src / extra).exists():
            shutil.copy(src / extra, sd / extra)
    t_end = float(ts[n - 1])
    ob = _oebin(rec_dir)
    keep_ev = {stream} | set(event_streams)
    ob["continuous"] = [c for c in ob["continuous"] if c["folder_name"].rstrip("/") == stream]
    evs = []
    for e in ob.get("events", []):
        folder = e["folder_name"].rstrip("/")
        if not any(folder.startswith(k) for k in keep_ev):
            continue
        es, ed = rec_dir / "events" / folder, dst / "events" / folder
        if not (es / "timestamps.npy").exists():
            continue
        ed.mkdir(parents=True, exist_ok=True)
        et = read_growing_npy(es / "timestamps.npy"); arrs = {"timestamps": et}
        for k in ("sample_numbers", "states", "full_words"):
            if (es / f"{k}.npy").exists():
                arrs[k] = read_growing_npy(es / f"{k}.npy")
        m = min(len(a) for a in arrs.values())
        sel = np.flatnonzero(arrs["timestamps"][:m] <= t_end)
        for k, a in arrs.items():
            np.save(ed / f"{k}.npy", a[:m][sel])
        evs.append(e)
    ob["events"], ob["spikes"] = evs, []
    (dst / "structure.oebin").write_text(json.dumps(ob, indent=2))
    for f in ("settings.xml",):
        s = rec_dir.parents[1] / f
        if s.exists():
            _retry(lambda: shutil.copy(s, dst.parents[1] / f))
    return dst
