"""Hide macOS AppleDouble sidecar files from directory iteration.

Network and FUSE mounts on macOS (here a fuse-t NFS scratch volume) synthesise
``._<name>`` companion entries for files that carry extended attributes.  They
appear in directory listings but are not real data: unlinking one fails with
FileNotFoundError, and loading one as ``.npy`` fails as an invalid pickle.
SpikeInterface discovers properties, sorter outputs and analyzer extensions by
iterating directories, so a single sidecar aborts a load.

:func:`install` patches the directory-iteration primitives for the current
process so that ``._*`` entries are skipped.  It is idempotent and only ever
*removes* AppleDouble names from listings -- no real file is hidden, since a
legitimate data file never begins with ``._``.  Call it once at process start,
in offline tooling only; the online decoder never iterates directories.
"""
from __future__ import annotations

import os
import pathlib

_PREFIX = "._"
_installed = False


def _is_sidecar(name: str) -> bool:
    return os.path.basename(name).startswith(_PREFIX)


def install() -> None:
    """Patch os.listdir, os.scandir and pathlib iteration to skip ``._*``."""
    global _installed
    if _installed:
        return

    _listdir = os.listdir
    _scandir = os.scandir
    _iterdir = pathlib.Path.iterdir
    _glob = pathlib.Path.glob
    _rglob = pathlib.Path.rglob

    def listdir(path="."):
        return [n for n in _listdir(path) if not _is_sidecar(n)]

    class _FilteredScandir:
        """Wraps an os.scandir iterator. Must be a real iterator, not just an
        iterable: os.walk calls next() on the scandir result directly."""

        def __init__(self, it):
            self._it = it

        def __iter__(self):
            return self

        def __next__(self):
            while True:
                e = next(self._it)
                if not _is_sidecar(e.name):
                    return e

        def __enter__(self):
            self._it.__enter__()
            return self

        def __exit__(self, *exc):
            return self._it.__exit__(*exc)

        def close(self):
            self._it.close()

    def scandir(path="."):
        return _FilteredScandir(_scandir(path))

    def iterdir(self):
        return (p for p in _iterdir(self) if not _is_sidecar(p.name))

    def glob(self, pattern, *a, **k):
        return (p for p in _glob(self, pattern, *a, **k) if not _is_sidecar(p.name))

    def rglob(self, pattern, *a, **k):
        return (p for p in _rglob(self, pattern, *a, **k) if not _is_sidecar(p.name))

    os.listdir = listdir
    os.scandir = scandir
    pathlib.Path.iterdir = iterdir
    pathlib.Path.glob = glob
    pathlib.Path.rglob = rglob
    _installed = True
