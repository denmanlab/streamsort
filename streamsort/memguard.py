"""Memory safety for heavy offline jobs on small machines.

The development machine has 8 GB shared with the desktop, and a job that
pushes it into swap can take the whole machine down (this happened once).
Two guards, both cheap:

* :func:`preflight` -- before starting, read available memory (free +
  inactive + speculative pages, macOS ``vm_stat``) and refuse to run if it is
  below the job's budget plus a margin.
* :class:`MemGuard` -- inside long loops, call ``guard.check()``; it raises
  :class:`MemoryBudgetExceeded` as soon as the process's peak resident set
  exceeds the budget, so the job stops cleanly instead of thrashing.

Peak RSS (``ru_maxrss``) is monotonic, so the guard errs on the side of
stopping early.  Neither guard needs psutil.
"""
from __future__ import annotations

import re
import resource
import subprocess
import sys


class MemoryBudgetExceeded(MemoryError):
    pass


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)   # macOS: bytes, Linux: KiB


def available_bytes() -> int | None:
    """Reclaimable memory now (macOS vm_stat); None if it cannot be read."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    if not m:
        return None
    page = int(m.group(1))
    def pages(label):
        mm = re.search(rf"{label}:\s+(\d+)\.", out)
        return int(mm.group(1)) if mm else 0
    return page * (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative"))


def preflight(budget_gb: float, margin_gb: float = 0.5, strict: bool = True) -> float | None:
    """Refuse to start if available memory < budget + margin. Returns GB available."""
    av = available_bytes()
    if av is None:
        print("[memguard] could not read available memory; proceeding with in-process guard only",
              flush=True)
        return None
    av_gb = av / 1e9
    need = budget_gb + margin_gb
    print(f"[memguard] available {av_gb:.2f} GB, job budget {budget_gb:.2f} GB (+{margin_gb:.1f} margin)",
          flush=True)
    if strict and av_gb < need:
        raise MemoryBudgetExceeded(
            f"only {av_gb:.2f} GB available, need {need:.2f} GB; close other applications "
            "or lower the job size")
    return av_gb


class MemGuard:
    def __init__(self, budget_gb: float, label: str = "job"):
        self.budget = int(budget_gb * 1e9)
        self.label = label

    def check(self, where: str = "") -> None:
        p = peak_rss_bytes()
        if p > self.budget:
            raise MemoryBudgetExceeded(
                f"{self.label}: peak RSS {p/1e9:.2f} GB exceeded budget "
                f"{self.budget/1e9:.2f} GB at {where or 'checkpoint'}; stopping cleanly")

    def report(self) -> str:
        return f"peak RSS {peak_rss_bytes()/1e9:.2f} GB of {self.budget/1e9:.2f} GB budget"
