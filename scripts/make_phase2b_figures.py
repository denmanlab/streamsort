"""Phase 2b figures (figures/phase2b/): phase2b_core_vs_python.png (+ svg)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_figure_style

OUT = Path("results/phase2b"); FIG = Path("figures/phase2b"); FIG.mkdir(parents=True, exist_ok=True)
apply_figure_style()


def core_vs_python():
    R = json.loads((OUT / "phase2b_core_validation.json").read_text()); z = np.load(OUT / "phase2b_core_rt.npz")
    py = np.load("results/phase2/rt_factor.npz")["serial"]          # Phase 2 result, read-only
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1, 1.35]))
    cols = dict(A="#4c72b0", B="#55a868", C="#c44e52")
    hi = 1
    for p, (cpp, ref) in R["per_unit"].items():
        cpp, ref = np.asarray(cpp), np.asarray(ref); hi = max(hi, cpp.max(), ref.max())
        tp = R["three_probe"]["threads_per_stream_1"]["agreement"][p]
        a.scatter(np.maximum(ref, 0.8), np.maximum(cpp, 0.8), s=5, color=cols[p], alpha=0.7, lw=0,
                  label=f"{p}: {len(ref)} units, {tp['n_ref']} spikes")
    a.plot([0.8, hi * 1.3], [0.8, hi * 1.3], color="k", lw=0.6, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("spikes per unit, Python decoder"); a.set_ylabel("spikes per unit, C++ core")
    a.legend(fontsize=5.0, frameon=False, loc="upper left"); a.set_title("a  per-unit agreement, 120 s (spike sets identical)", loc="left", fontsize=7)
    bins = np.logspace(-2.3, 0.7, 70)
    a2 = dict(color="#888888", alpha=0.35)
    b.hist(py, bins=bins, histtype="stepfilled", **a2, label=f"Python, serial: {np.median(py):.2f} / {np.percentile(py, 99):.2f}")
    for k, c, lab in (("par1", "#2c4f8f", "C++, 1 thread/stream"), ("par2", "#b5651d", "C++, 2 threads/stream")):
        v = z[k]; b.hist(v, bins=bins, histtype="step", color=c, lw=1.0, label=f"{lab}: {np.median(v):.2f} / {np.percentile(v, 99):.2f}")
    b.axvline(1.0, color="k", lw=0.8, ls="--"); b.text(0.95, 300, "real time", fontsize=6, rotation=90, ha="right", va="bottom")
    b.set_xscale("log"); b.set_xlabel("processing time / block duration (1024 samples, 3 probes)"); b.set_ylabel("blocks")
    b.set_ylim(0, b.get_ylim()[1] * 1.6); b.legend(fontsize=5.0, frameon=False, loc="upper left", title="median / p99", title_fontsize=5.0)
    b.set_title("b  real-time factor per block (C++: streams in parallel)", loc="left", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "phase2b_core_vs_python.png", dpi=300); fig.savefig(FIG / "phase2b_core_vs_python.svg"); plt.close(fig)


if __name__ == "__main__":
    core_vs_python()
