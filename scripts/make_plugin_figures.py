"""plugin_architecture.png and plugin_realtime_factor.png from results/phase2 (replay harness output)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_figure_style

OUT = Path("results/phase2"); FIG = Path("figures/phase2"); FIG.mkdir(parents=True, exist_ok=True)
apply_figure_style()


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#333333", fs=6.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012", fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, linespacing=1.15)


def arrow(ax, a, b, text=None, off=(0, 0.018), color="#333333", ls="-", rad=0.0, fs=5.8):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, ls=ls,
                                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, mutation_scale=7))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs, color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.4)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.53, 0.80, boxstyle="round,pad=0.005,rounding_size=0.015", fc="#f2f5fa", ec="#5a7bb5", lw=0.9))
    ax.text(0.02, 0.81, "Open Ephys GUI (shared Python interpreter)", fontsize=6.2, color="#2c4f8f", fontweight="bold")
    box(ax, 0.025, 0.30, 0.135, 0.42, "Neuropix-PXI\nsource\n\nprobe A/B/C\nAP streams\n\nstim TTL on\neach stream\n(basestation\nSMA input)", fs=5.4)
    ys = [0.58, 0.37, 0.16]
    for p, y in zip("ABC", ys):
        box(ax, 0.205, y, 0.32, 0.16, f"Python Processor {p}: streamsort_oe_{p}.py\n\u00b5V\u2192counts \u00b7 artifact suppressor\ncausal filter \u00b7 whitening \u00b7 template match", fs=5.2)
        arrow(ax, (0.16, 0.51), (0.205, y + 0.08))
    ax.text(0.275, 0.075, "processors run serially per block: their sum must stay below real time", fontsize=5.2, ha="center", color="#555555", style="italic")
    ax.add_patch(FancyBboxPatch((0.64, 0.30), 0.35, 0.54, boxstyle="round,pad=0.005,rounding_size=0.015", fc="#f6f6f2", ec="#8a7a3a", lw=0.9))
    ax.text(0.65, 0.81, "streamsort worker (own process)", fontsize=6.2, color="#6b5a1f", fontweight="bold")
    box(ax, 0.655, 0.64, 0.32, 0.14, "pool probes \u00b7 pair pulses (TTL index)\nbin 10 ms, 2\u2013502 ms after pulse", fs=5.2)
    box(ax, 0.655, 0.48, 0.32, 0.12, "reducer: PCA | FA | Isomap\nloss: Procrustes | linear readout", fs=5.2)
    box(ax, 0.655, 0.32, 0.32, 0.12, "stimulator thread: policy \u2192 reprogram\n(hold if next-pulse deadline missed)", fs=5.2)
    arrow(ax, (0.815, 0.64), (0.815, 0.60)); arrow(ax, (0.815, 0.48), (0.815, 0.44))
    box(ax, 0.73, 0.08, 0.17, 0.11, "stimulator", fc="#fff4e6", ec="#b5651d", bold=True)
    arrow(ax, (0.815, 0.32), (0.815, 0.19))
    for y in ys:
        arrow(ax, (0.525, y + 0.11), (0.655, 0.71), color="#2c4f8f")
        arrow(ax, (0.655, 0.36), (0.525, y + 0.04), color="#b5651d", ls="--")
    ax.text(0.59, 0.85, "spikes\n(frame, unit),\npulse onsets", fontsize=5.0, ha="center", va="bottom", color="#2c4f8f")
    ax.text(0.59, 0.20, "next\ncondition", fontsize=5.2, ha="center", color="#b5651d")
    box(ax, 0.205, 0.89, 0.32, 0.08, "streamsort watch / fit \u2192 templates.npz (atomic)", fc="#eef6ee", ec="#3a7a3a", fs=5.4, ls="--")
    arrow(ax, (0.365, 0.89), (0.365, 0.74), color="#3a7a3a", ls="--")
    ax.text(0.355, 0.866, "hot-swap on mtime change", fontsize=5.0, color="#3a7a3a", ha="right", va="center")
    fig.savefig(FIG / "plugin_architecture.png", dpi=300); fig.savefig(FIG / "plugin_architecture.svg"); plt.close(fig)


def realtime():
    R = json.loads((OUT / "harness.json").read_text())["end_to_end"]; z = np.load(OUT / "rt_factor.npz")
    L = json.loads((OUT / "worker_log.json").read_text()); T, S = L["trials"], L["stim"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
    bins = np.logspace(-2.3, 0.7, 70)
    for k, c in zip(("A", "B", "C"), ("#4c72b0", "#55a868", "#c44e52")):
        a.hist(z[k], bins=bins, histtype="step", color=c, lw=0.9, label=f"probe {k}: median {np.median(z[k]):.2f}")
    a.hist(z["serial"], bins=bins, histtype="stepfilled", color="#333333", alpha=0.25, label=f"A+B+C serial: median {np.median(z['serial']):.2f}, p99 {np.percentile(z['serial'], 99):.2f}")
    a.axvline(1.0, color="k", lw=0.8, ls="--"); a.text(0.95, 300, "real time", fontsize=6, ha="right", rotation=90, va="bottom")
    a.set_xscale("log"); a.set_xlabel("processing time / buffer duration (1024 samples)"); a.set_ylabel("buffers")
    a.set_ylim(0, a.get_ylim()[1] * 1.55); a.legend(fontsize=5.3, frameon=False, loc="upper left"); a.set_title("a  per-buffer real-time factor", loc="left", fontsize=7)
    pl = np.array([t["pulse_to_loss_ms"] for t in T]); ps = np.array([s["pulse_to_stim_ms"] for s in S])
    b.plot(np.arange(len(ps)), ps, ".", ms=3, color="#b5651d", label=f"stimulator reprogrammed (median {np.median(ps):.0f} ms)")
    b.plot(np.arange(len(pl)), pl, ".", ms=3, color="#2c4f8f", label=f"loss ready (median {np.median(pl):.0f} ms)")
    b.axhline(502, color="#2c4f8f", lw=0.6, ls=":"); b.text(len(pl) * 0.98, 470, "window end 502 ms", fontsize=5.5, ha="right", va="top", color="#2c4f8f")
    b.axhline(2000, color="k", lw=0.7, ls="--"); b.text(len(pl) * 0.98, 1930, "ITI 2000 ms", fontsize=5.5, ha="right", va="top")
    b.set_ylim(0, 2100); b.set_xlabel("trial (replay, real-time pace)"); b.set_ylabel("ms after pulse block arrived")
    b.legend(fontsize=5.5, frameon=False, loc="center right"); b.set_title(f"b  closed loop, 3 probes + worker ({len(pl)} trials)", loc="left", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "plugin_realtime_factor.png", dpi=300); fig.savefig(FIG / "plugin_realtime_factor.svg"); plt.close(fig)


architecture()
if (OUT / "rt_factor.npz").exists():
    realtime()
