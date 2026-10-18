"""Phase 1 summary figure (6 x 3 in): architecture + validation, jlh32, all three probes pooled where pooling applies.
Architecture: (1) stream -> (3) online decoding + within-ITI analysis -> stim, with (2) the initial sort as a parallel
branch on the first 600 s whose template bank rejoins (3).  Writes SVG (editable text), PDF and PNG (600 dpi).
Inputs: results/phase1_summary/snippet.npz (phase1_summary_data.py), coverage_jlh32.json (phase1_summary_coverage.py),
per-probe gate/agreement files, jlh32 pooled geometry + live-replay timing.
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.stats import spearmanr

OUT = "figures/phase1_summary"; os.makedirs(OUT, exist_ok=True)
W, H = 6.0, 3.0
FS, FS2, FS3 = 6.0, 5.0, 4.5             # role-mapped size ladder: titles+labels / annotations / ticks
try:
    apply_figure_style(sizes=(FS, FS2, FS3))  # noqa: F821  (kernel helper when run inside the session)
except NameError:
    pass
mpl.rcParams.update({"font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS3, "ytick.labelsize": FS3,
                     "legend.fontsize": FS2, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 1.5,
                     "savefig.bbox": None, "savefig.pad_inches": 0, "lines.linewidth": 0.8, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
C1, C2, C3, C4 = "#6E6E6E", "#3A6EA5", "#2E8B57", "#8E4FA3"     # stream / sort / decoding / analysis->stim
GREY = "0.35"
PROBES = {"A": "results", "B": "results/jlh32_probeB", "C": "results/jlh32_probeC"}

def ampcolor(a, amax=100):
    return plt.get_cmap("RdBu_r")(0.5 + 0.5 * np.sign(a) * (0.25 + 0.75 * np.log10(abs(a) / 5) / np.log10(amax / 5)))

fig = plt.figure(figsize=(W, H))
def ax_in(x, y, w, h, **kw): return fig.add_axes([x / W, y / H, w / W, h / H], **kw)
def title(ax, letter_, text, dy=0.05, x=None):
    p = ax.get_position(); y = p.y1 + dy / H; x0 = p.x0 if x is None else x / W
    fig.text(x0, y, letter_, fontsize=7, fontweight="bold", ha="left", va="bottom")
    fig.text(x0 + 0.11 / W, y, text, fontsize=FS, ha="left", va="bottom")
def _unused(ax, letter_, text, dy=0.05):
    p = ax.get_position(); y = p.y1 + dy / H
    fig.text(p.x0, y, letter_, fontsize=7, fontweight="bold", ha="left", va="bottom")
    fig.text(p.x0 + 0.11 / W, y, text, fontsize=FS, ha="left", va="bottom")

# ============================================================== (a) architecture
A = ax_in(0, 0, W, H, zorder=0); A.set_xlim(0, W); A.set_ylim(0, H); A.axis("off")
def box(xa, xb, ya, yb, col):
    A.add_patch(FancyBboxPatch((xa, ya), xb - xa, yb - ya, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=col, lw=0.9))
def badge(x, y, k, col):
    A.add_patch(mpl.patches.Circle((x, y), 0.062, fc=col, ec="none"))
    A.text(x, y - 0.003, str(k), ha="center", va="center", color="white", fontsize=FS2, fontweight="bold")
def arrow(p0, p1, col="0.25", lw=0.8, **kw):
    A.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=lw, color=col, shrinkA=0, shrinkB=0, **kw))
YB, YT = 2.10, 2.90
b1 = (0.14, 0.98); b2 = (1.18, 2.00, YB, 2.56); b3 = (2.30, 5.96)
# 1: stream
box(b1[0], b1[1], YB, YT, C1); badge(b1[0] + 0.10, YT - 0.10, 1, C1)
A.text(b1[0] + 0.19, YT - 0.10, "Stream", ha="left", va="center", fontsize=FS, fontweight="bold")
A.text(b1[0] + 0.06, YT - 0.20, "all probes, read\nsample-by-sample\nas a live stream", ha="left", va="top", fontsize=FS2, linespacing=1.1)
# 2: initial sort (parallel branch on a subset of the stream)
box(b2[0], b2[1], b2[2], b2[3], C2); badge(b2[0] + 0.10, b2[3] - 0.10, 2, C2)
A.text(b2[0] + 0.19, b2[3] - 0.10, "Initial sort", ha="left", va="center", fontsize=FS, fontweight="bold")
A.text(b2[0] + 0.06, b2[3] - 0.19, "Kilosort 4, probes\nin parallel →\ncausal templates", ha="left", va="top", fontsize=FS2, linespacing=1.1)
# arrows: 1 -> 3 (full stream, above box 2); 1 -> 2 (first 600 s); 2 -> 3 (template bank)
ys = 2.76; arrow((b1[1], ys), (b3[0], ys)); A.text((b1[1] + b3[0]) / 2, ys + 0.03, "full stream", ha="center", va="bottom", fontsize=FS3, color=GREY)
yb2 = 2.28; arrow((b1[1], yb2), (b2[0], yb2), col=C2); A.text((b1[1] + b2[0]) / 2, yb2 + 0.03, "600 s", ha="center", va="bottom", fontsize=FS3, color=C2)
arrow((b2[1], yb2), (b3[0], yb2), col=C2); A.text((b2[1] + b3[0]) / 2, yb2 + 0.03, "templates", ha="center", va="bottom", fontsize=FS3, color=C2)
# 3: online decoding + within-ITI analysis -> stim (same box: they run concurrently)
box(b3[0], b3[1], YB, YT, C3); badge(b3[0] + 0.10, YT - 0.10, 3, C3)
A.text(b3[0] + 0.19, YT - 0.10, "Online decoding  +  within-ITI analysis → stimulation", ha="left", va="center", fontsize=FS, fontweight="bold")
A.text(b3[0] + 0.07, YT - 0.20, "continuously: artifact suppression →\ncausal filter → whitened template matching", ha="left", va="top", fontsize=FS2, color=C3, linespacing=1.1)
A.text(4.18, YT - 0.20, "after each pulse: pooled PCA + Procrustes\n(or decoder) → loss → reprogram stimulator", ha="left", va="top", fontsize=FS2, color=C4, linespacing=1.1)
# pulse train: decoding runs continuously (green line); each pulse gets analysis + reprogramming within its ITI (purple)
x0p, dxp, npul = 2.50, 0.32, 11; yb_ = 2.19; th = 0.08
xs = x0p + dxp * np.arange(npul); frac_done = 651 / 2000
A.plot([b3[0] + 0.06, b3[1] - 0.06], [yb_, yb_], color=C3, lw=1.4, solid_capstyle="butt")
for x in xs:
    A.add_patch(Rectangle((x, yb_ + 0.005), dxp * frac_done, 0.035, fc=C4, ec="none", alpha=0.35))
    A.plot([x, x], [yb_, yb_ + th], color="k", lw=0.7)
A.text(b3[0] + 0.06, yb_ - 0.02, "decoding", ha="left", va="top", fontsize=FS3, color=C3)
A.text(xs[0], yb_ + th + 0.01, "pulses", ha="center", va="bottom", fontsize=FS3, color="k")
k = 3; xa_, xb_ = xs[k], xs[k + 1]; ytop = yb_ + th + 0.09
A.plot([xa_, xa_, xb_], [yb_ + th + 0.01, ytop, ytop], color=C4, lw=0.7)
arrow((xb_, ytop), (xb_, yb_ + th + 0.012), col=C4, lw=0.7)
A.text(xb_ + 0.05, ytop + 0.005, "next pulse uses the reprogrammed parameters", ha="left", va="center", fontsize=FS2, color=C4)
A.text(xb_ + 0.05, ytop - 0.065, "(loop timing validated in e; closed-loop control not yet run)", ha="left", va="center", fontsize=FS3, color=C4)
A.text(0.0, YT, "a", fontsize=7, fontweight="bold", ha="left", va="top")
def bracket(col, xa, xb, xstem, ystem_top, y=2.02):
    A.plot([xa, xa, xb, xb], [y - 0.035, y, y, y - 0.035], color=col, lw=0.6); A.plot([xstem, xstem], [ystem_top, y], color=col, lw=0.6)
bracket(C2, 0.06, 1.30, 1.59, b2[2]); bracket(C3, 1.44, 4.12, 2.80, YB); bracket(C4, 4.26, 5.96, 5.20, YB)

# ============================================================== (b) initial sort, 3 probes pooled (block diagonal)
blocks, n_m, n_g = [], 0, 0
for p, d in PROBES.items():
    g = json.load(open(f"{d}/gate_summary.json")); n_m += g["good_active_matched_gt0.5"]; n_g += g["n_ref_good_active"]
    agr = pd.read_csv(f"{d}/initial_sort_agreement_matrix.csv", index_col=0); rep = pd.read_csv(f"{d}/sort_agreement_report.csv", index_col=0)
    act = rep[(rep.group == "good") & (rep.n_ref_in_segment >= 50)].index
    Aa = agr.loc[[i for i in agr.index if int(i) in set(act)]].to_numpy()
    best = Aa.max(1); order = np.argsort(-best, kind="stable"); Aa = Aa[order]
    cols = []
    for r_ in Aa:
        c = int(np.argmax(r_))
        if r_[c] > 0 and c not in cols: cols.append(c)
    rest = [c for c in np.argsort(-Aa.max(0)) if c not in cols][: max(0, len(Aa) - len(cols))]
    blocks.append((p, Aa[:, cols + rest]))
nR = sum(b.shape[0] for _, b in blocks); nC = sum(b.shape[1] for _, b in blocks); M = np.full((nR, nC), np.nan); r0 = c0 = 0; spans = []
for p, b in blocks:
    M[r0:r0 + b.shape[0], c0:c0 + b.shape[1]] = b; spans.append((p, r0, r0 + b.shape[0], c0, c0 + b.shape[1])); r0 += b.shape[0]; c0 += b.shape[1]
ax = ax_in(0.30, 0.36, 0.88, 1.38)
cm = plt.get_cmap("viridis").copy(); cm.set_bad("white")
im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
for p, ra, rb, ca, cb in spans:
    ax.add_patch(Rectangle((ca - 0.5, ra - 0.5), cb - ca, rb - ra, fill=False, ec="0.6", lw=0.3))
    ax.text(ca - 3, (ra + rb) / 2, p, ha="right", va="center", fontsize=FS3, color="0.2")
ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xlabel("initial-sort unit (600 s)", labelpad=2); ax.set_ylabel("offline unit (full session)", labelpad=7)
ax.text(0.97, 0.97, f"{n_m}/{n_g} ({100 * n_m / n_g:.0f}%)\ngood units\nmatched", transform=ax.transAxes, ha="right", va="top", fontsize=FS2, color="0.1")
cax = ax_in(1.21, 0.36, 0.04, 0.50); cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 1]); cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=FS3, width=0.4, length=1.5, pad=1); cb.set_label("agreement", fontsize=FS3, labelpad=0)
title(ax, "b", "Sort matches offline")

# ============================================================== (c) raw snippet with template matches
Z = np.load("results/phase1_summary/snippet.npz", allow_pickle=True); T = np.load("results/templates_jlh32_probeA.npz", allow_pickle=True)
Y, fs, st, onset = Z["Y"], float(Z["fs"]), int(Z["start"]), int(Z["onset"])
uid = list(T["unit_ids"]); ui = {u: i for i, u in enumerate(uid)}
chans = list(range(145, 157)); off_uV = 110.0; TMAX = 30.0
ax = ax_in(1.50, 0.36, 1.62, 1.38)
tt = (np.arange(len(Y)) + st - onset) / fs * 1e3; tw = (tt >= -2) & (tt <= TMAX)
ev = json.loads(str(Z["sup_events"])); span_ms = (ev[0]["span"] / fs * 1e3 + 0.7) if ev else 1.0
ax.axvspan(-0.2, span_ms, color="0.88", lw=0, zorder=0)
for r, c in enumerate(chans):
    ax.plot(tt[tw], Y[tw, c] - r * off_uV, color="0.55", lw=0.35, zorder=1)
pal = plt.get_cmap("tab10").colors; pal = [pal[i] for i in (0, 1, 2, 4, 5, 6, 8, 9, 3)]
seen = {}
for f, u in zip(Z["frame"], Z["unit"]):
    i = ui[u]; lc = T["local_channels"][i]; tpl = T["templates"][i]; to = int(T["trough_offset"][i])
    if T["peak_channel"][i] not in chans: continue
    ts = (np.arange(tpl.shape[0]) - to + f - onset) / fs * 1e3
    if ts[-1] < -2 or ts[0] > TMAX: continue
    col = seen.setdefault(u, pal[len(seen) % len(pal)])
    for j, c in enumerate(lc):
        if c in chans and np.ptp(tpl[:, j]) >= 25:
            ax.plot(ts, tpl[:, j] - chans.index(c) * off_uV, color=col, lw=0.8, zorder=3)
ax.plot([0, 0], [off_uV * 0.75, off_uV * 1.3], color="k", lw=1.0, solid_capstyle="butt")
ax.text(0.8, off_uV * 1.05, "pulse  (grey: artifact suppressed)", fontsize=FS3, va="center", ha="left", color="0.15")
xb, yb = TMAX - 6.5, -(len(chans) - 0.1) * off_uV
ax.plot([xb, xb + 5], [yb, yb], color="k", lw=0.8, solid_capstyle="butt"); ax.text(xb + 2.5, yb - 25, "5 ms", ha="center", va="top", fontsize=FS3)
ax.plot([xb, xb], [yb, yb + 100], color="k", lw=0.8, solid_capstyle="butt"); ax.text(xb - 0.6, yb + 50, "100 µV", ha="right", va="center", fontsize=FS3)
ax.set_xlim(-2, TMAX + 0.5); ax.set_ylim(-(len(chans) + 0.6) * off_uV, off_uV * 1.4); ax.axis("off")
title(ax, "c", "Online matches, one −50 µA trial")

# ============================================================== (d) units and spikes captured, 3 probes pooled
CV = json.load(open("results/phase1_summary/coverage_jlh32.json")); PP = CV["pooled"]
rec_ = np.concatenate([CV["probes"][p]["recall"] for p in "ABC"]); prec_ = np.concatenate([CV["probes"][p]["precision"] for p in "ABC"])
ax = ax_in(3.70, 0.40, 0.44, 1.22)
rows = [("units matched", "bar", 100 * PP["units_matched"] / PP["units_active_good"], C2),
        ("all spikes", "bar", 100 * PP["coverage_all_good"], C3),
        ("spikes per unit", "box", 100 * rec_, C3),
        ("precision per unit", "box", 100 * prec_, C3)]
rng = np.random.default_rng(1)
for i, (lab, kind, v, c) in enumerate(rows):
    y = len(rows) - 1 - i
    if kind == "bar":
        ax.barh(y, v, height=0.55, color=c, alpha=0.85, lw=0); ax.text(v + 3, y, f"{v:.0f}%", ha="left", va="center", fontsize=FS2)
    else:
        ax.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=0.8, color=c, alpha=0.45, lw=0, zorder=2)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(Rectangle((q1, y - 0.28), q3 - q1, 0.56, fill=False, ec="k", lw=0.6, zorder=3)); ax.plot([md, md], [y - 0.28, y + 0.28], color="k", lw=1.0, zorder=3)
        ax.text(md, y + 0.31, f"{md:.0f}%", ha="center", va="bottom", fontsize=FS3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=FS3)
ax.set_xlim(0, 112); ax.set_xticks([0, 50, 100]); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("% of offline sort", labelpad=1)
ax.text(1.0, 1.01, f"{len(rec_)} matched units, held-out data", transform=ax.transAxes, fontsize=FS3, ha="right", va="bottom", color=GREY)
title(ax, "d", "Units & spikes captured", x=3.24, dy=0.13)

# ============================================================== (e) within-ITI timeline (3 probes decoded live)
PU = pd.read_csv("results/jlh32_population/iti_population_push.csv"); p5 = PU[PU.window_s == 0.5]; dec = p5.latency_ms - p5.geometry_ms
LT = json.load(open("results/jlh32_population/window_loss_timing.json")); REPROG = float(os.environ.get("REPROG_MS", "100")); ITI = 2000.0
ax = ax_in(4.30, 1.42, 1.66, 0.30)
t_dec, t_gl = dec.mean(), p5.geometry_ms.mean() + LT["procrustes_ms_mean"]
x_dec = 502.0; x_gl = x_dec + t_dec; x_rp = x_gl + t_gl; done = x_rp + REPROG
for x0, w_, c in [(1.0, 501.0, "#E7A977"), (x_dec, t_dec, C3), (x_gl, max(t_gl, 4), C4), (x_rp, REPROG, "0.25"), (ITI - 500, 500, "0.85")]:
    ax.barh(0, w_, left=x0, height=0.9, color=c, lw=0)
ax.text(251, 0, "0.5 s window", ha="center", va="center", fontsize=FS3, color="0.1")
ax.text(1750, 0, "baseline", ha="center", va="center", fontsize=FS3, color="0.2")
for i, (c, lab) in enumerate([(C3, f"decode {t_dec:.0f} ± {dec.std():.0f} ms"), (C4, f"geometry+loss {t_gl:.1f} ms"), ("0.25", f"reprogram {REPROG:.0f} ms (assumed)")]):
    yy = 0.30 - i * 0.30
    ax.add_patch(Rectangle((690, yy - 0.10), 30, 0.20, color=c, lw=0)); ax.text(733, yy, lab, fontsize=FS3, ha="left", va="center", color="0.15")
ax.axvline(0, color="k", lw=0.8); ax.axvline(ITI, color="k", lw=0.8)
ax.set_xlim(-20, ITI + 20); ax.set_ylim(-0.5, 0.5); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 500, 1000, 1500, 2000]); ax.set_xlabel("time after pulse (ms); 3 probes live, 2 s ITI", labelpad=1)
title(ax, "e", f"Reprogrammed {done:.0f} ms after the pulse", dy=0.04)

# ============================================================== (f) pooled PC1-PC2 (online) and (g) amplitude controls geometry
P = np.load("results/jlh32_population/geometry_per_trial_sup.npz", allow_pickle=True)
Zo, Bv, mu, cond = P["Z_online"], P["basis"], P["mean"], P["cond"]
proj = Zo + (mu @ Bv)[None, None, :]                 # baseline-referenced rates projected: origin = baseline
t_ax = 2.0 + 10 * (np.arange(Zo.shape[1]) + 0.5); win = (t_ax >= 150) & (t_ax < 260)
fam = "/bipolar/100/c3-14"; amps = [-100, -50, -25, -5, 5, 25, 50, 100]
ax = ax_in(4.30, 0.44, 0.66, 0.60)
for a in amps:
    m = proj[cond == f"{a}{fam}"].mean(0); ax.plot(m[:, 0], m[:, 1], color=ampcolor(a), lw=0.9 if abs(a) >= 50 else 0.6)
ax.plot(0, 0, "+", color="k", ms=4, mew=0.6, zorder=5); ax.text(6, 8, "baseline", fontsize=FS3, color="k", ha="left", va="bottom")
ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.08); [ax.spines[s].set_visible(False) for s in ("left", "bottom")]
xl, yl_ = ax.get_xlim(), ax.get_ylim(); dx_, dy_ = (xl[1] - xl[0]) * 0.22, (yl_[1] - yl_[0]) * 0.22
ax.annotate("", xy=(xl[0] + dx_, yl_[0]), xytext=(xl[0], yl_[0]), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.2", mutation_scale=4))
ax.annotate("", xy=(xl[0], yl_[0] + dy_), xytext=(xl[0], yl_[0]), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.2", mutation_scale=4))
ax.text(xl[0] + dx_ * 1.1, yl_[0], "PC1", fontsize=FS3, va="center"); ax.text(xl[0], yl_[0] + dy_ * 1.15, "PC2", fontsize=FS3, ha="center", va="bottom")
pf = ax.get_position(); ca = fig.add_axes([pf.x0 + 0.40 * pf.width, pf.y0 - 0.065, 0.55 * pf.width, 0.012])
ca.imshow(np.array([ampcolor(a) for a in amps])[None], aspect="auto", interpolation="nearest"); ca.set_xticks([]); ca.set_yticks([]); [s.set_visible(False) for s in ca.spines.values()]
fig.text(pf.x0 + 0.38 * pf.width, pf.y0 - 0.059, "−100", fontsize=FS3, ha="right", va="center"); fig.text(pf.x0 + 0.97 * pf.width, pf.y0 - 0.059, "+100 µA", fontsize=FS3, ha="left", va="center")
title(ax, "f", "Geometry", dy=0.03)

ax = ax_in(5.36, 0.44, 0.60, 0.54); pos = {a: i for i, a in enumerate(amps)}
pc1 = proj[:, win, 0].mean(1)
for a in amps:
    v = pc1[cond == f"{a}{fam}"]; x = pos[a] + rng.uniform(-0.22, 0.22, len(v))
    ax.scatter(x, v, s=0.8, color=ampcolor(a), alpha=0.45, lw=0, zorder=2); ax.plot([pos[a] - 0.34, pos[a] + 0.34], [np.median(v)] * 2, color="k", lw=0.9, zorder=3)
mono = [np.median(pc1[cond == f"{a}/monopolar/100/c3-0"]) for a in amps]
ax.plot([pos[a] for a in amps], mono, "o", ms=2.2, mfc="white", mec="0.2", mew=0.5, zorder=4)
ax.axhline(0, color="0.6", lw=0.4, zorder=1)
ax.set_xticks(range(len(amps))); ax.set_xticklabels([("−" if a < 0 else "+") + str(abs(a)) if abs(a) in (25, 100) else "" for a in amps]); ax.set_xlim(-0.6, len(amps) - 0.4)
ax.set_xlabel("amplitude (µA)", labelpad=1); ax.set_ylabel("PC1 at trough", labelpad=0); ax.set_ylim(-175, 115); ax.set_yticks([-150, -75, 0, 75])
sel = np.isin(cond, [f"{a}{fam}" for a in amps]); aa = np.array([int(c.split("/")[0]) for c in cond[sel]]); rho = spearmanr(np.abs(aa), -pc1[sel])
ax.text(0.03, 0.97, f"single trials, ρ = {rho.statistic:.2f}", transform=ax.transAxes, fontsize=FS3, ha="left", va="top")
ax.plot([0.05], [0.80], "o", ms=2.2, mfc="white", mec="0.2", mew=0.5, transform=ax.transAxes, clip_on=False)
ax.text(0.10, 0.80, "monopolar median", transform=ax.transAxes, fontsize=FS3, ha="left", va="center")
title(ax, "g", "Amplitude sets it", dy=0.03, x=5.12)

for ext in ("svg", "pdf", "png"):
    fig.savefig(f"{OUT}/phase1_summary.{ext}", dpi=600 if ext == "png" else None)
fig.set_dpi(100); fig.canvas.draw(); r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text()[:28], b.get_text()[:28]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
outside = [t.get_text()[:28] for t, bb in texts if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1 + 1 or bb.y1 > fig.bbox.y1 + 1]
print("overlaps:", ov); print("outside:", outside)
print(json.dumps(dict(units=f"{n_m}/{n_g}", recall_median=float(np.median(rec_)), precision_median=float(np.median(prec_)), n_units_validated=len(rec_),
                      coverage_all_good=PP["coverage_all_good"], rho=rho.statistic, done_ms=done, decode_mean=t_dec, decode_sd=dec.std(), units_in_snippet=len(seen))))
