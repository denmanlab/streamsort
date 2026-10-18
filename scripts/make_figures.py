"""All per-probe figures from saved results (no re-decoding).  REC/PROBE via ctx.
Figures: initial_sort_gate.png, validation_summary.png, near_pulse_diagnosis.png,
geometry_trajectories.png, geometry_comparison.png, trial_alignment.png,
demo_live_timeline.png, iti_budget.png.  Also writes validation_by_run_sup.csv, iti_budget.csv."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ctx, figstyle, geometry_step7 as g7
from streamsort.io import OpenEphysSession
figstyle.apply_figure_style(sizes=(8, 7, 6))
TITLE = f"{ctx.REC} probe {ctx.PROBE}"

def overlaps(fig):
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    return [(a.get_text()[:20], b.get_text()[:20]) for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:]
            if ba.overlaps(bb) and len(a.get_text()) > 6 and len(b.get_text()) > 6 and not a.get_text().startswith("$")]

def save(fig, name):
    fig.savefig(ctx.F(name), dpi=300, bbox_inches="tight"); ov = overlaps(fig); plt.close(fig)
    print(f"[fig] {name}" + (f"  overlaps: {ov[:4]}" if ov else ""), flush=True)

def load():
    D = {}
    D["tr"] = pd.read_csv(ctx.R("trials.csv")); D["sess"] = json.load(open(ctx.R("session.json")))
    D["on"] = dict(np.load(ctx.R("online_spikes_all_trials_sup.npz"))); D["rf"] = dict(np.load(ctx.R("reference_spikes_all_trials_sup.npz")))
    D["units"] = np.unique(D["rf"]["ks4_to_nwb"][:, 0]); D["fs"] = float(D["on"]["meta_fs"]); D["off"] = D["on"]["meta_pulse_offset"]
    D["q"] = (D["tr"].amplitude.abs() * D["tr"].pulse_duration).to_numpy(); D["cond"] = ctx.cond_labels(D["tr"])
    D["G"] = np.where(D["q"] >= 5000, "high", np.where(D["q"] >= 1500, "mid", "low"))
    return D

def by_run(D, tol=12):
    on, rf, units, fs = D["on"], D["rf"], D["units"], D["fs"]; run = on["meta_trial_run"]
    m = np.isin(on["unit"], units); of, ou, ot = on["frame"][m], on["unit"][m], on["trial"][m]
    tp = np.zeros(len(of), bool)
    for u in units:
        a = np.flatnonzero(ou == u); r = np.sort(rf["frame"][rf["unit"] == u])
        if len(a) == 0 or len(r) < 2: continue
        i = np.clip(np.searchsorted(r, of[a]), 1, len(r) - 1); tp[a] = np.minimum(np.abs(of[a] - r[i - 1]), np.abs(of[a] - r[i])) <= tol
    w0 = on["meta_window_start"]; tmin = (w0 - w0[0]) / fs / 60; rows = []
    for Rn in np.unique(run):
        k = run[ot] == Rn; nr = (run[rf["trial"]] == Rn).sum(); t_ = tp[k].sum(); P = t_ / max(k.sum(), 1); Rc = t_ / max(nr, 1)
        rows.append(dict(run=int(Rn), t_min=float(tmin[run == Rn].mean()), n_online=int(k.sum()), n_ref=int(nr), precision=P, recall=Rc,
                         f1=2 * P * Rc / max(P + Rc, 1e-9), online_over_ref=k.sum() / max(nr, 1)))
    RS = pd.DataFrame(rows); RS.to_csv(ctx.R("validation_by_run_sup.csv"), index=False); return RS

def fig_gate():
    gs_ = json.load(open(ctx.R("gate_summary.json"))); rep = pd.read_csv(ctx.R("sort_agreement_report.csv"), index_col=0)
    agr = pd.read_csv(ctx.R("initial_sort_agreement_matrix.csv"), index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.5, width_ratios=[1.3, 1, 1]))
    A = agr.values[: min(150, agr.shape[0]), : min(150, agr.shape[1])]
    im = axs[0].imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axs[0].set_xlabel("Initial-sort unit (Hungarian order)"); axs[0].set_ylabel("KS2.5 unit"); axs[0].set_title("a  Agreement matrix", loc="left")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=5)
    act = rep[(rep.group == "good") & (rep.n_ref_in_segment >= 50)] if "n_ref_in_segment" in rep else rep[rep.group == "good"]
    axs[1].hist(act.agreement.fillna(0), np.linspace(0, 1, 21), color="0.5"); axs[1].axvline(0.5, color="#C44E52", lw=0.8)
    axs[1].set_xlabel("Best agreement"); axs[1].set_ylabel("Active good KS2.5 units")
    axs[1].set_title(f"b  {gs_['good_active_matched_gt0.5']}/{gs_['n_ref_good_active']} matched", loc="left")
    seg_s = 600.0; rate = act.n_ref_in_segment / seg_s if "n_ref_in_segment" in act else None
    if rate is not None:
        bins = [(0, 1, "<1 Hz"), (1, 3, "1–3 Hz"), (3, 1e9, "≥3 Hz")]; fr, lab_, nn = [], [], []
        for lo, hi, l in bins:
            m = (rate >= lo) & (rate < hi); fr.append(float((act.agreement[m].fillna(0) > 0.5).mean()) if m.any() else 0); lab_.append(l); nn.append((int((act.agreement[m].fillna(0) > 0.5).sum()), int(m.sum())))
        axs[2].bar(range(3), fr, color=["#C44E52", "#8172B2", "#4C72B0"], width=0.6)
        for i, (a_, b_) in enumerate(nn): axs[2].text(i, fr[i] + 0.02, f"{a_}/{b_}", ha="center", fontsize=5.5)
        axs[2].set_xticks(range(3)); axs[2].set_xticklabels(lab_); axs[2].set_ylim(0, 1.1); axs[2].set_xlabel("KS2.5 rate in sort window")
    axs[2].set_ylabel("Fraction matched"); axs[2].set_title("c  By firing rate", loc="left")
    fig.suptitle(f"{TITLE}: initial sort (Kilosort 4, first 600 s) vs full KS2.5 — {'PASS' if gs_['PASS'] else 'FAIL'}", fontsize=7, x=0.02, ha="left", y=1.04)
    save(fig, "initial_sort_gate.png")

def fig_validation(D, RS):
    V = pd.read_csv(ctx.R("validation_units_sup.csv")); LT = pd.read_csv(ctx.R("validation_by_latency_sup.csv"))
    TC = pd.read_csv(ctx.R("validation_trial_counts_sup.csv")); on, rf, off, fs = D["on"], D["rf"], D["off"], D["fs"]; run = on["meta_trial_run"]
    st = OpenEphysSession(ctx.REC_DIR).stream(ctx.STREAM); w0 = on["meta_window_start"]
    seg_end_min = (st.time_to_sample(st.sample_to_time(0) + 600.0) - w0[0]) / fs / 60
    fig = plt.figure(figsize=(7.4, 8.2)); gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.55, height_ratios=[1, 1, 1.15])
    ax = fig.add_subplot(gs[0, 0]); sc = ax.scatter(V.recall_heldout, V.precision_heldout, c=V.template_trough_uV, cmap="viridis", s=10, lw=0, vmin=20, vmax=200)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("Recall (held-out)"); ax.set_ylabel("Precision (held-out)")
    ax.set_title(f"a  {len(V)} matched units,\n    after the sort window", loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03); cb.ax.set_title("trough\n(µV)", fontsize=5.5, pad=3); cb.ax.tick_params(labelsize=5)
    ax = fig.add_subplot(gs[0, 1]); g = TC.groupby("ks4_unit")[["n_online", "n_ref"]].sum(); g = g[(g > 0).all(1)]
    ax.loglog(g.n_ref, g.n_online, "o", ms=3, color="#4C72B0", alpha=0.8); lim = [max(10, g.values.min() * 0.8), g.values.max() * 1.2]; ax.plot(lim, lim, "k--", lw=0.7)
    import matplotlib.ticker as mtk
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(mtk.LogLocator(base=10)); a_.set_minor_formatter(mtk.NullFormatter())
    ax.set_xlabel("KS2.5 spikes (all trials)"); ax.set_ylabel("Online spikes"); ax.set_title("b  Counts vs reference", loc="left")
    ax.text(0.04, 0.92, "median per-trial\ncount r = %.2f" % V.count_corr_trials.median(), transform=ax.transAxes, fontsize=5.5, va="top")
    ax = fig.add_subplot(gs[0, 2]); ax.plot(RS.t_min, RS.f1, "o-", color="#DD8452", ms=3, lw=1)
    if seg_end_min > 0: ax.axvspan(min(0, RS.t_min.min()), seg_end_min, color="0.85", zorder=0)
    ax.set_ylim(0, 1); ax.set_xlabel("Time from first trial (min)"); ax.set_ylabel("F1 vs KS2.5 (per run)"); ax.set_title("c  Accuracy across the session\n    (grey: sort window)", loc="left")
    ax = fig.add_subplot(gs[1, 0:2]); mid = np.sqrt(LT.lat_lo_ms * LT.lat_hi_ms)
    ax.semilogx(mid, LT.precision, "o-", color="#C44E52", ms=4, label="precision"); ax.semilogx(mid, LT.recall, "s-", color="#4C72B0", ms=4, label="recall")
    ax.set_xticks([2, 4, 7.5, 15, 35, 75, 200, 650, 1450]); ax.set_xticklabels(["1–3", "3–5", "5–10", "10–20", "20–50", "50–100", "0.1–0.3 s", "0.3–1 s", "1–1.9 s"], fontsize=5.5)
    ax.set_xlabel("Time since end of pulse (ms)"); ax.set_ylabel("Online vs KS2.5"); ax.set_ylim(0, 1); ax.set_title("d  Agreement vs time since pulse", loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax = fig.add_subplot(gs[1, 2]); cats = ["insample", "heldout", "early", "late"]; lab = ["sort\nwindow", "after\nsort", "first\nquarter", "last\nquarter"]
    vals = [np.nanmedian(V[f"f1_{c}"]) if f"f1_{c}" in V and V[f"f1_{c}"].notna().any() else np.nan for c in cats]
    ax.bar(range(4), np.nan_to_num(vals), color=["#4C72B0", "#DD8452", "#4C72B0", "#DD8452"], width=0.6)
    for i, v in enumerate(vals): ax.text(i, (0 if np.isnan(v) else v) + 0.02, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=5.5); ax.set_ylim(0, 1); ax.set_ylabel("Median unit F1"); ax.set_title("e  Generalisation", loc="left")
    tr = D["tr"]; neg = tr[(tr.amplitude < 0) & ((tr.amplitude.abs() * tr.pulse_duration) == 5000)]
    rsel = int(neg.run.iloc[0]) if len(neg) else int(tr.assign(q=D["q"]).groupby("run").q.first().idxmax())
    trR = np.flatnonzero(run == rsel); cl = D["cond"][trR[0]]
    good = V[V.f1_all >= 0.8].sort_values("n_ref_all", ascending=False).ks4_unit.to_numpy()[:4]
    if len(good) < 4: good = V.sort_values("f1_all", ascending=False).ks4_unit.to_numpy()[:4]
    sub = gs[2, :].subgridspec(2, 4, height_ratios=[1.4, 1], hspace=0.15, wspace=0.35); bins = np.arange(0, 301, 5)
    for j, u in enumerate(good):
        axr = fig.add_subplot(sub[0, j]); axp = fig.add_subplot(sub[1, j], sharex=axr)
        for src, col, dy in [(rf, "0.25", 0.0), (on, "#DD8452", 0.35)]:
            m = (src["unit"] == u) & np.isin(src["trial"], trR); t_ms = (src["frame"][m] - off[src["trial"][m]]) / fs * 1e3; yy = np.searchsorted(trR, src["trial"][m]); k = t_ms <= 300
            axr.plot(t_ms[k], yy[k] + dy, "|", ms=1.6, color=col, mew=0.6); h, _ = np.histogram(t_ms[k], bins); axp.plot(bins[:-1] + 2.5, h / len(trR) / 0.005, color=col, lw=0.9)
        axr.set_title(f"unit {u}  F1 {V.set_index('ks4_unit').f1_all[u]:.2f}", fontsize=6); axr.set_yticks([]); plt.setp(axr.get_xticklabels(), visible=False)
        if j == 0:
            axr.set_ylabel("Trial"); axp.set_ylabel("Rate (Hz)")
            axr.text(0, 1.32, f"f  Run {rsel} ({cl}): KS2.5 (grey) vs online (orange), {len(trR)} trials", transform=axr.transAxes, fontsize=7)
        axp.set_xlabel("ms after pulse")
    fig.suptitle(TITLE, fontsize=8, x=0.02, ha="left", y=0.995)
    save(fig, "validation_summary.png")

def fig_near_pulse(D):
    on, rf, units, fs, off, G = D["on"], D["rf"], D["units"], D["fs"], D["off"], D["G"]; nU = len(units)
    edges = np.array([1, 2, 3, 5, 10, 20, 50, 100, 300, 1000]); mid = np.sqrt(edges[:-1] * edges[1:])
    def rate(src, grp):
        m = np.isin(src["unit"], units) & (G[src["trial"]] == grp); lat = (src["frame"][m] - off[src["trial"][m]]) / fs * 1e3
        h, _ = np.histogram(lat, edges); return h / max((G == grp).sum() * nU * 1, 1) / (np.diff(edges) / 1e3)
    base = rate(rf, "low")[-1] if (G == "low").any() else rate(rf, "mid")[-1]
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw=dict(wspace=0.38, hspace=0.6)); axs = axs.ravel()
    for ax, grp in [(axs[0], "high"), (axs[1], "low")]:
        if not (G == grp).any(): ax.axis("off"); continue
        ax.semilogx(mid, rate(on, grp), "o-", color="#DD8452", ms=2.5, lw=1.1, label="online (artifact suppressor)")
        ax.semilogx(mid, rate(rf, grp), "s-", color="#4C72B0", ms=2.5, lw=0.9, label="KS2.5 reference"); ax.axhline(base, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("Rate (Hz per unit)"); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_title(f"{'a' if grp == 'high' else 'b'}  {'High' if grp == 'high' else 'Low'} charge, {(G == grp).sum()} trials", loc="left")
    axs[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    ax = axs[2]
    if os.path.exists(ctx.R("exp_artifact_inject.csv")):
        inj = pd.read_csv(ctx.R("exp_artifact_inject.csv")); q_ = inj.amp.abs() * inj.dur; inj["grp"] = np.where(q_ >= 5000, "high", np.where(q_ >= 1500, "mid", "low"))
        nUu = int(pd.read_csv(ctx.R("sorting_report.csv")).usable.sum()); bw = dict(zip([1, 3, 5, 10, 20, 50], [2, 2, 5, 10, 30, 250])); x = np.array([2, 4, 7.5, 15, 35, 175])
        for grp, c in [("high", "#C44E52"), ("mid", "#8172B2"), ("low", "#55A868")]:
            d = inj[inj.grp == grp].groupby("lo").agg(n=("trial", "size"), new=("n_new", "sum"), lost=("n_lost", "sum"), clean=("n_clean", "sum"))
            if not len(d): continue
            w = d.n * d.index.map(bw).to_numpy() / 1e3 * nUu
            ax.semilogx(x[: len(d)], d.new / w, "o-", color=c, ms=2.5, lw=1); ax.semilogx(x[: len(d)], (d.lost / d.clean.clip(lower=1)) * 100 / 5, "x--", color=c, ms=3, lw=0.7)
        ax.set_xticks([2, 5, 10, 50, 100]); ax.set_xticklabels(["2", "5", "10", "50", "100"])
        h = [Line2D([], [], color=c, lw=1.2, label=l) for l, c in [("high charge", "#C44E52"), ("mid", "#8172B2"), ("low", "#55A868")]]
        h += [Line2D([], [], color="0.3", marker="o", ms=2.5, lw=1, label="added detections"), Line2D([], [], color="0.3", marker="x", ls="--", lw=0.7, label="lost spikes")]
        ax.legend(handles=h, frameon=False, fontsize=5.5, loc="upper right")
    ax.set_xlabel("Time after injected pulse (ms)"); ax.set_ylabel("Added detections (Hz/unit)\nor lost spikes (% / 5)"); ax.set_title("c  Median artifact injected into quiet data", loc="left")
    ax = axs[3]
    if os.path.exists(ctx.R("diag_early_waveforms.csv")):
        wf = pd.read_csv(ctx.R("diag_early_waveforms.csv"))
        ok = wf.groupby("unit").filter(lambda d: (d.set_index("cls").n.reindex(["2-5", "late"]).fillna(0) >= 15).all())
        cls = ["1-2", "2-5", "5-10", "late", "2-5_rand"]
        for i, c in enumerate(cls):
            v = ok[ok.cls == c].cos_sta.to_numpy()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=3, color="0.6", lw=0)
            if len(v): ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#DD8452" if c != "2-5_rand" else "k", lw=1.6)
        ax.set_xticks(range(5)); ax.set_xticklabels(["1–2", "2–5", "5–10", "≥20", "2–5\nrandom times"], fontsize=5.5); ax.set_ylim(-0.4, 1.05); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"d  Waveform fidelity ({ok.unit.nunique()} units)", loc="left")
    ax.set_ylabel("STA–template cosine"); ax.set_xlabel("Detection latency (ms)")
    fig.suptitle(TITLE, fontsize=8, x=0.02, ha="left", y=1.0)
    save(fig, "near_pulse_diagnosis.png")

def family(D):
    """The montage/contact family with the most amplitude levels (for trajectory plots)."""
    tr = D["tr"]; f = tr.groupby(["polarity", "contact_negative", "contact_positive", "pulse_duration"]).amplitude.nunique().sort_values(ascending=False)
    pol, cn, cp, du = f.index[0]
    sub = tr[(tr.polarity == pol) & (tr.contact_negative == cn) & (tr.contact_positive == cp) & (tr.pulse_duration == du)]
    amps = sorted(sub.amplitude.unique()); labs = {a: ctx.cond_labels(sub[sub.amplitude == a].iloc[:1])[0] for a in amps}
    return f"{pol}, {du} µs, c{cn}-{cp}", amps, labs

def ampcolor(a, amax):
    return plt.get_cmap("RdBu_r")(0.5 + 0.5 * np.sign(a) * (0.25 + 0.75 * np.log10(abs(a) / 5) / max(np.log10(amax / 5), 1e-6)))

def fig_geometry(D):
    P = np.load(ctx.R("geometry_per_trial_sup.npz"), allow_pickle=True); Zo, Zr, cond = P["Z_online"], P["Z_ks25"], P["cond"]
    sw = pd.read_csv(ctx.R("geometry_sweep_sup.csv")); tm = pd.read_csv(ctx.R("geometry_timing.csv")); summ = json.load(open(ctx.R("geometry_comparison_summary.json")))
    dm = json.load(open(ctx.R("demo_live_summary.json"))) if os.path.exists(ctx.R("demo_live_summary.json")) else None
    t_ax = g7.WIN0_MS + 10 * (np.arange(Zo.shape[1]) + 0.5); fam, amps, labs = family(D); amax = max(abs(a) for a in amps)
    So = {c: Zo[cond == c].mean(0) for c in np.unique(cond)}; Sr = {c: Zr[cond == c].mean(0) for c in np.unique(cond)}
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), gridspec_kw=dict(wspace=0.5, hspace=0.68)); axs = axs.ravel()
    ax = axs[0]
    for a in amps:
        c = labs[a]; ax.plot(So[c][:, 0], So[c][:, 1], "-", color=ampcolor(a, amax), lw=1.1); ax.plot(Sr[c][:, 0], Sr[c][:, 1], "--", color=ampcolor(a, amax), lw=0.7)
        ax.plot(So[c][0, 0], So[c][0, 1], "o", color=ampcolor(a, amax), ms=2.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("a  Condition means", loc="left")
    ax.legend(handles=[Line2D([], [], color="0.3", lw=1.1, label="online"), Line2D([], [], color="0.3", lw=0.7, ls="--", label="KS2.5"),
                       Line2D([], [], color="0.3", marker="o", ls="", ms=2.5, label="first bin")], frameon=False, fontsize=5.5, loc="best")
    ax = axs[1]
    for a in amps: ax.plot(t_ax, So[labs[a]][:, 0], color=ampcolor(a, amax), lw=1, label=f"{a:+d} µA")
    ax.set_xlim(0, 505); ax.set_xticks([0, 100, 200, 300, 400, 500])
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.42 * (y1 - y0), y1)
    ax.legend(frameon=False, fontsize=4.6, ncol=4, loc="lower left", handlelength=0.9, columnspacing=0.5, handletextpad=0.3)
    ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("PC1 (online)"); ax.set_title("b  PC1 time course", loc="left")
    ax = axs[2]; c0 = labs[min(amps)]; idx = np.flatnonzero(cond == c0); d_same = P["d_same"]; ex = idx[np.argsort(d_same[idx])[[5, len(idx) // 2, -5]]]
    sp = np.ptp(Zr[idx][:, :, 0]) / 2.0
    for j, i in enumerate(ex):
        ax.plot(t_ax, Zr[i][:, 0] - j * sp, color="0.45", lw=0.8); ax.plot(t_ax, Zo[i][:, 0] - j * sp, color="#DD8452", lw=1)
        ax.text(560, Zo[i][-1, 0] - j * sp, f"d={d_same[i]:.2f}", fontsize=5, va="center")
    ax.set_xlim(0, 505); ax.set_xticks([0, 100, 200, 300, 400, 500]); ax.set_yticks([])
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.22 * (y1 - y0), y1)
    ax.legend(handles=[Line2D([], [], color="#DD8452", label="online"), Line2D([], [], color="0.45", label="KS2.5")], frameon=False, fontsize=5.5, loc="lower right")
    ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("PC1, single trials (offset)"); ax.set_title(f"c  {min(amps)} µA: same trial", loc="left")
    ax = axs[3]; s = sw[sw.sort != "online_vs_ks25"].copy(); s["lab"] = s.bin_ms.astype(str) + "/" + s.smooth_ms.astype(str)
    labs_ = s[(s.sort == "online") & (s.metric == "euclid")].lab.tolist(); xx = np.arange(len(labs_))
    for (srt, met), st_ in [(("online", "euclid"), dict(color="#DD8452", marker="o")), (("ks25", "euclid"), dict(color="0.4", marker="s")),
                            (("online", "procrustes"), dict(color="#DD8452", marker="o", ls="--", mfc="none")), (("ks25", "procrustes"), dict(color="0.4", marker="s", ls="--", mfc="none"))]:
        v = s[(s.sort == srt) & (s.metric == met)].set_index("lab").loc[labs_].acc26
        ax.plot(xx, v, ms=2.5, lw=0.9, label=f"{'online' if srt == 'online' else 'KS2.5'}, {'Euclid' if met == 'euclid' else 'Procrustes'}", **st_)
    ch = summ["chance"]; ax.axhline(ch, color="k", lw=0.5, ls=":"); ax.text(0, ch + 0.006, "chance", fontsize=5)
    if "10/20" in labs_: ax.axvline(labs_.index("10/20"), color="#4C72B0", lw=0.6, alpha=0.6)
    ax.set_xticks(xx); ax.set_xticklabels(labs_, rotation=60, fontsize=5); ax.set_xlabel("bin / smoothing σ (ms)", labelpad=1)
    ax.set_ylabel(f"{summ['n_conditions']}-way accuracy, single trials"); ax.set_ylim(0, max(0.3, s.acc26.max() * 1.6))
    ax.set_title("d  Condition information", loc="left"); ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=2, columnspacing=0.6)
    ax.text(0.02, 0.78, "drift-controlled\n(same-chunk baseline removed)", transform=ax.transAxes, fontsize=5, va="top", color="0.3")
    ax = axs[4]; bb = np.linspace(0, 1.3, 40)
    ax.hist(P["d_same"], bb, histtype="step", color="#DD8452", lw=1.1, label=f"online vs KS2.5,\nsame trial (median {np.median(P['d_same']):.2f})")
    ax.hist(P["d_ref_own"], bb, histtype="step", color="0.4", lw=1.1, label=f"KS2.5 trial vs own\ncondition mean ({np.median(P['d_ref_own']):.2f})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.legend(frameon=False, fontsize=5, loc="upper left")
    ax.set_xlabel("Procrustes distance"); ax.set_ylabel("Trials"); ax.set_title("e  Decoder distortion", loc="left")
    ax = axs[5]; tt = tm[tm.smooth_ms == tm.bin_ms]
    for o, l, c in [("total_fixed", "fixed basis: bin+project+metrics", "#4C72B0"), ("refit_pca", "per-trial PCA refit", "#DD8452"),
                    ("refit_isomap", "per-trial Isomap", "#55A868"), ("refit_fa", "per-trial FA", "#C44E52")]:
        v = tt[tt.operation == o].sort_values("bin_ms"); ax.semilogy(v.bin_ms, v.p99_ms, "o-", color=c, ms=2.5, lw=1, label=l)
    if dm:
        p99 = dm["pull_decode_ms"]["1.9"]["p99"]; ax.axhline(p99, color="k", lw=0.6, ls="--"); ax.text(52, p99 * 0.72, f"spike decode p99 ({p99:.0f} ms)", fontsize=5, ha="right", va="top")
    ax.set_ylim(0.05, 2e5); ax.set_xticks([5, 10, 25, 50]); ax.set_xlabel("Bin width (ms)"); ax.set_ylabel("p99 time per trial (ms)")
    ax.set_title("f  Geometry cost per trial", loc="left"); ax.legend(frameon=False, fontsize=4.8, loc="upper right")
    fig.suptitle(f"{TITLE} — amplitude series in a–c: {fam}", fontsize=8, x=0.02, ha="left", y=1.0)
    save(fig, "geometry_trajectories.png")

def fig_comparison():
    M = np.load(ctx.R("geometry_mds.npz"), allow_pickle=True); summ = json.load(open(ctx.R("geometry_comparison_summary.json")))
    order, Do, Dr, Yr, Yo = M["order"], M["Do"], M["Dr"], M["Yr"], M["Yo"]; mont = M["mont"]; n = len(order); iu = np.triu_indices(n, 1)
    lc = np.log10(M["charge"].astype(float)); multi_dur = len(set(M["dur"].tolist())) > 1
    labs_c = [f"{int(a):+d} µA {m[:3]} " + (f"{int(d_)} µs " if multi_dur else "") + f"c{c}" for a, m, c, d_ in zip(M["amp"], mont, M["cneg"], M["dur"])]
    vmax = np.percentile(np.r_[Do[iu], Dr[iu]], 98)
    fig = plt.figure(figsize=(7.4, 7.0)); gs2 = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.12, height_ratios=[1.5, 1], width_ratios=[1, 1, 0.55])
    for k, (D_, ttl) in enumerate([(Dr, "a  KS2.5 reference"), (Do, "b  Online decoder")]):
        ax = fig.add_subplot(gs2[0, k]); im = ax.imshow(D_, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(labs_c, fontsize=4.4, rotation=90); ax.set_yticklabels(labs_c if k == 0 else [], fontsize=4.4)
        ax.tick_params(length=1.5, pad=1); ax.set_title(ttl, loc="left")
        fam = [f"{m}{c}" for m, c in zip(mont, M["cneg"])]
        for i in range(1, n):
            if fam[i] != fam[i - 1]: ax.axhline(i - 0.5, color="w", lw=0.8); ax.axvline(i - 0.5, color="w", lw=0.8)
    tax = fig.add_subplot(gs2[0, 2]); tax.axis("off")
    txt = (f"Rows: montage, contact,\nthen charge (pC), then sign.\n\nOnline vs KS2.5 ({len(iu[0])} pairs):\n Procrustes ρ = {summ['rho_procrustes_online_vs_ks25']:.2f}\n"
           f"  (Mantel p = {summ['mantel_p']:.4f})\n Euclidean  ρ = {summ['rho_euclid_online_vs_ks25']:.2f}\n\nSplit-half reliability:\n KS2.5 odd vs even  ρ = {summ['split_half_ks25']:.2f}\n"
           f" online even vs\n  KS2.5 odd         ρ = {summ['cross_half_online_even_vs_ks25_odd']:.2f}\n\nWhat sets distance (KS2.5):\n |Δ log charge| ρ = {summ['rho_D_vs_dlogcharge_ks25']:.2f}\n"
           f" current sign   ρ = {summ['rho_D_vs_sign_ks25']:+.2f}\n montage        ρ = {summ['rho_D_vs_montage_ks25']:+.2f}")
    tax.text(0.28, 0.98, txt, transform=tax.transAxes, fontsize=5.5, va="top", family="monospace")
    tp = tax.get_position(); cax = fig.add_axes([tp.x0 + 0.28 * tp.width, tp.y0 + 0.06, tp.width * 0.62, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Procrustes distance (condition means)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    gsb = gs2[1, :].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35); ax = fig.add_subplot(gsb[0])
    ax.scatter(Dr[iu], Do[iu], s=4, c=np.maximum(lc[iu[0]], lc[iu[1]]), cmap="viridis", lw=0); ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.5)
    ax.set_xlabel("KS2.5 Procrustes distance"); ax.set_ylabel("Online Procrustes distance"); ax.set_title("c  Pairwise distances", loc="left")
    ax = fig.add_subplot(gsb[1]); cm = plt.get_cmap("viridis"); nrm = plt.Normalize(lc.min(), lc.max() if lc.max() > lc.min() else lc.min() + 1)
    mk = {m: s_ for m, s_ in zip(sorted(set(mont)), ["o", "^", "s", "D"])}
    for i in range(n):
        col = cm(nrm(lc[i])); ax.plot([Yr[i, 0], Yo[i, 0]], [Yr[i, 1], Yo[i, 1]], "-", color="0.7", lw=0.4)
        ax.plot(Yr[i, 0], Yr[i, 1], mk[mont[i]], color=col, ms=5, mec="k", mew=0.3); ax.plot(Yo[i, 0], Yo[i, 1], mk[mont[i]], color=col, ms=5, mfc="none", mew=0.9)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_aspect("equal", "datalim"); ax.set_title("d  MDS of condition geometry", loc="left")
    sm_ = mpl.cm.ScalarMappable(norm=nrm, cmap=cm); cb2 = fig.colorbar(sm_, ax=ax, fraction=0.04, pad=0.02); cb2.ax.set_title("log10\ncharge (pC)", fontsize=5); cb2.ax.tick_params(labelsize=5)
    ax.legend(handles=[Line2D([], [], marker=mk[m], ls="", color="0.4", mec="k", mew=0.3, label=f"{m}, KS2.5") for m in mk] +
              [Line2D([], [], marker="o", ls="", mfc="none", color="0.4", label="online (aligned)")], frameon=False, fontsize=5, loc="upper left")
    fig.suptitle(TITLE, fontsize=8, x=0.02, ha="left", y=0.99)
    save(fig, "geometry_comparison.png")

def fig_alignment(D):
    TA = pd.read_csv(ctx.R("trial_alignment.csv")); TT = pd.read_csv(ctx.R("trial_alignment_trends.csv"))
    TA["grp"] = np.where(TA.charge >= 5000, "high", np.where(TA.charge >= 1500, "mid", "low")); fam, amps, labs = family(D); c0 = labs[min(amps)]
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw=dict(wspace=0.38, hspace=0.55)); axs = axs.ravel()
    ax = axs[0]
    for nm, col, mk in [("ks25", "0.45", "s"), ("online", "#DD8452", "o")]:
        d = TA[(TA.sort == nm) & (TA.cond == c0)].sort_values("rep")
        ax.plot(d.rep, d.gain_loo, mk, ms=2.2, color=col, alpha=0.7, lw=0, label=f"{'online' if nm == 'online' else 'KS2.5'} (leave-one-out mean)")
        ax.plot(d.rep, d.gain_loo.rolling(9, center=True, min_periods=3).mean(), "-", color=col, lw=1.1)
    d = TA[(TA.sort == "online") & (TA.cond == c0)].sort_values("rep"); ax.plot(d.rep, d.gain_past, "-", color="#4C72B0", lw=0.7, alpha=0.8, label="online (past trials only)")
    ax.axhline(1, color="k", lw=0.4, ls=":"); ax.set_ylim(min(0, d.gain_loo.min() - 0.1), max(2.05, d.gain_loo.max() + 0.3))
    ax.set_xlabel("Trial number within condition"); ax.set_ylabel("Gain onto condition mean")
    ax.set_title(f"a  {c0}, single trials", loc="left"); ax.legend(frameon=False, fontsize=5, loc="upper center", ncol=2, columnspacing=0.8, bbox_to_anchor=(0.56, 1.0), handletextpad=0.3)
    ax = axs[1]
    for grp, ls in [("high", "-"), ("mid", "--")]:
        for nm, col in [("ks25", "0.45"), ("online", "#DD8452")]:
            d = TA[(TA.sort == nm) & (TA.grp == grp)].copy()
            if not len(d): continue
            d["blk"] = (d.rep // 5) * 5 + 2; g = d.groupby("blk").gain_loo.agg(["mean", "sem"])
            ax.plot(g.index, g["mean"], ls, color=col, lw=1.1, label=f"{'online' if nm == 'online' else 'KS2.5'}, {grp} charge")
            ax.fill_between(g.index, g["mean"] - g["sem"], g["mean"] + g["sem"], color=col, alpha=0.2, lw=0)
    ax.axhline(1, color="k", lw=0.4, ls=":"); ax.set_ylim(-0.05, 1.3); ax.set_xlabel("Trial number within condition (5-trial bins)"); ax.set_ylabel("Gain (mean ± SEM)")
    ax.set_title("b  Gain across repetitions", loc="left"); ax.legend(frameon=False, fontsize=5, loc="upper center", ncol=2, columnspacing=0.8, bbox_to_anchor=(0.5, 1.02))
    ax = axs[2]
    for nm, col in [("ks25", "0.45"), ("online", "#DD8452")]:
        d = TA[(TA.sort == nm) & (TA.grp == "high")].copy(); d["blk"] = (d.rep // 5) * 5 + 2
        g = d.groupby("blk")[["projscore_loo", "projscore_null_othercond"]].mean()
        ax.plot(g.index, g.projscore_loo, "-", color=col, lw=1.1, label=f"{'online' if nm == 'online' else 'KS2.5'}: own condition (LOO)")
        ax.plot(g.index, g.projscore_null_othercond, ":", color=col, lw=1, label=f"{'online' if nm == 'online' else 'KS2.5'}: other conditions (null)")
    d = TA[(TA.sort == "online") & (TA.grp == "high")].copy(); d["blk"] = (d.rep // 5) * 5 + 2; ax.plot(d.groupby("blk").projscore_past.mean(), "-", color="#4C72B0", lw=0.8, label="online: past trials only")
    ax.set_ylim(0.05, 0.45); ax.set_xlabel("Trial number within condition (5-trial bins)"); ax.set_ylabel("Variance fraction in\ncondition subspace (k=3)")
    ax.set_title("c  Subspace alignment, high charge", loc="left"); ax.legend(frameon=False, fontsize=4.5, loc="upper left", ncol=2, columnspacing=0.5, handlelength=1.5)
    ax = axs[3]; tg = TT[TT.metric == "gain_loo"].pivot_table(index=["cond", "charge"], columns="sort", values=["slope_per10", "p"]).reset_index()
    sig = (tg[("p", "online")] < 0.05) | (tg[("p", "ks25")] < 0.05)
    scx = ax.scatter(tg[("slope_per10", "ks25")], tg[("slope_per10", "online")], c=np.log10(tg.charge), cmap="viridis", s=14, lw=0)
    ax.scatter(tg[("slope_per10", "ks25")][sig], tg[("slope_per10", "online")][sig], s=34, facecolors="none", edgecolors="k", lw=0.6)
    lim = np.array([-1, 1]) * max(0.05, np.nanmax(np.abs(tg[["slope_per10"]].values)) * 1.15); ax.plot(lim, lim, "k--", lw=0.5)
    ax.axhline(0, color="0.7", lw=0.4); ax.axvline(0, color="0.7", lw=0.4); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("KS2.5 gain slope (per 10 trials)"); ax.set_ylabel("Online gain slope (per 10 trials)"); ax.set_title("d  Per-condition trends", loc="left")
    cb = fig.colorbar(scx, ax=ax, fraction=0.045, pad=0.03); cb.ax.set_title("log10\ncharge (pC)", fontsize=5); cb.ax.tick_params(labelsize=5)
    r_sl = np.corrcoef(tg[("slope_per10", "ks25")], tg[("slope_per10", "online")])[0, 1]
    ax.text(0.03, 0.97, f"r = {r_sl:.2f}\nringed: p<0.05 in either", transform=ax.transAxes, fontsize=5, va="top")
    fig.suptitle(TITLE, fontsize=8, x=0.02, ha="left", y=1.0)
    save(fig, "trial_alignment.png")

def fig_demo(D):
    if not os.path.exists(ctx.R("demo_live_summary.json")): print("[fig] no demo results"); return
    S_ = json.load(open(ctx.R("demo_live_summary.json"))); L = pd.read_csv(ctx.R("demo_live_log.csv")); BL = pd.read_csv(ctx.R("demo_live_buffers.csv"))
    IB = pd.read_csv(ctx.R("iti_bench.csv")); fs, off = D["fs"], D["off"]; st0 = int(BL.start.iloc[0]); BL["t"] = (BL.start + 1024 - st0) / fs
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw=dict(hspace=0.62, wspace=0.38)); axs = axs.ravel()
    ax = axs[0]; ks = L.trial.values[30:33]; tsec = lambda s: (np.asarray(s) - st0) / fs; t_lo = tsec(off[ks[0]]) - 0.4; t_hi = tsec(off[ks[-1]]) + 2.25
    m = (BL.t >= t_lo) & (BL.t <= t_hi); ax.plot(BL.t[m], BL.behind_ms[m], "-", color="0.35", lw=0.6)
    for k in ks:
        a = tsec(off[k]); ax.axvline(a, color="#C44E52", lw=0.8); ax.axvspan(a + 0.001, a + 1.901, color="#DD8452", alpha=0.12, lw=0)
        lat = L.set_index("trial").latency_ms[k]; ax.plot(a + 1.901 + lat / 1e3, lat, "v", color="#4C72B0", ms=4)
    ax.set_xlim(t_lo, t_hi); ax.set_ylim(0, max(110, BL.behind_ms[m].max() * 1.5)); ax.set_xlabel("Stream time (s)"); ax.set_ylabel("ms after data written")
    ax.set_title("a  3 consecutive trials, real-time replay", loc="left")
    ax.legend(handles=[Line2D([], [], color="#C44E52", lw=0.8, label="pulse"), mpl.patches.Patch(color="#DD8452", alpha=0.25, label="decoded window (1.9 s)"),
                       Line2D([], [], color="0.35", lw=0.6, label="buffer lag"), Line2D([], [], color="#4C72B0", marker="v", ls="", ms=4, label="geometry ready")],
              frameon=False, fontsize=5, loc="upper left", ncol=2, columnspacing=0.8)
    ax = axs[1]; ax.plot(np.arange(len(L)), L.latency_ms, "o", ms=2, color="#4C72B0"); p99 = L.latency_ms.quantile(.99)
    ax.axhline(p99, color="k", lw=0.5, ls="--"); ax.text(len(L) - 1, p99 * 1.03, f"p99 {p99:.0f} ms", fontsize=5, ha="right", va="bottom")
    nb = (L.run == L.run.iloc[0]).sum(); ax.axvline(nb - 0.5, color="0.7", lw=0.5)
    for r_, xm in [(L.run.iloc[0], nb / 2), (L.run.iloc[-1], nb + (len(L) - nb) / 2)]: ax.text(xm, 3, f"run {r_} ({L[L.run == r_].cond.iloc[0].split('/')[0]} µA)", fontsize=5, ha="center")
    ax.set_ylim(0, max(80, L.latency_ms.max() * 1.25)); ax.set_xlabel("Trial (demo order)"); ax.set_ylabel("Window end → geometry (ms)"); ax.set_title(f"b  Per-trial latency (n={len(L)})", loc="left")
    ax = axs[2]; ax.hist(BL.proc_ms, np.linspace(0, max(90, BL.proc_ms.quantile(.999) * 1.1), 61), color="0.5", lw=0); bud = 1024 / fs * 1e3; ax.axvline(bud, color="#C44E52", lw=0.8)
    ax.set_yscale("log"); ax.text(bud + 1, ax.get_ylim()[1] * 0.9, f"real-time budget\n{bud:.1f} ms / buffer", fontsize=5, color="#C44E52", va="top")
    ax.set_xlabel("Processing time per 1024-sample buffer (ms)"); ax.set_ylabel("Buffers"); ax.set_title(f"c  Push throughput (RTF {S_['rtf']:.2f}, max lag {S_['behind_ms_max']:.0f} ms)", loc="left")
    ax = axs[3]
    for c, col in zip(L.cond.unique(), ["#2166AC", "#B2182B"]):
        d = L[L.cond == c]; ax.plot(d.rep, d.gain_past, "o", ms=2, color=col, alpha=0.6); ax.plot(d.rep, d.gain_past.rolling(9, center=True, min_periods=3).mean(), "-", color=col, lw=1, label=c)
    ax.axhline(1, color="k", lw=0.4, ls=":"); ax.set_xlabel("Trial number within condition"); ax.set_ylabel("Gain onto past-trial mean"); ax.set_title("d  Live single-trial geometry", loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    fig.suptitle(f"{TITLE}: real-time replay (Apple M1, 1024-sample buffers)", fontsize=8, x=0.02, ha="left", y=1.0)
    save(fig, "demo_live_timeline.png")
    # ITI budget
    lat_push = S_["latency_ms_p99"] / 1e3; geo = S_["geometry_ms_p99"] / 1e3; rows = []
    for wl in [0.5, 1.0, 1.9]:
        pull = IB[IB.window_s == wl].decode_ms.quantile(.99) / 1e3; phys = 1.0 if wl < 1.9 else 0
        for mode, lat in [("push (streaming)", lat_push), ("pull (decode after window)", pull + geo)]:
            closed = max(0.0012 + wl + lat, phys)
            rows.append(dict(window_s=wl, baseline="same chunk 1.0–1.9 s" if wl >= 1.9 else "pre-pulse 0.5 s", mode=mode, latency_after_window_p99_ms=round(lat * 1e3, 1),
                             min_iti_closed_loop_s=round(closed, 3), min_iti_open_loop_s=round(max(0.0012 + wl, lat), 3), fits_2s=bool(closed <= 2.0)))
    ITI = pd.DataFrame(rows); ITI.to_csv(ctx.R("iti_budget.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.2, 2.6)); labs = []
    for y, r in ITI.iterrows():
        w = r.window_s + 0.0012; lat = r.latency_after_window_p99_ms / 1e3; fl = r.min_iti_closed_loop_s
        ax.barh(y, w, color="#DD8452", height=0.6); ax.barh(y, lat, left=w, color="#4C72B0", height=0.6)
        if fl > w + lat + 1e-6: ax.barh(y, fl - w - lat, left=w + lat, color="0.85", height=0.6, hatch="////", edgecolor="0.6", lw=0)
        ax.text(max(fl, w + lat) + 0.03, y, f"{fl:.2f} s", va="center", fontsize=5.5); labs.append(f"{r.window_s:.1f} s window, {'push' if r['mode'].startswith('push') else 'pull'}")
    ax.axvline(2.0, color="#C44E52", lw=0.8, ls="--"); ax.text(2.02, -0.55, "current ITI 2.0 s", color="#C44E52", fontsize=5.5, ha="left", va="center")
    ax.set_yticks(range(len(ITI))); ax.set_yticklabels(labs, fontsize=6); ax.set_ylim(len(ITI) - 0.5, -0.8); ax.set_xlim(0, 2.75)
    ax.set_xlabel("Minimum closed-loop inter-trial interval (s)")
    ax.legend(handles=[mpl.patches.Patch(color="#DD8452", label="pulse + 1 ms + analysis window"), mpl.patches.Patch(color="#4C72B0", label="wait: last window sample written → geometry ready\n(99th percentile over trials)"),
                       mpl.patches.Patch(facecolor="0.85", hatch="////", edgecolor="0.6", label="0.5 s pre-pulse baseline after ~0.5 s response")],
              frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"{TITLE}: inter-trial interval budget", loc="left", fontsize=7)
    ax.text(0, -0.30, "Push wait was measured with the 1.9 s window only and is applied to the 0.5 s and 1.0 s rows. In the jlh32 3-probe test, measured p99 waits were 101 / 105 / 96 ms for 0.5 / 1.0 / 1.9 s.",
            transform=ax.transAxes, fontsize=4.8, color="0.4", ha="left", va="top")
    save(fig, "iti_budget.png")

def main():
    D = load(); RS = by_run(D)
    for f, a in [(fig_gate, ()), (fig_validation, (D, RS)), (fig_near_pulse, (D,)), (fig_geometry, (D,)), (fig_comparison, ()), (fig_alignment, (D,)), (fig_demo, (D,))]:
        try:
            f(*a)
        except Exception as e:
            import traceback; print(f"[fig] {f.__name__} FAILED: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()
