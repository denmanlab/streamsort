"""Recording-level (pooled-probe) figures.  REC=... python scripts/make_population_figures.py"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx, make_figures as mf
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.lines import Line2D

TITLE = f"{ctx.REC}, all probes pooled"
PCOL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}

def fig_population(D):
    P = np.load(ctx.RP("geometry_per_trial_sup.npz"), allow_pickle=True); Zo, Zr, cond = P["Z_online"], P["Z_ks25"], P["cond"]
    SW = pd.read_csv(ctx.RP("subset_sweep.csv")); S = json.load(open(ctx.RP("population_summary.json"))); dep = S["deployment"]
    summ = json.load(open(ctx.RP("geometry_comparison_summary.json"))); pou = np.load(ctx.RP("probe_of_unit.npy"), allow_pickle=True)
    t_ax = 2.0 + 10 * (np.arange(Zo.shape[1]) + 0.5); fam, amps, labs = mf.family(D); amax = max(abs(a) for a in amps)
    So = {c: Zo[cond == c].mean(0) for c in np.unique(cond)}; Sr = {c: Zr[cond == c].mean(0) for c in np.unique(cond)}
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), gridspec_kw=dict(wspace=0.55, hspace=0.72)); axs = axs.ravel()
    ax = axs[0]
    for a in amps:
        c = labs[a]; col = mf.ampcolor(a, amax); ax.plot(So[c][:, 0], So[c][:, 1], "-", color=col, lw=1.1); ax.plot(Sr[c][:, 0], Sr[c][:, 1], "--", color=col, lw=0.7)
        ax.plot(So[c][0, 0], So[c][0, 1], "o", color=col, ms=2.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"a  Pooled condition means\n    ({S['n_units_matched']} units)", loc="left")
    ax.legend(handles=[Line2D([], [], color="0.3", lw=1.1, label="online"), Line2D([], [], color="0.3", lw=0.7, ls="--", label="KS2.5")], frameon=False, fontsize=5.5, loc="best")
    ax = axs[1]
    for a in amps: ax.plot(t_ax, So[labs[a]][:, 0], color=mf.ampcolor(a, amax), lw=1, label=f"{a:+d} µA")
    ax.set_xlim(0, 505); ax.set_xticks([0, 100, 200, 300, 400, 500])
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.42 * (y1 - y0), y1)
    ax.legend(frameon=False, fontsize=4.6, ncol=4, loc="lower left", handlelength=0.9, columnspacing=0.5, handletextpad=0.3)
    ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("PC1 (online)"); ax.set_title("b  PC1 time course", loc="left")
    ax = axs[2]; x = np.arange(len(SW)); w = 0.38
    ax.bar(x - w / 2, SW.acc_online, w, color="#DD8452", label="online"); ax.bar(x + w / 2, SW.acc_ks25, w, color="0.55", label="KS2.5")
    ax.axhline(summ["chance"], color="k", lw=0.5, ls=":"); ax.set_xticks(x); ax.set_xticklabels([f"{p}\n{n}" for p, n in zip(SW.probes, SW.n_units)], fontsize=5)
    ax.set_ylabel(f"{summ['n_conditions']}-way single-trial accuracy"); ax.set_xlabel("probes (units)", labelpad=1); ax.set_title("c  Adding probes", loc="left")
    ax.legend(frameon=False, fontsize=5, loc="upper left"); ax.set_ylim(0, max(SW[["acc_online", "acc_ks25"]].values.max() * 1.3, 0.1))
    ax = axs[3]
    ax.plot(x, SW.procrustes_same_trial, "o-", color="#DD8452", ms=3, lw=1, label="online vs KS2.5, same trial")
    ax.plot(x, SW.procrustes_trial_vs_mean, "s-", color="0.45", ms=3, lw=1, label="KS2.5 trial vs condition mean")
    ax.set_xticks(x); ax.set_xticklabels(SW.probes, fontsize=5.5); ax.set_ylim(0, 1.05); ax.set_ylabel("Procrustes distance (median)")
    ax.set_xlabel("probes"); ax.set_title("d  Decoder distortion", loc="left"); ax.legend(frameon=False, fontsize=5, loc="lower left")
    ax = axs[4]; Bv = P["basis"]; E = (Bv ** 2); probes = list(dict.fromkeys(pou.tolist()))
    for j in range(Bv.shape[1]):
        bottom = 0
        for p in probes:
            v = E[pou == p, j].sum() / E[:, j].sum(); ax.bar(j, v, bottom=bottom, color=PCOL.get(p, "0.5"), width=0.6); bottom += v
    fr = [np.mean(pou == p) for p in probes]; bottom = 0
    for p, f in zip(probes, fr): ax.bar(3.2, f, bottom=bottom, color=PCOL.get(p, "0.5"), width=0.6, alpha=0.6); bottom += f
    ax.set_xticks([0, 1, 2, 3.2]); ax.set_xticklabels(["PC1", "PC2", "PC3", "unit\nshare"], fontsize=5.5); ax.set_ylim(0, 1); ax.set_ylabel("Fraction of loading energy")
    ax.set_title("e  Contribution by probe", loc="left")
    ax.set_xlim(-0.6, 4.6); ax.legend(handles=[mpl.patches.Patch(color=PCOL.get(p, "0.5"), label=p) for p in probes], frameon=False, fontsize=5, loc="center right", bbox_to_anchor=(1.02, 0.5), handlelength=0.8)
    ax = axs[5]; vals = [summ["acc_baselined_online"], summ["acc_baselined_ks25"], dep["acc_online_all"], dep["acc_ks25_good"]]
    labs_ = [f"matched\nonline\n({S['n_units_matched']})", f"matched\nKS2.5\n({S['n_units_matched']})", f"all usable\nonline\n({dep['n_online_usable']})", f"all good\nKS2.5\n({dep['n_ks25_good']})"]
    ax.bar(range(4), vals, color=["#DD8452", "0.55", "#C44E52", "0.3"], width=0.6); ax.axhline(summ["chance"], color="k", lw=0.5, ls=":")
    for i, v in enumerate(vals): ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs_, fontsize=5); ax.set_ylabel("Single-trial accuracy"); ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("f  Deployment population", loc="left")
    ax.text(0.98, 0.97, f"condition geometry ρ = {dep['rho_condition_geometry']:.2f}\nsame-trial Procrustes {dep['procrustes_same_trial_median']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5)
    fig.suptitle(f"{TITLE} — amplitude series in a–b: {fam}", fontsize=8, x=0.02, ha="left", y=1.01)
    mf.save(fig, "geometry_population.png")

def fig_pcsweep():
    if not os.path.exists(ctx.RP("pc_count_sweep.csv")): return
    K = pd.read_csv(ctx.RP("pc_count_sweep.csv")); summ = json.load(open(ctx.RP("geometry_comparison_summary.json")))
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.5), gridspec_kw=dict(wspace=0.35), sharey=True)
    for ax, norm in zip(axs, ["raw", "zscore"]):
        for sub, d in K[K.norm == norm].groupby("probes", sort=False):
            col = PCOL.get(sub, "k"); lw = 1.8 if len(sub) > 1 else 1.0
            ax.plot(d.k, d.acc_online, "o-", color=col, lw=lw, ms=3, label=f"{sub} ({d.n_units.iloc[0]})")
            ax.plot(d.k, d.acc_ks25, "--", color=col, lw=0.7)
        ax.axhline(summ["chance"], color="k", lw=0.5, ls=":"); ax.set_xlabel("Number of PCs"); ax.set_xticks(sorted(K.k.unique()))
        ax.set_title(f"{'a' if norm == 'raw' else 'b'}  {'Raw rates' if norm == 'raw' else 'Per-unit z-scored'}", loc="left")
    axs[0].set_ylabel(f"{summ['n_conditions']}-way single-trial accuracy"); axs[1].legend(frameon=False, fontsize=5, loc="lower right", title="probes (units)", title_fontsize=5)
    axs[0].text(0.03, 0.97, "solid: online   dashed: KS2.5", transform=axs[0].transAxes, fontsize=5, va="top")
    fig.suptitle(f"{TITLE}: condition information vs dimensionality", fontsize=8, x=0.02, ha="left", y=1.04)
    mf.save(fig, "pc_count_sweep.png")

def fig_popdemo():
    if not os.path.exists(ctx.RP("demo_population_summary.json")): print("[fig] no population demo"); return
    S = json.load(open(ctx.RP("demo_population_summary.json"))); L = pd.read_csv(ctx.RP("demo_population_log.csv")); BL = pd.read_csv(ctx.RP("demo_population_buffers.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw=dict(hspace=0.62, wspace=0.38)); axs = axs.ravel(); budget = 1024 / 30000 * 1e3
    ax = axs[0]; ax.plot(np.arange(len(L)), L.latency_ms, "o", ms=2, color="#4C72B0"); p99 = L.latency_ms.quantile(.99); ax.axhline(p99, color="k", lw=0.5, ls="--")
    ax.text(len(L) - 1, p99 * 1.03, f"p99 {p99:.0f} ms", fontsize=5, ha="right", va="bottom"); ax.set_ylim(0, max(80, L.latency_ms.max() * 1.25))
    ax.set_xlabel("Trial (demo order)"); ax.set_ylabel("Last window end → pooled geometry (ms)"); ax.set_title(f"a  Pooled latency ({len(S['probes'])} probes, {S['n_units_pooled']} units)", loc="left")
    ax = axs[1]
    for p in S["probes"]:
        d = BL[BL.probe == p]; ax.hist(d.proc_ms, np.linspace(0, max(90, BL.proc_ms.quantile(.999) * 1.1), 61), histtype="step", lw=1, color=PCOL.get(p, "0.5"), label=f"probe {p}")
    ax.axvline(budget, color="#C44E52", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("Processing time per 1024-sample buffer (ms)"); ax.set_ylabel("Buffers")
    ax.set_title("b  Per-probe buffer cost", loc="left"); ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax = axs[2]
    for p in S["probes"]:
        d = BL[BL.probe == p].reset_index(drop=True); t = np.arange(len(d)) * budget / 1e3; ax.plot(t, d.behind_ms, lw=0.4, color=PCOL.get(p, "0.5"), label=f"probe {p}")
    ax.set_xlabel("Stream time (s)"); ax.set_ylabel("Lag behind write head (ms)"); ax.set_title("c  Keeping up with the stream", loc="left"); ax.legend(frameon=False, fontsize=5, loc="upper left")
    ax = axs[3]; ks = list(S["rtf_per_probe"]); vals = [S["rtf_per_probe"][k] for k in ks] + [S["rtf_total"]]
    ax.bar(range(len(vals)), vals, color=[PCOL.get(k, "0.5") for k in ks] + ["0.3"], width=0.6); ax.axhline(1.0, color="#C44E52", lw=0.8, ls="--")
    for i, v in enumerate(vals): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([f"probe {k}" for k in ks] + ["all"], fontsize=5.5); ax.set_ylabel("Real-time factor (CPU s / stream s)")
    ax.set_title("d  Compute budget (1.0 = saturated)", loc="left"); ax.set_ylim(0, max(1.2, max(vals) * 1.2))
    fig.suptitle(f"{TITLE}: concurrent real-time replay of runs {S['runs']} (one process, Apple M1)", fontsize=8, x=0.02, ha="left", y=1.0)
    mf.save(fig, "demo_population.png")

WAIT_LABEL = "wait: last window sample written → geometry ready\n(99th percentile over trials; white tick = median)"

def iti_rows(push, pull, cycle_s=2.0):
    """Closed-loop ITI budget rows. Wait = per-trial time from the last sample of the analysis window being written
    to the pooled geometry being ready; p99 = 99th percentile over trials."""
    rows = []
    for w in sorted(push.window_s.unique()):
        for mode, d in [("push", push[push.window_s == w].latency_ms), ("pull", pull[pull.window_s == w].total_ms)]:
            p99, med = d.quantile(.99) / 1e3, d.median() / 1e3; floor = 1.0 if w < 1.9 else 0.0
            closed = max(0.0012 + w + p99, floor)
            rows.append(dict(window_s=w, mode=mode, n_trials=int(len(d)), wait_median_ms=round(med * 1e3, 1), wait_p99_ms=round(p99 * 1e3, 1),
                             wait_max_ms=round(d.max(), 1), min_iti_closed_loop_s=round(closed, 3), fits_current_iti=bool(closed <= cycle_s)))
    return pd.DataFrame(rows)

def fig_itipop():
    if not (os.path.exists(ctx.RP("iti_population_push.csv")) and os.path.exists(ctx.RP("iti_population_pull.csv"))): print("[fig] no population ITI bench"); return
    S = json.load(open(ctx.RP("demo_population_summary.json"))); PU = pd.read_csv(ctx.RP("iti_population_push.csv")); PL = pd.read_csv(ctx.RP("iti_population_pull.csv"))
    BL = pd.read_csv(ctx.RP("demo_population_buffers.csv")); probes = S["probes"]; ITI = iti_rows(PU, PL); ITI.to_csv(ctx.RP("iti_population_budget.csv"), index=False)
    fs = 29970.0757 if ctx.REC == "jlh32" else 30000.0; budget = 1024 / fs * 1e3
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(hspace=0.75, wspace=0.42, width_ratios=[1.25, 1])); axs = axs.ravel()
    # a: budget bars
    ax = axs[0]; labs = []
    for y, r in ITI.iterrows():
        w = r.window_s + 0.0012; lat = r.wait_p99_ms / 1e3; fl = r.min_iti_closed_loop_s
        ax.barh(y, w, color="#DD8452", height=0.6); ax.barh(y, lat, left=w, color="#4C72B0", height=0.6)
        ax.plot([w + r.wait_median_ms / 1e3] * 2, [y - 0.3, y + 0.3], color="w", lw=0.8)
        if fl > w + lat + 1e-6: ax.barh(y, fl - w - lat, left=w + lat, color="0.85", height=0.6, hatch="////", edgecolor="0.6", lw=0)
        ax.text(max(fl, w + lat) + 0.03, y, f"{fl:.2f} s  (+{r.wait_p99_ms:.0f} ms)", va="center", fontsize=5.2)
        labs.append(f"{r.window_s:.1f} s window, {r['mode']}")
    ax.axvline(2.0, color="#C44E52", lw=0.8, ls="--"); ax.text(2.02, -0.75, "current ITI 2.0 s", color="#C44E52", fontsize=5.2, ha="left", va="center")
    ax.set_yticks(range(len(ITI))); ax.set_yticklabels(labs, fontsize=5.5); ax.set_ylim(len(ITI) - 0.5, -1.0); ax.set_xlim(0, 3.6)
    ax.set_xlabel("Minimum closed-loop inter-trial interval (s)"); ax.set_title(f"a  ITI budget, {len(probes)} probes live", loc="left")
    ax.legend(handles=[mpl.patches.Patch(color="#DD8452", label="pulse + 1 ms + analysis window"), mpl.patches.Patch(color="#4C72B0", label=WAIT_LABEL),
                       mpl.patches.Patch(facecolor="0.85", hatch="////", edgecolor="0.6", label="floor: 0.5 s quiet pre-pulse baseline\nafter a ~0.5 s response")],
              frameon=False, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    # b: push wait by window, with single-probe reference
    ax = axs[1]; ws = sorted(PU.window_s.unique()); rng = np.random.default_rng(0)
    for i, w in enumerate(ws):
        d = PU[PU.window_s == w].latency_ms; ax.plot(i + rng.uniform(-0.18, 0.18, len(d)), d, "o", ms=1.6, color="#4C72B0", alpha=0.6)
        q = d.quantile(.99); ax.plot([i - 0.28, i + 0.28], [q] * 2, color="k", lw=0.9); ax.text(i + 0.3, q, f"p99 {q:.0f}", fontsize=5, va="center")
    ref = []
    for p in probes:
        f_ = ctx.R_probe(ctx.REC, p, "demo_live_summary.json")
        if os.path.exists(f_): ref.append(json.load(open(f_))["latency_ms_p99"])
    if ref:
        ax.axhspan(min(ref), max(ref), color="0.85", lw=0, zorder=0); ax.text(0.02, 0.97, f"grey band: each probe alone, 1.9 s window,\np99 {min(ref):.0f}–{max(ref):.0f} ms", transform=ax.transAxes, fontsize=4.8, ha="left", va="top", color="0.4")
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f"{w:.1f} s" for w in ws]); ax.set_xlim(-0.5, len(ws) - 0.2); ax.set_ylim(0, max(170, PU.latency_ms.max() * 1.3))
    ax.set_xlabel("Analysis window"); ax.set_ylabel("Push wait per trial (ms)"); ax.set_title("b  Push: wait after window end", loc="left")
    # c: CPU cost across the trial cycle (one thread services all probes)
    ax = axs[2]; B = BL[(BL.since_pulse_ms > 0) & (BL.since_pulse_ms < 2000)].copy(); edges = np.arange(0, 2000 + budget, budget); B["bin"] = np.digitize(B.since_pulse_ms, edges) - 1
    ctr = (edges[:-1] + edges[1:]) / 2; tot = np.zeros(len(ctr))
    for p in probes:
        m = B[B.probe == p].groupby("bin").proc_ms.mean().reindex(range(len(ctr))).to_numpy(); ax.plot(ctr, m, lw=0.8, color=PCOL.get(p, "0.5"), label=f"probe {p}"); tot += np.nan_to_num(m)
    ax.plot(ctr, tot, lw=1.2, color="k", label="sum (one thread)"); ax.axhline(budget, color="#C44E52", lw=0.8, ls="--")
    ax.text(1900, budget * 1.04, f"real time: {budget:.1f} ms of data per buffer", fontsize=4.8, color="#C44E52", ha="right", va="bottom")
    ax.set_xlim(0, 2000); ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_ylim(0, max(tot.max() * 1.15, budget * 1.5)); ax.set_xlabel("Buffer end, time after pulse onset (ms)"); ax.set_ylabel("CPU time per buffer (ms, mean)")
    ax.set_title("c  Where the CPU time goes", loc="left"); ax.legend(frameon=False, fontsize=4.8, loc="upper right")
    # d: pull decomposition
    ax = axs[3]; wsl = sorted(PL.window_s.unique()); x = np.arange(len(wsl))
    for i, w in enumerate(wsl):
        d = PL[PL.window_s == w]; bot = 0
        for p in probes:
            v = d[f"decode_{p}_ms"].median(); ax.bar(i, v, bottom=bot, color=PCOL.get(p, "0.5"), width=0.55); bot += v
        q = d.total_ms.quantile(.99); ax.plot([i - 0.3, i + 0.3], [q] * 2, color="k", lw=0.9); ax.text(i + 0.32, q, f"p99 {q:.0f}", fontsize=5, va="center")
    ax.legend(handles=[mpl.patches.Patch(color=PCOL.get(p, "0.5"), label=f"probe {p} decode (median)") for p in probes] + [Line2D([], [], color="k", lw=0.9, label="total wait, p99")],
              frameon=False, fontsize=4.8, loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels([f"{w:.1f} s" for w in wsl]); ax.set_xlim(-0.5, len(wsl) - 0.1); ax.set_xlabel("Analysis window"); ax.set_ylabel("Pull wait per trial (ms)")
    ax.set_title("d  Pull: probes decoded in turn after window end", loc="left"); ax.set_ylim(0, PL.total_ms.quantile(.99) * 1.35 if len(PL) else 1)
    fig.suptitle(f"{TITLE}: inter-trial interval budget for live decoding of {len(probes)} probes (one process, Apple M1; runs {S['runs']})", fontsize=7.5, x=0.02, ha="left", y=1.0)
    mf.save(fig, "iti_population.png")

LOSS_MS = float(os.environ.get("LOSS_MS", "5"))      # metric -> loss/distance (assumed)
CMD_MS = float(os.environ.get("CMD_MS", "50"))       # serial command to change stimulation parameters
ITI_MS = float(os.environ.get("ITI_MS", "2000"))     # current pulse-to-pulse interval

def closed_loop_table(PU, PL):
    """Per-trial closed loop: pulse+1 ms+window, then decode+geometry wait (measured), loss (LOSS_MS), serial command (CMD_MS);
    slack = ITI_MS - total. The 1.9 s push wait is corrected for validation work done inside the timed span in older runs."""
    g0 = PU[PU.window_s < 1.9].geometry_ms.median() if (PU.window_s < 1.9).any() else 0.7
    PU = PU.assign(wait=np.where(PU.window_s >= 1.9, PU.latency_ms - PU.geometry_ms + g0, PU.latency_ms)); PL = PL.assign(wait=PL.total_ms)
    rows, per = [], []
    for mode, df in [("push", PU), ("pull", PL)]:
        for w, g in df.groupby("window_s"):
            tot = 1.2 + w * 1e3 + g.wait + LOSS_MS + CMD_MS; sl = ITI_MS - tot
            per.append(pd.DataFrame(dict(mode=mode, window_s=w, trial=g.trial.values, wait_ms=g.wait.values, slack_ms=sl.values)))
            rows.append(dict(window_s=w, mode=mode, n_trials=len(g), wait_p99_ms=round(g.wait.quantile(.99), 1), loss_ms=LOSS_MS, command_ms=CMD_MS,
                             ready_p99_ms=round(tot.quantile(.99), 1), slack_median_ms=round(sl.median(), 1), slack_worst1pct_ms=round(sl.quantile(.01), 1),
                             frac_trials_in_time=round(float((sl >= 0).mean()), 3), min_iti_p99_s=round(max(tot.quantile(.99), 1000.0 if w < 1.9 else 0.0) / 1e3, 3),
                             max_window_at_iti_p99_s=round((ITI_MS - 1.2 - g.wait.quantile(.99) - LOSS_MS - CMD_MS) / 1e3, 3)))
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)

def fig_closed_loop():
    if not (os.path.exists(ctx.RP("iti_population_push.csv")) and os.path.exists(ctx.RP("iti_population_pull.csv"))): return
    S = json.load(open(ctx.RP("demo_population_summary.json"))); PU = pd.read_csv(ctx.RP("iti_population_push.csv")); PL = pd.read_csv(ctx.RP("iti_population_pull.csv"))
    CL, PER = closed_loop_table(PU, PL); CL.to_csv(ctx.RP("closed_loop_budget.csv"), index=False); PER.to_csv(ctx.RP("closed_loop_per_trial.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.5, width_ratios=[1.35, 1])); labs = []
    ax = axs[0]; C = dict(wait="#4C72B0", loss="#8172B3", cmd="#C44E52")
    for y, r in CL.iterrows():
        x = 0
        for key, wdt in [("wait", r.wait_p99_ms), ("loss", r.loss_ms), ("cmd", r.command_ms)]:
            ax.barh(y, wdt, left=x, color=C[key], height=0.6, lw=0); x += wdt
        nxt = ITI_MS - 1.2 - r.window_s * 1e3; sl = nxt - x
        ax.plot([nxt, nxt], [y - 0.42, y + 0.42], color="k", lw=1.2)
        ax.text(max(x, nxt) + 25, y, (f"{sl:.0f} ms spare" if sl >= 0 else f"{-sl:.0f} ms late") + f" ({r.frac_trials_in_time:.0%} in time)", va="center", fontsize=4.9,
                color="k" if sl >= 0 else "#C44E52")
        labs.append(f"{r.window_s:.1f} s window, {r['mode']}")
    ax.set_yticks(range(len(CL))); ax.set_yticklabels(labs, fontsize=5.5); ax.set_ylim(len(CL) - 0.5, -0.6); ax.set_xlim(0, 2300)
    ax.set_xlabel("Time after the analysis window ends (ms)"); ax.set_title(f"a  After the window: p99 per step (ITI {ITI_MS / 1e3:.1f} s)", loc="left")
    ax.legend(handles=[mpl.patches.Patch(color=C["wait"], label="decode + geometry wait (p99, measured)"), mpl.patches.Patch(color=C["loss"], label=f"loss / distance ({LOSS_MS:.0f} ms, assumed)"),
                       mpl.patches.Patch(color=C["cmd"], label=f"serial stim command ({CMD_MS:.0f} ms, assumed)"), Line2D([], [], color="k", lw=1.2, label="next pulse")],
              frameon=False, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axs[1]; w_ = max(PU.window_s); avail = ITI_MS - 1.2 - w_ * 1e3; d = PER[(PER["mode"] == "push") & (PER.window_s == w_)]
    need = d.wait_ms + LOSS_MS + CMD_MS; bins = np.arange(np.floor(need.min() / 5) * 5, need.max() + 10, 5)
    ax.hist(need, bins, color="#4C72B0", lw=0); ax.axvline(avail, color="k", lw=1.0); ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.text(avail - 2, ax.get_ylim()[1] * 0.66, f"next pulse:\n{avail:.0f} ms after\nwindow end", fontsize=5, va="top", ha="right")
    r = CL[(CL["mode"] == "push") & (CL.window_s == w_)].iloc[0]
    ax.text(0.98, 0.98, f"{r.frac_trials_in_time:.0%} of trials in time\nITI needed (p99): {r.min_iti_p99_s:.3f} s\nor window ≤ {r.max_window_at_iti_p99_s:.3f} s at {ITI_MS / 1e3:.1f} s ITI",
            transform=ax.transAxes, fontsize=5, va="top", ha="right")
    ax.set_xlabel("Window end → stimulator updated (ms)"); ax.set_ylabel("Trials"); ax.set_title(f"b  {w_:.1f} s window, push: each trial", loc="left")
    fig.suptitle(f"{TITLE}: closed-loop budget, {len(S['probes'])} probes live (decode + geometry measured; loss and command assumed)", fontsize=7.5, x=0.02, ha="left", y=1.03)
    mf.save(fig, "closed_loop_budget.png")

def fig_window(D):
    if not os.path.exists(ctx.RP("window_energy.csv")): print("[fig] no window analysis"); return
    EN = pd.read_csv(ctx.RP("window_energy.csv")); TC = pd.read_csv(ctx.RP("window_timecourse.csv")); PZ = np.load(ctx.RP("window_psth.npz"), allow_pickle=True)
    tr = D["tr"]; lab = np.asarray(ctx.cond_labels(tr)); q = (tr.amplitude.abs() * tr.pulse_duration).to_numpy()
    qc = {c: float(np.log10(q[lab == c][0])) for c in np.unique(lab)}; cmap = plt.cm.viridis; nq = mpl.colors.Normalize(min(qc.values()), max(qc.values()))
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.9), gridspec_kw=dict(hspace=0.65, wspace=0.55, right=0.9)); axs = axs.ravel()
    U = EN[(EN["sort"] == "online") & (EN.space == "units")]
    def shade(ax):
        ax.axvspan(2, 502, color="#DD8452", alpha=0.10, lw=0); ax.axvspan(1000, 1900, color="0.6", alpha=0.15, lw=0); ax.axvline(1000, color="k", lw=0.5, ls=":")
    for i, ax in enumerate(axs[:2]):
        shade(ax)
        for c, g in U.groupby("cond"): ax.plot(g.t_ms, g.signal, lw=0.6, color=cmap(nq(qc[c])))
        ax.axhline(0, color="k", lw=0.4); ax.set_xlim(0, 1900); ax.set_xticks([0, 500, 1000, 1500])
        ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("Response energy (Hz², noise-corrected)")
    late = U[U.t_ms > 400].signal; axs[1].set_ylim(np.percentile(late, 0.5) * 1.3, np.percentile(late, 99.5) * 1.3)
    axs[0].set_title("a  Distance from baseline", loc="left"); axs[1].set_title("b  Zoom on the tail", loc="left")
    axs[0].text(0.98, 0.95, "orange: geometry window (2–502 ms)\ngrey: current baseline (1.0–1.9 s)", transform=axs[0].transAxes, fontsize=4.8, ha="right", va="top")
    # c: cumulative fraction of 2-1892 ms signal energy, responsive conditions only
    ax = axs[2]; tot = U.groupby("cond").signal.sum(); nz = U.groupby("cond").noise.sum(); resp = tot[tot > 3 * nz].index
    C = np.stack([np.cumsum(np.clip(g.signal.values, 0, None)) / np.clip(g.signal.values, 0, None).sum() for c, g in U.groupby("cond") if c in resp]); tt = U[U.cond == resp[0]].t_ms.values
    for row, c in zip(C, [c for c in U.cond.unique() if c in resp]): ax.plot(tt, row, lw=0.5, color=cmap(nq(qc[c])), alpha=0.8)
    med = np.median(C, 0); ax.plot(tt, med, color="k", lw=1.2)
    for tm in (500, 1000):
        v = med[np.searchsorted(tt, tm)]; ax.plot([tm], [v], "o", color="k", ms=3); ax.text(tm + 30, v - (0.09 if tm == 500 else 0.20), f"{v:.0%} by {tm} ms", fontsize=5)
    ax.set_xlim(0, 1900); ax.set_ylim(0, 1.02); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("Cumulative fraction of response energy")
    ax.set_title(f"c  Cumulative energy ({len(resp)} strongest conditions)", loc="left")
    # d: energy in PCA(2-1002 ms, k=3) space
    ax = axs[3]; shade(ax); P3 = EN[(EN["sort"] == "online") & (EN.space == "pca3_fit_2-1002ms")]
    for c, g in P3.groupby("cond"): ax.plot(g.t_ms, g.signal, lw=0.6, color=cmap(nq(qc[c])))
    ax.axhline(0, color="k", lw=0.4); ax.set_xlim(0, 1900); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("Energy in PC1–3 (Hz², noise-corrected)")
    ax.set_title("d  PCA fitted on 2–1002 ms", loc="left")
    # e: PC1 time course over the whole chunk, amplitude series
    ax = axs[4]; shade(ax); cs = list(PZ["conds"]); fam, amps, labs_ = mf.family(D); amax = max(abs(a) for a in amps)
    for a in amps:
        if labs_[a] not in cs: continue
        i = cs.index(labs_[a]); col = mf.ampcolor(a, amax); ax.plot(PZ["t_ms"], PZ["pc_online"][i, :, 0], lw=0.8, color=col); ax.plot(PZ["t_ms"], PZ["pc_ks25"][i, :, 0], lw=0.5, ls="--", color=col)

    ax.axhline(0, color="k", lw=0.4); ax.set_xlim(0, 1900); ax.set_xticks([0, 500, 1000, 1500]); ax.set_xlabel("Time after pulse (ms)"); ax.set_ylabel("PC1 (0 = pre-pulse baseline)")
    ax.set_title(f"e  PC1, {fam}", loc="left")
    ax.legend(handles=[Line2D([], [], color=mf.ampcolor(-amax, amax), lw=0.8, label=f"−{amax:g} µA"), Line2D([], [], color=mf.ampcolor(amax, amax), lw=0.8, label=f"+{amax:g} µA"), Line2D([], [], color="0.4", lw=0.8, label="online"),
                       Line2D([], [], color="0.4", lw=0.5, ls="--", label="KS2.5")], frameon=False, fontsize=4.8, loc="upper right", ncol=2)
    # f: condition information per 100 ms segment
    ax = axs[5]; shade(ax)
    for bl, ls in (("pre", "-"), ("none", ":")):
        g = TC[TC.baseline == bl]; x = (g.start_ms + g.end_ms) / 2
        ax.plot(x, g.acc_online, ls, color="#DD8452", lw=1, marker="o", ms=2, label=f"online, {'pre-pulse baseline' if bl == 'pre' else 'no baseline'}")
        ax.plot(x, g.acc_ks25, ls, color="0.4", lw=0.8, label=f"KS2.5, {'pre-pulse baseline' if bl == 'pre' else 'no baseline'}")
    ax.axhline(1 / len(cs), color="k", lw=0.5, ls="--"); ax.text(1880, 1 / len(cs) + 0.005, "chance", fontsize=4.8, ha="right")
    ax.set_xlim(0, 1900); ax.set_xticks([0, 500, 1000, 1500]); ax.set_ylim(0, None); ax.set_xlabel("100 ms segment (ms after pulse)"); ax.set_ylabel(f"{len(cs)}-way accuracy (3 PCs)")
    ax.set_title("f  Condition information over time", loc="left"); ax.legend(frameon=False, fontsize=4.5, loc="upper right")
    sm = mpl.cm.ScalarMappable(norm=nq, cmap=cmap); cb = fig.colorbar(sm, cax=fig.add_axes([0.93, 0.58, 0.008, 0.28])); cb.set_label("log10 charge (pC)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    fig.suptitle(f"{TITLE}: is there stimulus-driven signal after 500 ms? (pre-pulse baseline = previous trial's 1.4–1.9 s)", fontsize=7.5, x=0.02, ha="left", y=1.0)
    mf.save(fig, "window_analysis.png")

REPROG_MS = float(os.environ.get("REPROG_MS", "100"))   # stimulator reprogramming (assumed fixed)

def pooled_lag(BL, budget):
    """Per trial and per buffer position after the pulse: the pooled lag = the slowest probe's lag behind the write head."""
    B = BL[(BL.since_pulse_ms >= 0) & (BL.since_pulse_ms < 2000)].copy(); parts = []
    for p, g in BL.groupby("probe", sort=False):
        g = g.copy(); g["cycle"] = (g.since_pulse_ms.diff() < 0).cumsum(); parts.append(g)
    B = pd.concat(parts); B = B[(B.since_pulse_ms >= 0) & (B.since_pulse_ms < 2000)]
    B["bin"] = (B.since_pulse_ms // budget).astype(int)
    M = B.groupby(["cycle", "bin"]).behind_ms.max().unstack("bin")
    t = (M.columns.values + 0.5) * budget
    return t, M.mean(0).values, M.std(0).values, M.quantile(0.99).values, len(M)

def fig_live_budget(D):
    need = ["iti_population_push.csv", "demo_population_buffers.csv", "window_sweep.csv", "window_loss_timing.json", "demo_population_summary.json"]
    if not all(os.path.exists(ctx.RP(n)) for n in need): print("[fig] live budget: inputs missing"); return
    S = json.load(open(ctx.RP("demo_population_summary.json"))); PU = pd.read_csv(ctx.RP("iti_population_push.csv")); BL = pd.read_csv(ctx.RP("demo_population_buffers.csv"))
    SW = pd.read_csv(ctx.RP("window_sweep.csv")); LT = json.load(open(ctx.RP("window_loss_timing.json")))
    fs = 29970.0757 if ctx.REC == "jlh32" else 30000.0; budget = 1024 / fs * 1e3; W = 500.0; ITI = ITI_MS
    t, mu, sd, q99, ncyc = pooled_lag(BL, budget)
    p5 = PU[PU.window_s == 0.5]; dec = (p5.latency_ms - p5.geometry_ms); geo = p5.geometry_ms; loss = LT["procrustes_ms_mean"]
    w_end = 1.0 + W; t_dec = w_end + dec.mean(); t_geo = t_dec + geo.mean(); t_loss = t_geo + loss; t_prog = t_loss + REPROG_MS
    fig = plt.figure(figsize=(7.4, 4.6)); gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], height_ratios=[1.5, 1], hspace=0.12, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); axg = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.fill_between(t, np.clip(mu - sd, 0, None), mu + sd, color="#4C72B0", alpha=0.25, lw=0, label="± 1 SD across trials")
    ax.plot(t, mu, color="#4C72B0", lw=1.2, label=f"mean pooled lag (slowest of {len(S['probes'])} probes)"); ax.plot(t, q99, color="#4C72B0", lw=0.6, ls=":", label="99th percentile")
    ax.axvspan(1.0, w_end, color="#DD8452", alpha=0.10, lw=0); ax.axvline(w_end, color="#DD8452", lw=0.8)
    ax.set_ylabel("Decoder lag behind\nwrite head (ms)"); ax.set_ylim(0, max(q99.max() * 1.15, 50)); ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=5, loc="upper right"); ax.set_title(f"a  One trial cycle, {len(S['probes'])} probes decoded live ({ncyc} trials, real-time replay)", loc="left")
    ax.text(w_end + 8, ax.get_ylim()[1] * 0.92, "window end", color="#DD8452", fontsize=5, va="top")
    # timeline
    rows = [("analysis window\n(2–502 ms)", 1.0, W, "#DD8452"), ("decode catch-up", w_end, dec.mean(), "#4C72B0"), ("geometry", t_dec, geo.mean(), "#55A868"),
            ("loss (Procrustes)", t_geo, loss, "#8172B3"), (f"stim reprogramming\n({REPROG_MS:.0f} ms, assumed)", t_loss, REPROG_MS, "#C44E52"),
            ("next trial's baseline\n(0.5 s pre-pulse)", ITI - 500, 500, "0.75")]
    for y, (lab, x0, wd, col) in enumerate(rows):
        axg.barh(y, max(wd, 3), left=x0, color=col, height=0.62, lw=0)
    axg.errorbar([w_end + dec.mean()], [1], xerr=[[dec.std()], [dec.std()]], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
    txt = {1: f"{dec.mean():.0f} ± {dec.std():.0f} ms (p99 {dec.quantile(.99):.0f})", 2: f"{geo.mean():.2f} ms", 3: f"{loss:.3f} ms", 4: f"done at {t_prog:.0f} ms"}
    for y, tx in txt.items(): axg.text(rows[y][1] + max(rows[y][2], 3) + 15, y, tx, va="center", fontsize=5)
    axg.axvline(0, color="k", lw=1.0); axg.axvline(ITI, color="k", lw=1.0); axg.text(ITI - 10, -0.9, f"next pulse ({ITI / 1e3:.1f} s)", ha="right", fontsize=5); axg.text(10, -0.9, "pulse", fontsize=5)
    axg.annotate("", xy=(ITI - 500, 5.0), xytext=(t_prog, 5.0), arrowprops=dict(arrowstyle="<->", lw=0.6, color="0.3"))
    axg.text((t_prog + ITI - 500) / 2, 4.6, f"{ITI - 500 - t_prog:.0f} ms spare", ha="center", fontsize=5, color="0.3")
    fig.text(0.02, -0.10, "Decode catch-up = time from the last window sample being written until every probe's decoder has finalised it (buffer fill ≤ 34 ms + decoder look-ahead + lag; mean ± SD, 150 trials). "
             "Geometry = binning + projection of the pooled 2–502 ms window; loss = Procrustes distance to a condition target (both measured). Reprogramming time is an assumed constant (REPROG_MS).",
             fontsize=4.8, color="0.3", ha="left", va="top", wrap=True)
    axg.set_yticks(range(len(rows))); axg.set_yticklabels([r[0] for r in rows], fontsize=5); axg.set_ylim(len(rows) - 0.5, -1.2)
    axg.set_xlim(-30, ITI + 30); axg.set_xticks(np.arange(0, ITI + 1, 250)); axg.set_xlabel("Time after pulse (ms)")
    for a_ in (ax, axg): a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
    # b: accuracy vs window length
    axb = fig.add_subplot(gs[:, 1]); ch = SW.chance.iloc[0]
    for (bl, k), g in SW.groupby(["baseline", "k"]):
        col = "#DD8452" if bl == "pre" else "#4C72B0"; ls = "-" if k == 3 else "--"
        axb.plot(g.window_end_ms, g.acc_online, ls, color=col, lw=1.1 if k == 3 else 0.8, marker="o", ms=2.2, label=f"{'pre-pulse' if bl == 'pre' else 'same-chunk 1.0–1.9 s'} baseline, {k} PCs")
        axb.plot(g.window_end_ms, g.acc_ks25, ls, color=col, lw=0.4, alpha=0.6)
    axb.axhline(ch, color="k", lw=0.5, ls=":"); axb.text(1500, ch + 0.004, "chance", fontsize=5, ha="right")
    axb.axvline(502, color="0.5", lw=0.6); axb.text(512, ch + 0.02, "current\n(502 ms)", fontsize=5, va="bottom")
    axb.set_xlim(0, 1550); axb.set_xticks([0, 250, 500, 750, 1000, 1250, 1500]); axb.set_ylim(0, None)
    axb.set_xlabel("Window end (ms after pulse)"); axb.set_ylabel(f"{int(round(1 / ch))}-way single-trial accuracy")
    axb.set_title("b  Accuracy vs window length", loc="left"); axb.legend(frameon=False, fontsize=4.6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, title="window 2 ms → end; thick: online, thin: KS2.5", title_fontsize=4.6)
    fig.suptitle(f"{TITLE}: live closed-loop timeline with a 0.5 s window and pre-pulse baseline (one process, Apple M1)", fontsize=7.5, x=0.02, ha="left", y=1.0)
    mf.save(fig, "live_timeline.png")

def main():
    # point the shared figure helpers at the population outputs
    mf.ctx.R = ctx.RP; mf.ctx.F = ctx.FP; mf.TITLE = TITLE
    probes = json.load(open(ctx.RP("population_summary.json")))["probes"]
    D = dict(tr=pd.read_csv(ctx.R_probe(ctx.REC, probes[0], "trials.csv")))
    for f, a in [(fig_population, (D,)), (fig_pcsweep, ()), (mf.fig_comparison, ()), (mf.fig_alignment, (D,)), (fig_popdemo, ()), (fig_itipop, ()), (fig_closed_loop, ()), (fig_window, (D,)), (fig_live_budget, (D,))]:
        try:
            f(*a)
        except Exception as e:
            import traceback; print(f"[fig] {f.__name__} FAILED: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()
