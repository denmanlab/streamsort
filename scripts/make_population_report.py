"""Recording-level (pooled probes) review report.  build(marker) for in-session artifact links;
`python scripts/make_population_report.py` writes reports/<REC>_population_report.md with relative links."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx

def build(marker, notes=None, probe_reports=None):
    j = lambda n: json.load(open(ctx.RP(n))) if os.path.exists(ctx.RP(n)) else {}
    S, M, D = j("population_summary.json"), j("geometry_comparison_summary.json"), j("demo_population_summary.json")
    SW = pd.read_csv(ctx.RP("subset_sweep.csv")); dep = S.get("deployment", {})
    per = {p: json.load(open(ctx.R_probe(ctx.REC, p, "geometry_comparison_summary.json"))) for p in S["probes"]}
    single = SW[SW.n_probes == 1].set_index("probes"); allp = SW[SW.n_probes == len(S["probes"])].iloc[0]
    fg = lambda fn, cap: f"![{cap}]({marker(fn)})" if marker(fn) else f"*({fn} not available)*"
    v = lambda ok, part=False: "✓ Supported" if ok else ("◐ Partly" if part else "✗ Not supported")
    rows = [
        ("P1", "Pooling probes adds condition information beyond the best single probe",
         v(allp.acc_online > single.acc_online.max() * 1.1, allp.acc_online >= single.acc_online.max()),
         f"Drift-controlled {M['n_conditions']}-way accuracy (online): " + ", ".join(f"{p} {single.acc_online[p]:.3f}" for p in single.index)
         + f"; all {len(S['probes'])} probes {allp.acc_online:.3f} (KS2.5 {allp.acc_ks25:.3f}; chance {M['chance']:.3f}). High charge: {M['acc_baselined_high_online']:.2f}."),
        ("P2", "Pooled online geometry matches pooled offline geometry",
         v(M["rho_procrustes_online_vs_ks25"] >= 0.8 and M["cross_half_online_even_vs_ks25_odd"] >= 0.9 * M["split_half_ks25"], M["rho_procrustes_online_vs_ks25"] >= 0.6),
         f"Condition-distance matrices ρ = {M['rho_procrustes_online_vs_ks25']:.2f}; online-even vs KS2.5-odd ρ = {M['cross_half_online_even_vs_ks25_odd']:.2f} against KS2.5 split-half {M['split_half_ks25']:.2f}. "
         f"Same-trial Procrustes {M['procrustes_same_trial_median']:.2f} vs trial-to-condition-mean {M['procrustes_ks25_vs_own_mean_median']:.2f}."),
        ("P3", "The deployment population (all usable online units) recovers the offline condition geometry",
         v(dep.get("rho_condition_geometry", 0) >= 0.8, dep.get("rho_condition_geometry", 0) >= 0.6),
         f"{dep.get('n_online_usable')} usable online units vs {dep.get('n_ks25_good')} good KS2.5 units (each with its own basis): condition geometry ρ = {dep.get('rho_condition_geometry', float('nan')):.2f}; "
         f"accuracy {dep.get('acc_online_all', float('nan')):.3f} vs {dep.get('acc_ks25_good', float('nan')):.3f}; same-trial Procrustes {dep.get('procrustes_same_trial_median', float('nan')):.2f}."),
        ("P4", "Pooled geometry is cheap enough to compute between trials", v(S.get("geometry_ms_p99", 1e9) < 10),
         f"Bin + project + Procrustes for {S['n_units_matched']} pooled units: median {S.get('geometry_ms_median', float('nan')):.2f} ms, p99 {S.get('geometry_ms_p99', float('nan')):.2f} ms per trial."),
        ("P5", "All probes can be decoded live together on the test machine",
         v(bool(D) and D.get("rtf_total", 9) < 0.8 and D.get("latency_ms_p99", 1e9) < 500, bool(D) and D.get("rtf_total", 9) < 1.0),
         (f"Concurrent replay of runs {D.get('runs')} ({D.get('n_trials')} trials): pooled geometry {D.get('latency_ms_median', float('nan')):.0f} / {D.get('latency_ms_p99', float('nan')):.0f} ms (median / p99) after the last probe's window end; "
          f"total real-time factor {D.get('rtf_total', float('nan')):.2f} (" + ", ".join(f"{k} {x:.2f}" for k, x in D.get('rtf_per_probe', {}).items()) + f"); max lag " + ", ".join(f"{k} {x:.0f} ms" for k, x in D.get('behind_ms_max', {}).items())
          + f"; spikes identical to per-probe pull decode (min Jaccard {D.get('min_jaccard_vs_pull', float('nan')):.3f})." if D else "not run")),
    ]
    out = [f"# streamsort — {ctx.REC}: all probes pooled into one population", "",
           f"**Probes:** {', '.join(S['probes'])} (option `PROBES=`; default all finished probes). **Pooled units:** {S['n_units_matched']} KS2.5-matched ("
           + ", ".join(f"{p} {n}" for p, n in S["units_per_probe"].items()) + f"); {dep.get('n_online_usable')} usable online units in the deployment population.",
           "**Method:** every probe is decoded with its own clock, artifact suppressor and template bank. Per trial, the probes' binned rates (2–502 ms, 10 ms bins, σ 20 ms, same-chunk baseline referencing) are concatenated into one population, and the same geometry suite as for a single probe is applied (fixed 3-PC basis, Procrustes, condition geometry, trial alignment).", "",
           "## 1. Verdicts", "", "| # | Claim | Verdict | Evidence |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    out += ["", "## 2. Probe subsets", "", "| Probes | Units | Accuracy online | Accuracy KS2.5 | High-charge online | Same-trial Procrustes | Condition-geometry ρ | Split-half KS2.5 |", "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r.probes} | {r.n_units} | {r.acc_online:.3f} | {r.acc_ks25:.3f} | {r.acc_high_online:.3f} | {r.procrustes_same_trial:.2f} | {r.rho_online_vs_ks25:.2f} | {r.split_half_ks25:.2f} |" for _, r in SW.iterrows()]
    cb = M.get("cross_block")
    if cb:
        out += ["", f"**Cross-block test (pooled):** {cb['online']['cross_block_acc']:.2f} across blocks vs {cb['online']['within_block_acc']:.2f} within (chance {cb['chance']:.2f})."]
    if os.path.exists(ctx.RP("pc_count_sweep.csv")):
        PC = pd.read_csv(ctx.RP("pc_count_sweep.csv")); ks = sorted(PC.k.unique())
        out += ["", "## 3. Dimensionality: accuracy vs number of PCs", "",
                "Same drift-controlled nearest-mean readout as in section 2; only the number of PCs in the fixed basis changes. Online accuracy (KS2.5 in parentheses).", "",
                "| Normalisation | Probes | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
        for (nm, pr), g in PC.assign(_o=PC.probes.map({q: i for i, q in enumerate(PC.probes.unique())})).sort_values(["norm", "_o"]).groupby(["norm", "probes"], sort=False):
            g = g.set_index("k"); out.append(f"| {nm} | {pr} ({int(g.n_units.iloc[0])}) | " + " | ".join(f"{g.acc_online[k]:.3f} ({g.acc_ks25[k]:.3f})" for k in ks) + " |")
        out += ["", fg(f"{ctx.POP_TAG}_pc_count_sweep.png", "Accuracy vs PC count")]
    try:
        Pz = np.load(ctx.RP("geometry_per_trial_sup.npz"), allow_pickle=True); Bv = Pz["basis"]; pou = np.load(ctx.RP("probe_of_unit.npy"), allow_pickle=True)
        E = Bv ** 2; out += ["", "## 4. Contribution of each probe to the pooled basis", "", "| Probe | Unit share | " + " | ".join(f"PC{j+1} energy" for j in range(Bv.shape[1])) + " |", "|---|---|" + "---|" * Bv.shape[1]]
        for pr in S["probes"]:
            out.append(f"| {pr} | {np.mean(pou == pr):.2f} | " + " | ".join(f"{E[pou == pr, j].sum() / E[:, j].sum():.2f}" for j in range(Bv.shape[1])) + " |")
    except Exception as e:
        out += ["", f"*(basis loadings unavailable: {e})*"]
    if os.path.exists(ctx.RP("iti_population_budget.csv")):
        IT = pd.read_csv(ctx.RP("iti_population_budget.csv"))
        out += ["", f"## 4b. Inter-trial interval budget, {len(S['probes'])} probes live", "",
                "**Wait** = per trial, time from the last sample of the analysis window being written to the pooled geometry being ready. **p99** = 99th percentile of that wait over trials (only ~1 trial in 100 is slower). "
                "**Push** decodes each 1024-sample buffer as it arrives; **pull** decodes each probe's whole window from disk, one probe after another, once the window has ended. "
                "**Min closed-loop ITI** = 1.2 ms (pulse + 1 ms) + window + p99 wait, with a 1.0 s floor for windows < 1.9 s (0.5 s quiet pre-pulse baseline after a ~0.5 s response).", "",
                "| Window | Mode | Trials | Wait median (ms) | Wait p99 (ms) | Wait max (ms) | Min closed-loop ITI (s) | Fits 2.0 s |", "|---|---|---|---|---|---|---|---|"]
        out += [f"| {r.window_s:.1f} s | {r['mode']} | {r.n_trials} | {r.wait_median_ms:.0f} | {r.wait_p99_ms:.0f} | {r.wait_max_ms:.0f} | {r.min_iti_closed_loop_s:.2f} | {'✓' if r.fits_current_iti else '✗'} |" for _, r in IT.iterrows()]
        out += ["", fg(f"{ctx.POP_TAG}_iti_population.png", "ITI budget, all probes live")]
    if os.path.exists(ctx.RP("closed_loop_budget.csv")):
        CL = pd.read_csv(ctx.RP("closed_loop_budget.csv"))
        out += ["", "## 4c. Closed-loop budget: decode → geometry → loss → stimulator update before the next pulse", "",
                f"Decode + geometry wait measured per trial; loss/distance ({CL.loss_ms.iloc[0]:.0f} ms) and serial stimulation command ({CL.command_ms.iloc[0]:.0f} ms) are assumed constants "
                "(`LOSS_MS`, `CMD_MS`, `ITI_MS` settings). Spare = time left before the next pulse. "
                "Not included: acquisition-to-software latency on the rig (in the replay a sample is available the instant it is 'written').", "",
                "| Window | Mode | Wait p99 (ms) | Ready for next pulse, p99 (ms after pulse) | Spare, median (ms) | Spare, worst 1% (ms) | Trials in time | ITI needed (s) | Longest window at current ITI (s) |",
                "|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {r.window_s:.1f} s | {r['mode']} | {r.wait_p99_ms:.0f} | {r.ready_p99_ms:.0f} | {r.slack_median_ms:.0f} | {r.slack_worst1pct_ms:.0f} | {r.frac_trials_in_time:.0%} | {r.min_iti_p99_s:.3f} | {r.max_window_at_iti_p99_s:.3f} |" for _, r in CL.iterrows()]
        out += ["", fg(f"{ctx.POP_TAG}_closed_loop_budget.png", "Closed-loop budget")]
    if os.path.exists(ctx.RP("window_analysis.csv")):
        WA = pd.read_csv(ctx.RP("window_analysis.csv")); EN = pd.read_csv(ctx.RP("window_energy.csv")); BB = pd.read_csv(ctx.RP("window_baseline_bias.csv"))
        U = EN[(EN["sort"] == "online") & (EN.space == "pca3_fit_2-1002ms")]; tot = U.groupby("cond").signal.sum(); resp = tot[tot > 3 * U.groupby("cond").noise.sum()].index
        G = U[U.cond.isin(resp)].assign(s=lambda d: d.signal.clip(lower=0)); fr = lambda a, b: (G[(G.t_ms >= a) & (G.t_ms < b)].groupby("cond").s.sum() / G.groupby("cond").s.sum()).median()
        pk = G.groupby("cond").signal.max(); lv = lambda a, b: (G[(G.t_ms >= a) & (G.t_ms < b)].groupby("cond").signal.mean() / pk).median(); bo = BB[BB["sort"] == "online"]
        out += ["", "## 4d. Which part of the post-pulse chunk carries the response?", "",
                "Pre-pulse baseline = the previous trial's 1.4–1.9 s (≈0.6–0.1 s before this pulse); trials whose previous trial is in another run are excluded. "
                f"In a PCA fitted on 2–1002 ms condition means (3 PCs, baseline at the origin), the {len(resp)} clearly responsive conditions put a median {fr(0, 502):.0%} of their noise-corrected response energy in 2–502 ms, "
                f"{fr(502, 1002):.0%} in 502–1002 ms and {fr(1002, 1900):.1%} after 1 s. Energy at 400–500 ms is {lv(400, 500):.1%} of the peak, at 900–1000 ms {lv(900, 1000):.1%}, and after 1.4 s {lv(1400, 1890):+.1%} (noise). "
                f"The same-chunk 1.0–1.9 s baseline differs from the pre-pulse baseline by more than 2× its noise in {int((bo.signal > 2 * bo.noise).sum())}/{len(bo)} conditions.", "",
                "| Response window | Baseline | Accuracy online, 3 PCs | Accuracy KS2.5, 3 PCs | Accuracy online, 10 PCs | Condition-geometry ρ |", "|---|---|---|---|---|---|"]
        for (w, bl), g in WA[WA.window != "baseline only"].groupby(["window", "baseline"], sort=False):
            g3, g10 = g[g.k == 3].iloc[0], g[g.k == 10].iloc[0]
            out.append(f"| {w} | {bl} | {g3.acc_online:.3f} | {g3.acc_ks25:.3f} | {g10.acc_online:.3f} | {g3.rho_online_vs_ks25:.2f} |")
        bl_only = WA[WA.window == "baseline only"].set_index("baseline")
        out += ["", f"Chance {WA.chance.iloc[0]:.3f}. Baseline rates alone decode condition at {bl_only.acc_online.get('same', float('nan')):.3f} (same-chunk) / {bl_only.acc_online.get('pre', float('nan')):.3f} (pre-pulse) because conditions were recorded in blocks: 'none' rows measure session drift, not stimulus information.",
                "", fg(f"{ctx.POP_TAG}_window_analysis.png", "Response over the whole chunk")]
        if os.path.exists(ctx.RP("window_sweep.csv")):
            WS = pd.read_csv(ctx.RP("window_sweep.csv")); WS = WS[WS.k == 3]; ends = [e for e in (102, 202, 302, 502, 702, 1002, 1502) if e in set(WS.window_end_ms)]
            out += ["", "**Accuracy vs window length** (window 2 ms → end, 3 PCs, online; KS2.5 in parentheses):", "",
                    "| Baseline | " + " | ".join(f"{int(e)} ms" for e in ends) + " |", "|---|" + "---|" * len(ends)]
            for bl, g in WS.groupby("baseline"):
                g = g.set_index("window_end_ms")
                out.append(f"| {'pre-pulse' if bl == 'pre' else 'same-chunk 1.0–1.9 s'} | " + " | ".join(f"{g.acc_online[e]:.3f} ({g.acc_ks25[e]:.3f})" if e in g.index else "—" for e in ends) + " |")
    if os.path.exists(ctx.RP("window_loss_timing.json")) and os.path.exists(ctx.RP("iti_population_push.csv")):
        LT = json.load(open(ctx.RP("window_loss_timing.json"))); PU = pd.read_csv(ctx.RP("iti_population_push.csv")); p5 = PU[PU.window_s == 0.5]; dec = p5.latency_ms - p5.geometry_ms
        rp = float(os.environ.get("REPROG_MS", "100")); done = 1.0 + 500 + dec.mean() + p5.geometry_ms.mean() + LT["procrustes_ms_mean"] + rp
        out += ["", f"## 4e. Live closed-loop timeline (0.5 s window, pre-pulse baseline, {rp:.0f} ms stimulator reprogramming)", "",
                f"Decode catch-up after the window end {dec.mean():.0f} ± {dec.std():.0f} ms (p99 {dec.quantile(.99):.0f}); geometry {p5.geometry_ms.mean():.2f} ms; loss (Procrustes to a condition target) {LT['procrustes_ms_mean']:.3f} ms; "
                f"reprogramming {rp:.0f} ms (assumed) → stimulator updated at ~{done:.0f} ms after the pulse, {1500 - done:.0f} ms before the next trial's 0.5 s pre-pulse baseline starts (2.0 s ITI).",
                "", fg(f"{ctx.POP_TAG}_live_timeline.png", "Live closed-loop timeline")]
    out += ["", "## 5. Figures", "", fg(f"{ctx.POP_TAG}_geometry_population.png", "Pooled geometry") + "\n", fg(f"{ctx.POP_TAG}_geometry_comparison.png", "Pooled condition geometry") + "\n",
            fg(f"{ctx.POP_TAG}_trial_alignment.png", "Pooled single-trial alignment") + "\n", fg(f"{ctx.POP_TAG}_demo_population.png", "Concurrent real-time replay") + "\n",
            "## 6. Per-probe reports", ""] + [(f"- [probe {p}]({probe_reports[p]})" if probe_reports and probe_reports.get(p) else f"- probe {p}") + f": accuracy {per[p]['acc_baselined_online']:.3f}, condition-geometry ρ {per[p]['rho_procrustes_online_vs_ks25']:.2f} ({per[p]['n_units']} units)" for p in S["probes"]]
    if notes:
        out += ["", "## 7. Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(out)

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    mk = lambda fn: os.path.relpath(ctx.POP_FIG / fn, "reports") if (ctx.POP_FIG / fn).exists() else None
    open(f"reports/{ctx.POP_TAG}_report.md", "w").write(build(mk)); print(f"reports/{ctx.POP_TAG}_report.md")
