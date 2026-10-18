"""Per-probe review report (markdown), same claims and fixed thresholds as the Phase 1 report.

build_report(marker) -> str ; `marker(filename)` returns a link target for a saved artifact
(or None).  Run from the analysis kernel so artifact markers can be resolved."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx

def _j(name):
    p = ctx.R(name); return json.load(open(p)) if os.path.exists(p) else {}

def verdict(ok, partial=None):
    return "✓ Supported" if ok else ("◐ Partly" if partial else "✗ Not supported")

def build_report(marker):
    S, G, V, M, D = _j("session.json"), _j("gate_summary.json"), _j("validation_summary_sup.json"), _j("geometry_comparison_summary.json"), _j("demo_live_summary.json")
    fit = {}
    fp = ctx.SCRATCH / "fit_report_kilosort4_seg600.json"
    if fp.exists(): fit = json.load(open(fp))
    T = pd.read_csv(ctx.R("geometry_timing.csv")) if os.path.exists(ctx.R("geometry_timing.csv")) else None
    wf = pd.read_csv(ctx.R("diag_early_waveforms.csv")) if os.path.exists(ctx.R("diag_early_waveforms.csv")) else None
    LT = pd.read_csv(ctx.R("validation_by_latency_sup.csv")); RS = pd.read_csv(ctx.R("validation_by_run_sup.csv"))
    ITI = pd.read_csv(ctx.R("iti_budget.csv")) if os.path.exists(ctx.R("iti_budget.csv")) else None
    lk = lambda fn, label=None: (f"[{label or fn}]({marker(fn)})" if marker(fn) else f"`{label or fn}`")
    fg = lambda fn, cap: (f"![{cap}]({marker(fn)})" if marker(fn) else f"*(figure {fn} not saved)*")
    pre = "" if ctx.LEGACY else f"{ctx.TAG}_"
    sta = None
    if wf is not None:
        ok = wf.groupby("unit").filter(lambda d: (d.set_index("cls").n.reindex(["2-5", "late"]).fillna(0) >= 15).all())
        sta = {c: float(ok[ok.cls == c].cos_sta.median()) if (ok.cls == c).any() else float("nan") for c in ["1-2", "2-5", "late", "2-5_rand"]}; sta["n_units"] = int(ok.unit.nunique())
    gsplit = ""
    if os.path.exists(ctx.R("sort_agreement_report.csv")):
        rp = pd.read_csv(ctx.R("sort_agreement_report.csv"), index_col=0); a_ = rp[(rp.group == "good") & (rp.n_ref_in_segment >= 50)]
        mm = a_.agreement.fillna(0) > 0.5; fast = a_.n_ref_in_segment / 600 >= 1
        gsplit = (f" By rate: units ≥ 1 Hz in the sort window matched {int(mm[fast].sum())}/{int(fast.sum())} ({mm[fast].mean():.0%}), units < 1 Hz {int(mm[~fast].sum())}/{int((~fast).sum())} ({mm[~fast].mean():.0%}).")
    tfix = float(T[(T.operation == "total_fixed")].p99_ms.max()) if T is not None else float("nan")
    ho = V.get("heldout", {}).get("median_unit_f1", float("nan")); late = V.get("late", {}).get("median_unit_f1", float("nan"))
    lat = V.get("decode_latency_ms", {}); ch = M.get("chance", float("nan")); acc = M.get("acc_baselined_online", float("nan"))
    rows = [
        ("1", "The recording replays as a causal stream with exact sample timing",
         verdict(S.get("reconcile", {}).get("n_matched") == S.get("n_trials") and S.get("artifact", {}).get("frac_onset_detected", 0) == 1.0),
         f"{S.get('reconcile', {}).get('n_matched')}/{S.get('n_trials')} trials matched to TTL edges; artifact onset found on {S.get('artifact', {}).get('frac_onset_detected', 0):.0%} of sampled trials. {S.get('ttl_offset_note', '')}."),
        ("2", "A sort fitted on the first 600 s matches the full offline sort",
         ("✓ Supported" + (" (narrowly)" if G.get("frac_good_active_matched_gt0.5", 0) < 0.75 else "")) if G.get("PASS", False) else "✗ Not supported",
         f"Kilosort 4: {G.get('n_new_units')} units in {fit.get('sort_seconds', float('nan'))/60:.0f} min. {G.get('good_active_matched_gt0.5')}/{G.get('n_ref_good_active')} active good KS2.5 units matched at agreement > 0.5 ({G.get('frac_good_active_matched_gt0.5', float('nan')):.1%}; criterion ≥ 70%); median accuracy {G.get('median_accuracy_matched', float('nan')):.3f}, template r {G.get('median_waveform_corr_matched', float('nan')):.2f}.{gsplit}"),
        ("3", "Each 1.9 s post-pulse chunk is template-matched in < 1 s", verdict(lat.get("p99", 9e9) < 1000),
         f"Pull mode, all trials: median {lat.get('median', float('nan')):.0f} ms, p99 {lat.get('p99', float('nan')):.0f} ms, max {lat.get('max', float('nan')):.0f} ms." + (f" Push mode: geometry ready {D.get('latency_ms_median', float('nan')):.0f} / {D.get('latency_ms_p99', float('nan')):.0f} ms (median / p99) after the window's last sample." if D else "")),
        ("4", "Online spikes reproduce the offline sort", verdict(ho >= 0.75, ho >= 0.5),
         f"{V.get('n_pairs')} matched units. Median unit F1: in-sort-window {V.get('insample', {}).get('median_unit_f1', float('nan')):.2f}, after {ho:.2f}, first quarter {V.get('early', {}).get('median_unit_f1', float('nan')):.2f}, last quarter {late:.2f}. Per-trial count r {V.get('median_count_corr', float('nan')):.2f}; per-run F1 range {RS.f1.min():.2f}–{RS.f1.max():.2f}; online/reference count ratio {RS.online_over_ref.min():.2f}–{RS.online_over_ref.max():.2f}."),
        ("5", "Near-pulse detections (2–5 ms) are spikes, not artifact", verdict(sta is not None and sta["2-5"] >= 0.7, sta is not None and sta["2-5"] >= 0.5),
         (f"STA–template cosine: 2–5 ms {sta['2-5']:.2f}, ≥ 20 ms {sta['late']:.2f}, random times {sta['2-5_rand']:.2f}, 1–2 ms {sta['1-2']:.2f} ({sta['n_units']} units). " if sta else "not run. ")
         + f"Precision / recall vs KS2.5 at 1–3 ms {LT.precision.iloc[0]:.2f} / {LT.recall.iloc[0]:.2f}; from 10 ms on {LT[LT.lat_lo_ms >= 10].precision.mean():.2f} / {LT[LT.lat_lo_ms >= 10].recall.mean():.2f}."),
        ("6", "Per-trial geometry is cheap enough to compute between trials", verdict(tfix < 5), f"Fixed-basis path ≤ {tfix:.2f} ms p99 per trial. {lk(pre + 'geometry_timing.csv', 'geometry_timing.csv') if False else ''}"),
        ("7", "Online geometry matches geometry from the offline sort",
         verdict(M.get("rho_procrustes_online_vs_ks25", 0) >= 0.8 and M.get("cross_half_online_even_vs_ks25_odd", 0) >= 0.9 * M.get("split_half_ks25", 1), M.get("rho_procrustes_online_vs_ks25", 0) >= 0.6),
         f"Same-trial Procrustes online vs KS2.5 {M.get('procrustes_same_trial_median', float('nan')):.2f} vs trial-to-condition-mean {M.get('procrustes_ks25_vs_own_mean_median', float('nan')):.2f}. Condition-distance matrices ρ = {M.get('rho_procrustes_online_vs_ks25', float('nan')):.2f} (Mantel p = {M.get('mantel_p', float('nan')):.4f}); online-even vs KS2.5-odd ρ = {M.get('cross_half_online_even_vs_ks25_odd', float('nan')):.2f} against KS2.5 split-half {M.get('split_half_ks25', float('nan')):.2f}. Per-trial gain r = {M.get('gain_corr_per_trial_online_vs_ks25', float('nan')):.2f}."),
        ("8", "Single-trial geometry carries condition information", ("✓ Supported" if acc >= 0.5 else "◐ Modest" if acc >= 2 * ch else "✗ Weak"),
         f"Drift-controlled {M.get('n_conditions')}-way accuracy: online {acc:.3f}, KS2.5 {M.get('acc_baselined_ks25', float('nan')):.3f} (chance {ch:.3f}); high charge {M.get('acc_baselined_high_online', float('nan')):.2f}. Without baseline referencing: {M.get('acc_raw_ks25', float('nan')):.3f}; baseline window alone: {M.get('acc_baseline_only_ks25', float('nan')):.3f} (session-drift confound; per-run PC2 vs run r = {M.get('drift_pc2_vs_run_r_raw', float('nan')):.2f} raw, {M.get('drift_pc2_vs_run_r_baselined', float('nan')):.2f} referenced). Charge sets condition distance (ρ = {M.get('rho_D_vs_dlogcharge_ks25', float('nan')):.2f}); sign ρ = {M.get('rho_D_vs_sign_ks25', float('nan')):+.2f}, montage ρ = {M.get('rho_D_vs_montage_ks25', float('nan')):+.2f}."),
        ("9", "The full pipeline runs live", verdict(bool(D) and D.get("match_vs_pull_min", 0) >= 0.999 and D.get("latency_ms_p99", 9e9) < 200),
         (f"Real-time push replay of runs {D.get('runs')} ({D.get('n_trials')} trials, {D.get('wall_s', 0):.0f} s): spikes identical to pull mode on {'all' if D.get('match_vs_pull_min', 0) >= 0.999 else 'not all'} trials (min Jaccard {D.get('match_vs_pull_min', float('nan')):.3f}); RTF {D.get('rtf', float('nan')):.2f}; max lag {D.get('behind_ms_max', float('nan')):.0f} ms." if D else "not run")),
    ]
    cb = M.get("cross_block")
    out = [f"# streamsort — {ctx.REC} probe {ctx.PROBE} review report", "",
           f"**Recording:** {ctx.REC}, Neuropixels probe {ctx.PROBE}, channels 0–{ctx.NCH - 1} (the lab KS2.5 channel map), {S.get('n_trials')} single biphasic pulses, {S.get('n_runs')} runs, {S.get('n_conditions_with_contacts', S.get('n_conditions'))} conditions (amplitude × montage × contacts).",
           f"**Artifact on this probe:** clipping on a median {S.get('artifact', {}).get('median_clipped_ch', float('nan')):.0f} of {ctx.NCH} channels per pulse (span {S.get('artifact', {}).get('median_span', float('nan')):.1f} samples).",
           f"**Reference:** offline Kilosort 2.5 units in the NWB file ({S.get('ref_units', {}).get('n')} on this probe, {S.get('ref_units', {}).get('good')} good). **Machine:** Apple M1 dev machine.",
           "**Method:** identical to the Phase 1 pipeline (no parameter retuning). The same verdict thresholds are applied automatically.", "",
           "## 1. Verdicts", "", "| # | Claim | Verdict | Evidence |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    if cb:
        out += ["", "## 2. Is geometry about the stimulus or about elapsed time?", "",
                f"{len(cb['conditions'])} conditions were presented in two separate blocks. Condition means fitted on one block classify the other block's trials at **{cb['online']['cross_block_acc']:.2f}** (online) / **{cb['ks25']['cross_block_acc']:.2f}** (KS2.5), against {cb['online']['within_block_acc']:.2f} / {cb['ks25']['within_block_acc']:.2f} within a block (chance {cb['chance']:.2f}).",
                f"Procrustes distance between a condition's two blocks is {cb['procrustes_same_cond_across_blocks_median']:.2f}, against {cb['procrustes_diff_cond_same_block_median']:.2f} between different conditions in the first block.",
                "So cross-block accuracy measures how much of the single-trial geometry survives a change of time while the stimulus is held fixed."]
    figs = [("validation_summary.png", "Online decoder vs KS2.5"), ("near_pulse_diagnosis.png", "Near-pulse diagnosis"), ("initial_sort_gate.png", "Initial sort vs KS2.5"),
            ("geometry_trajectories.png", "Per-trial geometry"), ("geometry_comparison.png", "Geometry across conditions"), ("trial_alignment.png", "Single-trial alignment vs trial number"),
            ("demo_live_timeline.png", "Real-time demo"), ("iti_budget.png", "Inter-trial interval budget")]
    out += ["", "## 3. Figures", ""] + [fg(pre + f, c) + "\n" for f, c in figs]
    if ITI is not None:
        out += ["## 4. Inter-trial interval budget (closed loop)", "", "| Window | Baseline | Mode | p99 after window (ms) | Min ITI (s) |", "|---|---|---|---|---|"]
        out += [f"| {r.window_s:.1f} s | {r.baseline} | {r['mode']} | {r.latency_after_window_p99_ms:.0f} | {r.min_iti_closed_loop_s:.2f} |" for _, r in ITI.iterrows()]
    out += ["", "## 5. Files", "", f"- All per-probe tables and results: {lk(f'results_{ctx.TAG}.tar.gz')}", ""]
    return "\n".join(out)

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    mk = lambda fn: (os.path.relpath(ctx.FIG / fn, "reports") if (ctx.FIG / fn).exists() else
                     os.path.relpath(fn, "reports") if os.path.exists(fn) else None)
    open(f"reports/report_{ctx.TAG}.md", "w").write(build_report(mk)); print(f"reports/report_{ctx.TAG}.md")
