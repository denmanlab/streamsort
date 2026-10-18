"""Steps 7b-8: chosen-parameter geometry, drift checks, cross-condition comparison, trial alignment.

Inputs : online/reference spikes (ctx.R), trials.csv
Outputs: geometry_per_trial_sup.npz, geometry_trial_distances.csv, geometry_distances.csv,
         trial_alignment.csv, trial_alignment_trends.csv, geometry_comparison_summary.json, geometry_mds.npz
"""
import os, sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
warnings.filterwarnings("ignore", category=FutureWarning)
import ctx, geometry_step7 as g7
from scipy.stats import spearmanr, linregress
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import MDS
from streamsort.geometry import procrustes_distance, projection_score

B, SM, K = 10, 20, 3

def top_basis(M, k=3):
    return np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:k].T

def main():
    on, rf, tr, units = g7.load()
    fs = float(on["meta_fs"]); off = on["meta_pulse_offset"]; n_tr = len(off); run = on["meta_trial_run"]
    cond = ctx.cond_labels(tr)[:n_tr]; q = (tr.amplitude.abs() * tr.pulse_duration).to_numpy()[:n_tr]
    g7.BASELINE = False
    Xr_raw = g7.bin_ref(rf, units, off, fs, B, SM, n_tr)
    br = g7.base_rates(rf, units, off, fs, n_tr)
    g7.BASELINE = True
    Xo = g7.bin_ref(on, units, off, fs, B, SM, n_tr); Xr = g7.bin_ref(rf, units, off, fs, B, SM, n_tr)
    geometry_suite(Xo, Xr, Xr_raw, br, tr, cond, run, q, R=ctx.R, tag=ctx.TAG, extra=dict(units=units))


def geometry_suite(Xo, Xr, Xr_raw, br, tr, cond, run, q, R, tag, extra=None, light=False):
    """Steps 7b-8 on any population (one probe or several concatenated along units).

    Xo, Xr : (trials, bins, units) baseline-referenced online / KS2.5 rates (same units, same order)
    Xr_raw : KS2.5 rates without baseline referencing (drift-confound check); br: KS2.5 baseline rates
    R      : callable name -> output path.   light=True skips trial-alignment (for subset sweeps).
    """
    n_tr = len(cond); units = (extra or {}).get("units", np.arange(Xo.shape[2]))
    cs = np.unique(cond); N = {c: (cond == c).sum() for c in cs}
    summ = dict(tag=tag, n_units=int(Xo.shape[2]), n_trials=int(n_tr), n_conditions=int(len(cs)),
                params=dict(window_ms=[g7.WIN0_MS, g7.WIN0_MS + g7.WIN_MS], bin_ms=B, smooth_sigma_ms=SM, baseline_s=list(g7.BASE_S), k=K))
    red_raw = g7.fixed_basis(Xr_raw, cond); Zraw = np.stack([red_raw.transform(x) for x in Xr_raw])
    summ["drift_pc2_vs_run_r_raw"] = float(np.corrcoef(np.unique(run), [Zraw[run == r][:, :, 1].mean() for r in np.unique(run)])[0, 1])
    ok, _ = g7.classify(Zraw, cond, "euclid"); summ["acc_raw_ks25"] = float(ok.mean())
    ok, _ = g7.classify(((br - red_raw.mean_) @ red_raw.basis)[:, None, :], cond, "euclid"); summ["acc_baseline_only_ks25"] = float(ok.mean())
    del Zraw
    red = g7.fixed_basis(Xr, cond)
    Zo = np.stack([red.transform(x) for x in Xo]); Zr = np.stack([red.transform(x) for x in Xr])
    summ["evr"] = [float(v) for v in getattr(red, "explained_variance_ratio_", [])]
    summ["drift_pc2_vs_run_r_baselined"] = float(np.corrcoef(np.unique(run), [Zr[run == r][:, :, 1].mean() for r in np.unique(run)])[0, 1])
    for nm, Z in [("online", Zo), ("ks25", Zr)]:
        ok, _ = g7.classify(Z, cond, "euclid"); summ[f"acc_baselined_{nm}"] = float(ok.mean()); summ[f"acc_baselined_high_{nm}"] = float(ok[q >= 5000].mean())
    summ["chance"] = 1 / len(cs)
    Sr = {c: Zr[cond == c].sum(0) for c in cs}; So = {c: Zo[cond == c].sum(0) for c in cs}
    d_same = np.array([procrustes_distance(Zr[i], Zo[i]) for i in range(n_tr)])
    d_ref_own = np.array([procrustes_distance((Sr[cond[i]] - Zr[i]) / (N[cond[i]] - 1), Zr[i]) for i in range(n_tr)])
    d_on_own = np.array([procrustes_distance((So[cond[i]] - Zo[i]) / (N[cond[i]] - 1), Zo[i]) for i in range(n_tr)])
    summ.update(procrustes_same_trial_median=float(np.median(d_same)), procrustes_ks25_vs_own_mean_median=float(np.median(d_ref_own)),
                procrustes_online_vs_own_mean_median=float(np.median(d_on_own)))
    np.savez_compressed(R("geometry_per_trial_sup.npz"), Z_online=Zo.astype(np.float32), Z_ks25=Zr.astype(np.float32), basis=red.basis,
                        mean=red.mean_, cond=cond, q=q, run=run, bin_ms=B, smooth_ms=SM, units=units, d_same=d_same,
                        d_ref_own=d_ref_own, d_on_own=d_on_own)
    pd.DataFrame(dict(trial=np.arange(n_tr), cond=cond, run=run, procrustes_online_vs_ks25=d_same,
                      procrustes_ks25_vs_own_condmean=d_ref_own, procrustes_online_vs_own_condmean=d_on_own)).to_csv(R("geometry_trial_distances.csv"), index=False)
    # ---- condition x condition ----------------------------------------------
    first = tr.groupby(ctx.cond_labels(tr)).first()
    ci = pd.DataFrame(dict(cond=cs, amp=first.loc[cs].amplitude.astype(int).values, mont=first.loc[cs].polarity.astype(str).values,
                           dur=first.loc[cs].pulse_duration.astype(int).values, cneg=first.loc[cs].contact_negative.astype(int).values))
    ci["charge"] = ci.amp.abs() * ci.dur; ci = ci.sort_values(["mont", "cneg", "charge", "amp"]).reset_index(drop=True); order = ci.cond.tolist()
    def mats(S):
        M = {c: S[c] / N[c] for c in cs}
        D = np.array([[procrustes_distance(M[a], M[b]) for b in order] for a in order]); D = (D + D.T) / 2
        E = np.array([[np.linalg.norm(M[a] - M[b]) for b in order] for a in order]); return D, E
    Do, Eo = mats(So); Dr, Er = mats(Sr); n = len(order); iu = np.triu_indices(n, 1)
    rho = spearmanr(Do[iu], Dr[iu]).correlation; rng = np.random.default_rng(0)
    null = [spearmanr(Do[iu], Dr[p][:, p][iu]).correlation for p in (rng.permutation(n) for _ in range(2000))]
    def half(Z, par):
        M = {c: Z[(cond == c) & (np.arange(n_tr) % 2 == par)].mean(0) for c in cs}
        D = np.array([[procrustes_distance(M[a], M[b]) for b in order] for a in order]); return (D + D.T) / 2
    Dh0, Dh1, Dho0, Dho1 = half(Zr, 0), half(Zr, 1), half(Zo, 0), half(Zo, 1)
    lc = np.log10(ci.charge.values); dq = np.abs(lc[:, None] - lc[None]); sg = np.sign(ci.amp.values); mo = ci.mont.values.astype(object)
    summ.update(rho_procrustes_online_vs_ks25=float(rho), mantel_p=float((np.sum(np.array(null) >= rho) + 1) / 2001),
                rho_euclid_online_vs_ks25=float(spearmanr(Eo[iu], Er[iu]).correlation),
                split_half_ks25=float(spearmanr(Dh0[iu], Dh1[iu]).correlation), split_half_online=float(spearmanr(Dho0[iu], Dho1[iu]).correlation),
                cross_half_online_even_vs_ks25_odd=float(spearmanr(Dho0[iu], Dh1[iu]).correlation),
                rho_D_vs_dlogcharge_ks25=float(spearmanr(dq[iu], Dr[iu]).correlation),
                rho_D_vs_sign_ks25=float(spearmanr((sg[:, None] != sg[None])[iu].astype(float), Dr[iu]).correlation),
                rho_D_vs_montage_ks25=float(spearmanr((mo[:, None] != mo[None])[iu].astype(float), Dr[iu]).correlation))
    rows = [dict(cond_a=a, cond_b=b, procrustes_online=Do[i, j], procrustes_ks25=Dr[i, j], euclid_online=Eo[i, j], euclid_ks25=Er[i, j])
            for i, a in enumerate(order) for j, b in enumerate(order) if j > i]
    pd.DataFrame(rows).to_csv(R("geometry_distances.csv"), index=False)
    Yr = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8, normalized_stress="auto").fit_transform(Dr)
    Yo = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8, normalized_stress="auto").fit_transform(Do)
    Yr -= Yr.mean(0); Yo -= Yo.mean(0); Rm, _ = orthogonal_procrustes(Yo, Yr); Yo = Yo @ Rm
    np.savez_compressed(R("geometry_mds.npz"), order=np.array(order), Do=Do, Dr=Dr, Yr=Yr, Yo=Yo, amp=ci.amp.values, mont=ci.mont.values.astype(str),
                        dur=ci.dur.values, cneg=ci.cneg.values, charge=ci.charge.values)
    # ---- cross-block test (conditions presented in >1 run) ----------------------
    rep_c = [c for c in cs if len(np.unique(run[cond == c])) >= 2]
    summ["n_conditions_repeated_blocks"] = len(rep_c)
    if len(rep_c) >= 2:
        blocks = {c: np.unique(run[cond == c])[:2] for c in rep_c}
        res = {}
        for nm, Z in [("online", Zo), ("ks25", Zr)]:
            within, cross = [], []
            for bi in (0, 1):
                Mtr = {cc: Z[(cond == cc) & (run == blocks[cc][1 - bi])].mean(0) for cc in rep_c}     # other block
                Sw = {cc: Z[(cond == cc) & (run == blocks[cc][bi])].sum(0) for cc in rep_c}           # same block
                Nw = {cc: ((cond == cc) & (run == blocks[cc][bi])).sum() for cc in rep_c}
                for c in rep_c:
                    for i in np.flatnonzero((cond == c) & (run == blocks[c][bi])):
                        dx = {cc: np.linalg.norm(Mtr[cc] - Z[i]) for cc in rep_c}; cross.append(min(dx, key=dx.get) == c)
                        dw = {cc: np.linalg.norm(((Sw[cc] - Z[i]) / (Nw[cc] - 1) if cc == c else Sw[cc] / Nw[cc]) - Z[i]) for cc in rep_c}
                        within.append(min(dw, key=dw.get) == c)
            res[nm] = dict(cross_block_acc=float(np.mean(cross)), within_block_acc=float(np.mean(within)), n_test=len(cross))
        Mb = {(c, k): Zr[(cond == c) & (run == blocks[c][k])].mean(0) for c in rep_c for k in (0, 1)}
        same = [procrustes_distance(Mb[(c, 0)], Mb[(c, 1)]) for c in rep_c]
        diff = [procrustes_distance(Mb[(a_, 0)], Mb[(b_, 0)]) for i_, a_ in enumerate(rep_c) for b_ in rep_c[i_ + 1:]]
        summ["cross_block"] = dict(conditions=rep_c, chance=1 / len(rep_c), **res,
                                   procrustes_same_cond_across_blocks_median=float(np.median(same)),
                                   procrustes_diff_cond_same_block_median=float(np.median(diff)))
    if light:
        return summ, dict(Zo=Zo, Zr=Zr, Do=Do, Dr=Dr, order=order)
    # ---- trial alignment vs repetition -------------------------------------------
    rows = []
    for nm, X, Z in [("online", Xo, Zo), ("ks25", Xr, Zr)]:
        Xd = X.astype(np.float64); Mc = {c: Xd[cond == c].mean(0) for c in cs}; Bc = {c: top_basis(Mc[c]) for c in cs}
        for c in cs:
            idx = np.flatnonzero(cond == c); n_c = len(idx); Ssum = Xd[idx].sum(0); Zsum = Z[idx].sum(0); others = [o for o in cs if o != c]
            for r, i in enumerate(idx):
                x = Xd[i]; Ml = (Ssum - x) / (n_c - 1); g_l = (x * Ml).sum() / (Ml * Ml).sum()
                ps_l = projection_score(x, top_basis(Ml)); ps_null = np.mean([projection_score(x, Bc[o]) for o in others])
                d_l = procrustes_distance((Zsum - Z[i]) / (n_c - 1), Z[i])
                if r >= 5:
                    Mp = Xd[idx[:r]].mean(0); g_p = (x * Mp).sum() / (Mp * Mp).sum(); ps_p = projection_score(x, top_basis(Mp))
                    d_p = procrustes_distance(Z[idx[:r]].mean(0), Z[i])
                else:
                    g_p = ps_p = d_p = np.nan
                rows.append(dict(sort=nm, trial=int(i), cond=c, rep=r, run=int(run[i]), charge=float(q[i]), gain_loo=g_l, gain_past=g_p,
                                 projscore_loo=ps_l, projscore_past=ps_p, projscore_null_othercond=ps_null, procrustes_loo=d_l, procrustes_past=d_p))
    TA = pd.DataFrame(rows); TA.to_csv(R("trial_alignment.csv"), index=False)
    tr_rows = []
    for (nm, c), d in TA.groupby(["sort", "cond"]):
        for met in ["gain_loo", "projscore_loo", "procrustes_loo"]:
            lr = linregress(d.rep, d[met]); tr_rows.append(dict(sort=nm, cond=c, charge=d.charge.iloc[0], metric=met, slope_per10=lr.slope * 10, p=lr.pvalue, mean=d[met].mean()))
    TT = pd.DataFrame(tr_rows); TT.to_csv(R("trial_alignment_trends.csv"), index=False)
    m = TA.pivot_table(index="trial", columns="sort", values="gain_loo"); g = TT[TT.metric == "gain_loo"]
    tg = g.pivot_table(index="cond", columns="sort", values="slope_per10")
    summ.update(gain_corr_per_trial_online_vs_ks25=float(m.corr().iloc[0, 1]), gain_slope_corr_conditions=float(tg.corr().iloc[0, 1]),
                n_cond_gain_trend_p05_online=int((g[g.sort == "online"].p < 0.05).sum()), n_cond_gain_trend_p05_ks25=int((g[g.sort == "ks25"].p < 0.05).sum()),
                mean_gain_high_charge_online=float(TA[(TA.sort == "online") & (TA.charge >= 5000)].gain_loo.mean()),
                mean_gain_high_charge_ks25=float(TA[(TA.sort == "ks25") & (TA.charge >= 5000)].gain_loo.mean()))
    json.dump(summ, open(R("geometry_comparison_summary.json"), "w"), indent=2, default=float)
    return summ, dict(Zo=Zo, Zr=Zr, Do=Do, Dr=Dr, order=order)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k not in ("params", "evr")}, default=str), flush=True)

if __name__ == "__main__":
    main()
