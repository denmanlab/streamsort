"""Recording-level stage: pool the chosen probes into one population for geometry.

  REC=jlh32 PROBES=A,B,C python scripts/population_stage.py      (PROBES default: all finished probes)

Outputs in results/<REC>_population/: population_units.csv, geometry_* (same suite as one probe),
subset_sweep.csv, deployment_comparison.json, population_timing.csv, population_summary.json.
"""
import os, sys, json, time, itertools, tempfile, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx, geometry_step7 as g7, geometry_step8 as g8
from streamsort.memguard import MemGuard, preflight
from streamsort.io import OpenEphysSession, load_nwb_units
from streamsort.geometry import bin_trial, procrustes_distance
from scipy.stats import spearmanr

B, SM = g8.B, g8.SM

def load_probe(rec, p):
    R = lambda n: ctx.R_probe(rec, p, n)
    on = dict(np.load(R("online_spikes_all_trials_sup.npz"))); rf = dict(np.load(R("reference_spikes_all_trials_sup.npz")))
    tr = pd.read_csv(R("trials.csv")); rep = pd.read_csv(R("sorting_report.csv"))
    return dict(probe=p, on=on, rf=rf, tr=tr, fs=float(on["meta_fs"]), off=on["meta_pulse_offset"], w0=on["meta_window_start"],
                nW=int(on["meta_window_len"]), matched=np.unique(rf["ks4_to_nwb"][:, 0]), usable=rep.unit_id[rep.usable].to_numpy())

def ref_good(rec, P):
    """All KS2.5 'good' units of this probe, as spikes inside the same trial windows (probe's own clock)."""
    st = OpenEphysSession(ctx.RECS[rec]["root"] + "/Record Node 105/experiment1/recording1").stream(f"Neuropix-PXI-104.Probe{P['probe']}-AP")
    U = load_nwb_units(ctx.RECS[rec]["root"] + "/" + ctx.RECS[rec]["nwb"], probe=P["probe"]); U = U[U.group == "good"]
    w0, nW = P["w0"], P["nW"]; fr, un, tt = [], [], []
    for i, r in U.iterrows():
        s = np.sort(st.time_to_sample(np.asarray(r.spike_times)))
        a = np.searchsorted(s, w0); b = np.searchsorted(s, w0 + nW)
        for k in np.flatnonzero(b > a):
            fr.append(s[a[k]:b[k]]); un.append(np.full(b[k] - a[k], i)); tt.append(np.full(b[k] - a[k], k))
    return dict(frame=np.concatenate(fr), unit=np.concatenate(un), trial=np.concatenate(tt).astype(np.int32)), U.index.to_numpy()

def bin_pop(srcs, unitsets, Ps, baseline=True):
    g7.BASELINE = baseline
    X = [g7.bin_ref(s, u, P["off"], P["fs"], B, SM, len(P["off"])) for s, u, P in zip(srcs, unitsets, Ps)]
    g7.BASELINE = True
    return ctx.cat_units(X, axis=2)

def base_pop(srcs, unitsets, Ps):
    return np.concatenate([g7.base_rates(s, u, P["off"], P["fs"], len(P["off"])) for s, u, P in zip(srcs, unitsets, Ps)], axis=1)

def pc_sweep(Xo, Xr, cond, pou, probes, ks=(3, 6, 10, 20)):
    """Single-trial accuracy vs number of PCs, raw rates vs per-unit z-scored, for each probe and all probes."""
    from streamsort.geometry import make_reducer
    zs = lambda X: X / np.maximum(X.reshape(-1, X.shape[2]).std(0), 1e-6)[None, None, :]
    rows = []
    for sub in list(probes) + ["".join(probes)]:
        cols = np.flatnonzero(np.isin(pou, list(sub)))
        for norm in ("raw", "zscore"):
            A, Bm = (Xo[:, :, cols], Xr[:, :, cols]) if norm == "raw" else (zs(Xo[:, :, cols]), zs(Xr[:, :, cols]))
            for k in ks:
                kk = min(k, len(cols)); M = np.concatenate([Bm[cond == c].mean(0) for c in np.unique(cond)]); red = make_reducer("pca", kk).fit(M)
                r = dict(probes=sub, n_units=len(cols), norm=norm, k=k, k_used=kk)
                for nm, X in (("online", A), ("ks25", Bm)):
                    ok, _ = g7.classify(np.stack([red.transform(x) for x in X]), cond, "euclid"); r[f"acc_{nm}"] = float(ok.mean())
                rows.append(r)
    return pd.DataFrame(rows)

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "population")
    rec = ctx.REC; probes = os.environ.get("PROBES", "").replace(" ", "").split(",") if os.environ.get("PROBES") else ctx.available_probes(rec)
    print(f"[pop] {rec}: probes {probes}", flush=True)
    Ps = [load_probe(rec, p) for p in probes]
    tr = Ps[0]["tr"]
    for P in Ps[1:]:
        assert len(P["tr"]) == len(tr) and np.allclose(P["tr"].ttl_onset, tr.ttl_onset), f"trials differ between probes ({P['probe']})"
        assert (P["on"]["meta_trial_run"] == Ps[0]["on"]["meta_trial_run"]).all()
    n_tr = len(Ps[0]["off"]); run = Ps[0]["on"]["meta_trial_run"]
    cond = ctx.cond_labels(tr)[:n_tr]; q = (tr.amplitude.abs() * tr.pulse_duration).to_numpy()[:n_tr]
    units_tab = pd.concat([pd.DataFrame(dict(probe=P["probe"], ks4_unit=P["matched"], set="matched")) for P in Ps] +
                          [pd.DataFrame(dict(probe=P["probe"], ks4_unit=P["usable"], set="usable_online")) for P in Ps])
    units_tab.to_csv(ctx.RP("population_units.csv"), index=False)
    # ---- 1. matched population: full suite --------------------------------------
    on_s = [P["on"] for P in Ps]; rf_s = [P["rf"] for P in Ps]; mu = [P["matched"] for P in Ps]
    Xo = bin_pop(on_s, mu, Ps); Xr = bin_pop(rf_s, mu, Ps); Xr_raw = bin_pop(rf_s, mu, Ps, baseline=False); br = base_pop(rf_s, mu, Ps)
    probe_of_unit = np.concatenate([[P["probe"]] * len(P["matched"]) for P in Ps])
    t0 = time.perf_counter()
    summ, prod = g8.geometry_suite(Xo, Xr, Xr_raw, br, tr, cond, run, q, R=ctx.RP, tag=ctx.POP_TAG,
                                   extra=dict(units=np.arange(Xo.shape[2])))
    print(f"[pop] matched population: {Xo.shape[2]} units, suite {time.perf_counter()-t0:.0f} s | acc online {summ['acc_baselined_online']:.3f} "
          f"KS2.5 {summ['acc_baselined_ks25']:.3f} | rho {summ['rho_procrustes_online_vs_ks25']:.3f}", flush=True)
    np.save(ctx.RP("probe_of_unit.npy"), probe_of_unit)
    # ---- 2. probe-subset sweep (matched units) ------------------------------------
    rows = []; tmp = tempfile.mkdtemp()
    for k in range(1, len(probes) + 1):
        for sub in itertools.combinations(range(len(probes)), k):
            cols = np.flatnonzero(np.isin(probe_of_unit, [probes[i] for i in sub]))
            s_, _ = g8.geometry_suite(Xo[:, :, cols], Xr[:, :, cols], Xr_raw[:, :, cols], br[:, cols], tr, cond, run, q,
                                      R=lambda n: os.path.join(tmp, n), tag="sub", light=True)
            rows.append(dict(probes="".join(probes[i] for i in sub), n_probes=k, n_units=len(cols),
                             acc_online=s_["acc_baselined_online"], acc_ks25=s_["acc_baselined_ks25"],
                             acc_high_online=s_["acc_baselined_high_online"], acc_high_ks25=s_["acc_baselined_high_ks25"],
                             procrustes_same_trial=s_["procrustes_same_trial_median"], procrustes_trial_vs_mean=s_["procrustes_ks25_vs_own_mean_median"],
                             rho_online_vs_ks25=s_["rho_procrustes_online_vs_ks25"], split_half_ks25=s_["split_half_ks25"],
                             cross_half=s_["cross_half_online_even_vs_ks25_odd"],
                             cross_block_online=s_.get("cross_block", {}).get("online", {}).get("cross_block_acc", np.nan)))
            print(f"[pop] subset {rows[-1]['probes']}: {len(cols)} units, acc {rows[-1]['acc_online']:.3f}/{rows[-1]['acc_ks25']:.3f}", flush=True)
    guard.check("subsets")
    SW = pd.DataFrame(rows); SW.to_csv(ctx.RP("subset_sweep.csv"), index=False)
    PCS = pc_sweep(Xo, Xr, cond, probe_of_unit, probes); PCS.to_csv(ctx.RP("pc_count_sweep.csv"), index=False)
    print("[pop] pc sweep (online, zscore):", PCS[PCS.norm == "zscore"].pivot_table(index="probes", columns="k", values="acc_online").round(3).to_dict(), flush=True)
    # ---- 3. deployment population: all usable online units vs all good KS2.5 units ---
    us = [P["usable"] for P in Ps]; Xo_all = bin_pop(on_s, us, Ps)
    rg = [ref_good(rec, P) for P in Ps]; Xg = bin_pop([r[0] for r in rg], [r[1] for r in rg], Ps)
    dep = dict(n_online_usable=int(Xo_all.shape[2]), n_ks25_good=int(Xg.shape[2]))
    Z = {}
    for nm, X in [("online_all", Xo_all), ("ks25_good", Xg)]:
        red = g7.fixed_basis(X, cond); Z[nm] = np.stack([red.transform(x) for x in X])
        ok, _ = g7.classify(Z[nm], cond, "euclid"); dep[f"acc_{nm}"] = float(ok.mean()); dep[f"acc_high_{nm}"] = float(ok[q >= 5000].mean())
    cs = sorted(np.unique(cond)); iu = np.triu_indices(len(cs), 1)
    Dm = {nm: np.array([[procrustes_distance(Z[nm][cond == a].mean(0), Z[nm][cond == b].mean(0)) for b in cs] for a in cs]) for nm in Z}
    dep["rho_condition_geometry"] = float(spearmanr(Dm["online_all"][iu], Dm["ks25_good"][iu]).correlation)
    dep["procrustes_same_trial_median"] = float(np.median([procrustes_distance(Z["ks25_good"][i], Z["online_all"][i]) for i in range(n_tr)]))
    dep["chance"] = 1 / len(cs)
    guard.check("deployment")
    json.dump(dep, open(ctx.RP("deployment_comparison.json"), "w"), indent=2)
    print("[pop] deployment:", {k: round(v, 3) if isinstance(v, float) else v for k, v in dep.items()}, flush=True)
    # ---- 4. per-trial cost of pooled geometry (fixed basis) -------------------------
    red = g7.fixed_basis(Xr, cond); mu_c = {c: red.transform(Xr[cond == c].mean(0)) for c in np.unique(cond)}
    trows = []; rng = np.random.default_rng(0)
    idx = []   # per probe: spikes pre-split by trial, as the live tool holds only the current trial
    for P, u in zip(Ps, mu):
        on = P["on"]; m = np.isin(on["unit"], u); f, uu, t = on["frame"][m], on["unit"][m], on["trial"][m]
        o = np.argsort(t, kind="stable"); f, uu, t = f[o], uu[o], t[o]; idx.append((f, uu, np.searchsorted(t, np.arange(n_tr + 1))))
    for k in rng.choice(n_tr, 200, replace=False):
        t0 = time.perf_counter(); xs = []
        for P, u, (f, uu, bd) in zip(Ps, mu, idx):
            sl = slice(bd[k], bd[k + 1])
            xs.append(bin_trial(f[sl], uu[sl], u, P["off"][k] + int(round(g7.WIN0_MS * 1e-3 * P["fs"])),
                                int(round(g7.WIN_MS * 1e-3 * P["fs"])), P["fs"], bin_ms=B, smooth_ms=SM))
        t1 = time.perf_counter(); z = red.transform(ctx.cat_units(xs, axis=1) / (B * 1e-3)); procrustes_distance(mu_c[cond[k]], z); t2 = time.perf_counter()
        trows.append(dict(trial=int(k), bin_ms=(t1 - t0) * 1e3, project_metric_ms=(t2 - t1) * 1e3, total_ms=(t2 - t0) * 1e3))
    T = pd.DataFrame(trows); T.to_csv(ctx.RP("population_timing.csv"), index=False)
    out = dict(rec=rec, probes=probes, n_units_matched=int(Xo.shape[2]), units_per_probe={P["probe"]: int(len(P["matched"])) for P in Ps},
               usable_per_probe={P["probe"]: int(len(P["usable"])) for P in Ps},
               geometry_ms_p99=float(T.total_ms.quantile(.99)), geometry_ms_median=float(T.total_ms.median()), deployment=dep,
               best_single_probe_acc=float(SW[SW.n_probes == 1].acc_online.max()), all_probes_acc=float(SW[SW.n_probes == len(probes)].acc_online.iloc[0]))
    json.dump(out, open(ctx.RP("population_summary.json"), "w"), indent=2)
    print("[pop] DONE", json.dumps(out)[:600], flush=True)

if __name__ == "__main__":
    main()
