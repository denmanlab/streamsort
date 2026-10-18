"""Phase 1 step 7: per-trial population geometry, swappable reducers, timing.

Inputs : results/online_spikes_all_trials{SUF}.npz, reference_spikes_all_trials{SUF}.npz
Units  : the matched usable units (online KS4 id <-> NWB KS2.5), so both sorts
         describe the same neurons.
Window : pulse offset + 2 ms .. + 502 ms (excitation 2-10, suppression 20-100,
         rebound 100-300 ms; 1-2 ms excluded as artifact-mixed).
Outputs: results/geometry_sweep.csv   condition information per bin/smoothing
         results/geometry_timing.csv  per-trial compute cost by operation
         results/geometry_per_trial.npz  trajectories for steps 8-9 (chosen params)
"""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from streamsort.geometry import bin_trial, make_reducer, procrustes_distance, projection_score
from streamsort.memguard import MemGuard, preflight
sys.path.insert(0, os.path.join(os.getcwd(), "scripts")); import ctx

SUF = os.environ.get("SUF", "_sup")
WIN0_MS, WIN_MS = 2.0, 500.0
BINS = [5, 10, 25, 50]; SMOOTH_X = [0, 1, 2]      # smoothing sigma = factor x bin width
K = 3
BASE_S = (1.0, 1.9)      # same-chunk baseline window (s after pulse offset); removes slow session drift
BASELINE = os.environ.get("BASELINE", "1") == "1"

def load():
    on = dict(np.load(ctx.R(f"online_spikes_all_trials{SUF}.npz")))
    rf = dict(np.load(ctx.R(f"reference_spikes_all_trials{SUF}.npz")))
    tr = pd.read_csv(ctx.R("trials.csv"))
    units = np.unique(rf["ks4_to_nwb"][:, 0])
    return on, rf, tr, units

def bin_all(src, units, off, fs, bin_ms, smooth_ms, n_tr):
    s0 = off + int(round(WIN0_MS * 1e-3 * fs)); n = int(round(WIN_MS * 1e-3 * fs))
    m = np.isin(src["unit"], units); f, u, t = src["frame"][m], src["unit"][m], src["trial"][m]
    order = np.argsort(t, kind="stable"); f, u, t = f[order], u[order], t[order]
    bnd = np.searchsorted(t, np.arange(n_tr + 1))
    X = np.stack([bin_trial(f[bnd[k]:bnd[k + 1]], u[bnd[k]:bnd[k + 1]], units, s0[k], n, fs,
                            bin_ms=bin_ms, smooth_ms=smooth_ms) for k in range(n_tr)]).astype(np.float32)
    return X / (bin_ms * 1e-3)                    # Hz

def base_rates(src, units, off, fs, n_tr):
    """Per-trial baseline rate (Hz) per unit from the late part of the same chunk."""
    m = np.isin(src["unit"], units); f, u, t = src["frame"][m], src["unit"][m], src["trial"][m]
    lat = (f - off[t]) / fs; k = (lat >= BASE_S[0]) & (lat < BASE_S[1])
    M = np.zeros((n_tr, len(units)), np.float32); np.add.at(M, (t[k], np.searchsorted(units, u[k])), 1)
    return M / (BASE_S[1] - BASE_S[0])

def bin_ref(src, units, off, fs, b, sm, n_tr):
    X = bin_all(src, units, off, fs, b, sm, n_tr)
    return X - base_rates(src, units, off, fs, n_tr)[:, None, :] if BASELINE else X

def fixed_basis(X, cond):
    """PCA on condition-averaged trajectories (all conditions stacked)."""
    M = np.concatenate([X[cond == c].mean(0) for c in np.unique(cond)])
    r = make_reducer("pca", K).fit(M); return r

def loo_means(X, cond):
    S = {c: X[cond == c].sum(0) for c in np.unique(cond)}; N = {c: (cond == c).sum() for c in S}
    return S, N

def classify(Z, cond, metric):
    """Nearest leave-one-out condition mean.  Z: (n_trials, n_bins, K)."""
    cs = np.unique(cond); S = {c: Z[cond == c].sum(0) for c in cs}; N = {c: (cond == c).sum() for c in cs}
    correct = np.zeros(len(Z), bool); dist_own = np.zeros(len(Z))
    for i in range(len(Z)):
        d = []
        for c in cs:
            mu = (S[c] - Z[i]) / (N[c] - 1) if c == cond[i] else S[c] / N[c]
            d.append(procrustes_distance(mu, Z[i]) if metric == "procrustes" else np.linalg.norm(mu - Z[i]))
        d = np.array(d); correct[i] = cs[np.argmin(d)] == cond[i]; dist_own[i] = d[list(cs).index(cond[i])]
    return correct, dist_own

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "geometry7")
    on, rf, tr, units = load()
    fs = float(on["meta_fs"]); off = on["meta_pulse_offset"]; n_tr = len(off)
    cond = ctx.cond_labels(tr)[:n_tr]
    q = (tr.amplitude.abs() * tr.pulse_duration).to_numpy()[:n_tr]
    rows = []; keep = {}
    for b in BINS:
        for sx in SMOOTH_X:
            sm = sx * b
            Xs = {nm: bin_ref(src, units, off, fs, b, sm, n_tr) for nm, src in [("online", on), ("ks25", rf)]}
            red = fixed_basis(Xs["ks25"], cond)             # common basis -> comparable trajectories
            Zs = {nm: np.stack([red.transform(x) for x in X]) for nm, X in Xs.items()}
            for nm, Z in Zs.items():
                for metric in ["euclid", "procrustes"]:
                    ok, d = classify(Z, cond, metric)
                    rows.append(dict(bin_ms=b, smooth_ms=sm, sort=nm, metric=metric, acc26=ok.mean(),
                                     acc26_highq=ok[q >= 5000].mean(), chance=1 / len(np.unique(cond)),
                                     dist_own_med=np.median(d)))
            # online-vs-reference distortion, same trial, same basis
            pd_ = np.array([procrustes_distance(Zs["ks25"][i], Zs["online"][i]) for i in range(n_tr)])
            rows.append(dict(bin_ms=b, smooth_ms=sm, sort="online_vs_ks25", metric="procrustes_same_trial",
                             dist_own_med=np.median(pd_), acc26=np.nan))
            keep[(b, sm)] = (Zs, red)
            guard.check(f"bin {b} sm {sm}")
            print(f"[geo] bin {b} sm {sm}: " + " | ".join(f"{r['sort']}/{r['metric']} {r['acc26']:.3f}" for r in rows[-5:-1])
                  + f" | online-vs-ref median Procrustes {np.median(pd_):.3f} | {guard.report()}", flush=True)
            del Xs
    sw = pd.DataFrame(rows); sw.to_csv(ctx.R(f"geometry_sweep{SUF}.csv"), index=False)
    # chosen parameters: best mean(online, ks25) Euclidean accuracy
    e = sw[sw.metric == "euclid"].groupby(["bin_ms", "smooth_ms"]).acc26.mean()
    b_best, sm_best = e.idxmax(); print("[geo] chosen bin/smooth:", b_best, sm_best, round(e.max(), 3))
    Zs, red = keep[(b_best, sm_best)]
    np.savez_compressed(ctx.R(f"geometry_per_trial_auto{SUF}.npz"), Z_online=Zs["online"], Z_ks25=Zs["ks25"],
                        basis=red.basis, mean=red.mean_ if hasattr(red, "mean_") else np.zeros(1),
                        cond=cond, q=q, bin_ms=b_best, smooth_ms=sm_best, units=units)
    # ---- timing: one trial at a time, as the live tool would ------------------
    trows = []
    rng = np.random.default_rng(0); idx = rng.choice(n_tr, 200, replace=False)
    m = np.isin(on["unit"], units)
    f_all, u_all, t_all = on["frame"][m], on["unit"][m], on["trial"][m]
    order = np.argsort(t_all, kind="stable"); f_all, u_all, t_all = f_all[order], u_all[order], t_all[order]
    bnd = np.searchsorted(t_all, np.arange(n_tr + 1))
    for b in BINS:
        for sx in [0, 1]:
            sm = sx * b
            Xref = bin_ref(rf, units, off, fs, b, sm, n_tr); red = fixed_basis(Xref, cond)
            mu_c = {c: red.transform(Xref[cond == c].mean(0)) for c in np.unique(cond)}
            V = red.basis; del Xref
            for name in ["bin", "fixed_project", "procrustes_vs_condmean", "projection_score",
                         "refit_pca", "refit_fa", "refit_isomap", "total_fixed"]:
                ts = []
                for k in idx:
                    sl = slice(bnd[k], bnd[k + 1]); s0 = off[k] + int(round(WIN0_MS * 1e-3 * fs)); n = int(round(WIN_MS * 1e-3 * fs))
                    t0 = time.perf_counter()
                    if name in ("bin", "total_fixed"):
                        x = bin_trial(f_all[sl], u_all[sl], units, s0, n, fs, bin_ms=b, smooth_ms=sm)
                    else:
                        x = bin_trial(f_all[sl], u_all[sl], units, s0, n, fs, bin_ms=b, smooth_ms=sm); t0 = time.perf_counter()
                    if name in ("fixed_project", "total_fixed"):
                        z = red.transform(x)
                    if name == "total_fixed":
                        procrustes_distance(mu_c[cond[k]], z); projection_score(x, V)
                    if name == "procrustes_vs_condmean":
                        z = red.transform(x); t0 = time.perf_counter(); procrustes_distance(mu_c[cond[k]], z)
                    if name == "projection_score":
                        projection_score(x, V)
                    if name.startswith("refit_"):
                        try:
                            make_reducer(name.split("_")[1], K).fit_transform(x)
                        except Exception:
                            pass
                    ts.append(time.perf_counter() - t0)
                ts = np.array(ts) * 1e3
                trows.append(dict(bin_ms=b, smooth_ms=sm, n_bins=int(WIN_MS / b), operation=name, median_ms=np.median(ts),
                                  p99_ms=np.percentile(ts, 99), max_ms=ts.max()))
            print(f"[geo] timing bin {b} sm {sm} done", flush=True)
    pd.DataFrame(trows).to_csv(ctx.R("geometry_timing.csv"), index=False)
    print(pd.DataFrame(trows).pivot_table(index="operation", columns=["bin_ms", "smooth_ms"], values="p99_ms").round(2).to_string())

if __name__ == "__main__":
    main()
