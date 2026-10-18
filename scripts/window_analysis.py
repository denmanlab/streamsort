"""Recording-level: which part of the post-pulse chunk carries condition information, and which baseline to use.

Pooled KS2.5-matched units (all probes), online and KS2.5 spikes, 10 ms bins (sigma 20 ms) from 2 to 1892 ms after pulse offset.
Baselines (per trial, per unit, Hz):
  same : 1.0-1.9 s of the same chunk (current default; only valid for response windows ending <= 1.0 s)
  pre  : 1.4-1.9 s of the PREVIOUS trial's chunk = ~0.6-0.1 s before this pulse (what a live pre-pulse baseline would see);
         trials whose previous trial is in another run are excluded from ALL comparisons
  none : no baseline (slow session drift is then confounded with the blocked conditions)
Response energy: per condition and 10 ms bin, ||condition mean - pre-pulse baseline||^2 summed over units, minus the noise floor
  (sum over units of trial variance / n trials), in unit space and in a PCA space fitted on 2-1002 ms condition means.
Window sweep: response window 2 ms .. W for W = 0.1-1.5 s (pre-pulse baseline; same-chunk baseline while W <= 1.0 s) -> window_sweep.csv.
Loss timing: Procrustes / Euclidean distance of each trial's 2-502 ms trajectory to its condition target -> window_loss_timing.json.
Outputs: window_energy.csv, window_analysis.csv (response window x baseline x k), window_timecourse.csv (100 ms segments), window_psth.npz
  REC=jlh32 [PROBES=A,B,C] python scripts/window_analysis.py
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx, geometry_step7 as g7
from population_stage import load_probe
from streamsort.geometry import bin_trial, make_reducer, procrustes_distance
from streamsort.memguard import MemGuard, preflight

B_MS, SM_MS, T0_MS, T1_MS = 10, 20, 2.0, 1892.0

def bin_full(src, units, off, fs, n_tr):
    s0 = off + int(round(T0_MS * 1e-3 * fs)); n = int(round((T1_MS - T0_MS) * 1e-3 * fs))
    m = np.isin(src["unit"], units); f, u, t = src["frame"][m], src["unit"][m], src["trial"][m]
    o = np.argsort(t, kind="stable"); f, u, t = f[o], u[o], t[o]; bnd = np.searchsorted(t, np.arange(n_tr + 1))
    return np.stack([bin_trial(f[bnd[k]:bnd[k + 1]], u[bnd[k]:bnd[k + 1]], units, s0[k], n, fs, bin_ms=B_MS, smooth_ms=SM_MS)
                     for k in range(n_tr)]).astype(np.float32) / (B_MS * 1e-3)

def late_rate(src, units, off, fs, n_tr, a, b):
    m = np.isin(src["unit"], units); f, u, t = src["frame"][m], src["unit"][m], src["trial"][m]
    lat = (f - off[t]) / fs; k = (lat >= a) & (lat < b)
    M = np.zeros((n_tr, len(units)), np.float32); np.add.at(M, (t[k], np.searchsorted(units, u[k])), 1); return M / (b - a)

def cond_geometry_rho(Zo, Zr, cond):
    cs = np.unique(cond); Mo = [Zo[cond == c].mean(0) for c in cs]; Mr = [Zr[cond == c].mean(0) for c in cs]
    iu = np.triu_indices(len(cs), 1)
    Do = np.array([[procrustes_distance(a, b) for b in Mo] for a in Mo]); Dr = np.array([[procrustes_distance(a, b) for b in Mr] for a in Mr])
    return float(spearmanr(Do[iu], Dr[iu])[0])

def evaluate(Xo, Xr, bo, br, sel, cond, k):
    """Xo/Xr: (trials, bins, units) raw rates; bo/br: (trials, units) baselines; sel: bin mask. No large copies."""
    cs = np.unique(cond)
    M = np.concatenate([(Xr[cond == c][:, sel] - br[cond == c][:, None, :]).mean(0) for c in cs]); red = make_reducer("pca", k).fit(M)
    Zo = np.stack([red.transform(Xo[i][sel] - bo[i]) for i in range(len(Xo))]); Zr = np.stack([red.transform(Xr[i][sel] - br[i]) for i in range(len(Xr))])
    return float(g7.classify(Zo, cond, "euclid")[0].mean()), float(g7.classify(Zr, cond, "euclid")[0].mean()), Zo, Zr

def energy(Xv, bpre, cond, t, red=None):
    """Noise-corrected response energy per condition and bin (pre-pulse baseline)."""
    rows = []
    for c in np.unique(cond):
        Y = Xv[cond == c] - bpre[cond == c][:, None, :]
        if red is not None: Y = Y @ red.components_.T.astype(np.float32)   # baseline stays at the origin (no PCA centring)
        n = len(Y); m = Y.mean(0); E = (m ** 2).sum(1); N = (Y.var(0, ddof=1) / n).sum(1)
        rows.append(pd.DataFrame(dict(cond=c, t_ms=t, energy=E, noise=N, signal=E - N, n_trials=n)))
    return pd.concat(rows, ignore_index=True)

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "window_analysis")
    rec = ctx.REC; probes = os.environ["PROBES"].split(",") if os.environ.get("PROBES") else ctx.available_probes(rec)
    Ps = [load_probe(rec, p) for p in probes]; tr = Ps[0]["tr"]; n_tr = len(Ps[0]["off"]); run = Ps[0]["on"]["meta_trial_run"]
    cond = np.asarray(ctx.cond_labels(tr)[:n_tr])
    X = {}; base = {}; NB = int((T1_MS - T0_MS) // B_MS); valid = np.r_[False, run[1:] == run[:-1]]
    for nm, key in (("online", "on"), ("ks25", "rf")):
        parts = [bin_full(P[key], P["matched"], P["off"], P["fs"], n_tr)[valid] for P in Ps]; NB = min(NB, min(x.shape[1] for x in parts))
        X[nm] = np.concatenate([x[:, :NB] for x in parts], axis=2); del parts
        same = np.concatenate([late_rate(P[key], P["matched"], P["off"], P["fs"], n_tr, 1.0, 1.9) for P in Ps], axis=1)
        tail = np.concatenate([late_rate(P[key], P["matched"], P["off"], P["fs"], n_tr, 1.4, 1.9) for P in Ps], axis=1)
        pre = np.full_like(tail, np.nan); pre[1:] = tail[:-1]
        base[nm] = {k_: v[valid] for k_, v in dict(same=same, pre=pre, none=np.zeros_like(same)).items()}; guard.check(f"binned {nm}")
    X = {k_: v[:, :NB] for k_, v in X.items()}
    print(f"[win] {rec} {probes}: {X['online'].shape} ; {valid.sum()}/{n_tr} trials with a same-run previous trial", flush=True)
    t = T0_MS + B_MS * (np.arange(NB) + 0.5); cv = cond[valid]
    seg = lambda a, b: (t >= a) & (t < b)
    # ---- response energy over the whole chunk (pre-pulse baseline): unit space and PCA(2-1002 ms) space
    EN = []
    for nm in ("online", "ks25"):
        EN.append(energy(X[nm], base[nm]["pre"], cv, t).assign(sort=nm, space="units"))
        for k in (3, 10):
            sel = seg(2, 1002); M = np.concatenate([(X["ks25"][cv == c][:, sel] - base["ks25"]["pre"][cv == c][:, None, :]).mean(0) for c in np.unique(cv)])
            red = make_reducer("pca", k).fit(M); EN.append(energy(X[nm], base[nm]["pre"], cv, t, red).assign(sort=nm, space=f"pca{k}_fit_2-1002ms"))
        guard.check(f"energy {nm}")
    EN = pd.concat(EN, ignore_index=True); EN.to_csv(ctx.RP("window_energy.csv"), index=False)
    # same-chunk baseline bias: is the 1.0-1.9 s rate still different from the pre-pulse rate?
    bias = []
    for nm in ("online", "ks25"):
        for c in np.unique(cv):
            D = base[nm]["same"][cv == c] - base[nm]["pre"][cv == c]; n = len(D)
            bias.append(dict(sort=nm, cond=c, energy=float((D.mean(0) ** 2).sum()), noise=float((D.var(0, ddof=1) / n).sum()), mean_rate_change_hz=float(D.mean())))
    pd.DataFrame(bias).assign(signal=lambda d: d.energy - d.noise).to_csv(ctx.RP("window_baseline_bias.csv"), index=False)
    rows = []
    for (a, b) in [(2, 502), (2, 1002), (2, 1892), (502, 1002), (1002, 1892)]:
        for bl in ("same", "pre", "none"):
            if bl == "same" and b > 1000: continue
            for k in (3, 10):
                ao, ar, Zo, Zr = evaluate(X["online"], X["ks25"], base["online"][bl], base["ks25"][bl], seg(a, b), cv, k)
                rows.append(dict(window=f"{a}-{b} ms", start_ms=a, end_ms=b, baseline=bl, k=k, acc_online=ao, acc_ks25=ar, rho_online_vs_ks25=cond_geometry_rho(Zo, Zr, cv) if k == 3 else np.nan))
                print(f"[win] {a}-{b} ms, baseline {bl}, k={k}: acc {ao:.3f} / {ar:.3f}", flush=True)
        guard.check(f"window {a}-{b}")
    # ---- accuracy vs response-window length
    srows = []
    for W in range(102, 1503, 100):
        for bl in ("pre", "same"):
            if bl == "same" and W > 1002: continue
            for k in (3, 10):
                ao, ar, _, _ = evaluate(X["online"], X["ks25"], base["online"][bl], base["ks25"][bl], seg(2, W), cv, k)
                srows.append(dict(window_end_ms=W, baseline=bl, k=k, acc_online=ao, acc_ks25=ar))
        print(f"[win] sweep {W} ms: " + ", ".join(f"{r['baseline']} k{r['k']} {r['acc_online']:.3f}" for r in srows if r["window_end_ms"] == W), flush=True)
    pd.DataFrame(srows).assign(chance=1 / len(np.unique(cv))).to_csv(ctx.RP("window_sweep.csv"), index=False); guard.check("sweep")
    # ---- loss timing: distance of each trial's trajectory (2-502 ms, 3 PCs, pre-pulse baseline) to its condition target
    import time as _t
    _, _, Zo, _ = evaluate(X["online"], X["ks25"], base["online"]["pre"], base["ks25"]["pre"], seg(2, 502), cv, 3)
    tgt = {c: Zo[cv == c].mean(0) for c in np.unique(cv)}; tp, te = [], []
    for i in range(len(Zo)):
        a = _t.perf_counter(); procrustes_distance(tgt[cv[i]], Zo[i]); b = _t.perf_counter(); float(np.linalg.norm(tgt[cv[i]] - Zo[i])); c_ = _t.perf_counter()
        tp.append((b - a) * 1e3); te.append((c_ - b) * 1e3)
    tp, te = np.array(tp[5:]), np.array(te[5:])
    import json as _j
    _j.dump(dict(shape=list(Zo.shape[1:]), n=int(len(tp)), procrustes_ms_mean=float(tp.mean()), procrustes_ms_sd=float(tp.std()), procrustes_ms_p99=float(np.percentile(tp, 99)),
                 euclid_ms_mean=float(te.mean()), euclid_ms_p99=float(np.percentile(te, 99))), open(ctx.RP("window_loss_timing.json"), "w"), indent=2)
    print(f"[win] loss timing: Procrustes {tp.mean():.3f} ms mean, p99 {np.percentile(tp, 99):.3f} ms", flush=True)
    # baseline-only control: how much condition (= run = time) information is in the baseline itself
    for bl in ("same", "pre"):
        for nm in ("online", "ks25"):
            Bv = base[nm][bl][:, None, :]; red = make_reducer("pca", 3).fit(np.concatenate([Bv[cv == c].mean(0) for c in np.unique(cv)]))
            rows.append(dict(window="baseline only", start_ms=np.nan, end_ms=np.nan, baseline=bl, k=3, **{f"acc_{nm}": float(g7.classify(np.stack([red.transform(x) for x in Bv]), cv, "euclid")[0].mean())}))
    W = pd.DataFrame(rows).groupby(["window", "start_ms", "end_ms", "baseline", "k"], dropna=False, sort=False).first().reset_index()
    W["chance"] = 1 / len(np.unique(cv)); W.to_csv(ctx.RP("window_analysis.csv"), index=False)
    # time-resolved information, 100 ms segments
    rows = []
    for a in np.arange(2, 1892, 100):
        b = min(a + 100, 1892)
        for bl in ("pre", "none"):
            ao, ar, _, _ = evaluate(X["online"], X["ks25"], base["online"][bl], base["ks25"][bl], seg(a, b), cv, 3); rows.append(dict(start_ms=a, end_ms=b, baseline=bl, acc_online=ao, acc_ks25=ar))
    pd.DataFrame(rows).to_csv(ctx.RP("window_timecourse.csv"), index=False)
    # condition-mean trajectories, PCA fitted on 2-1002 ms (pre-pulse baseline), projected over the whole chunk
    cs = np.unique(cv); sel = seg(2, 1002)
    Mo = np.stack([(X["online"][cv == c] - base["online"]["pre"][cv == c][:, None, :]).mean(0) for c in cs])
    Mr = np.stack([(X["ks25"][cv == c] - base["ks25"]["pre"][cv == c][:, None, :]).mean(0) for c in cs])
    red = make_reducer("pca", 3).fit(np.concatenate([m[sel] for m in Mr]))
    np.savez_compressed(ctx.RP("window_psth.npz"), t_ms=t, conds=cs, pc_online=Mo @ red.components_.T, pc_ks25=Mr @ red.components_.T,   # origin = pre-pulse baseline
                        evr=np.asarray(getattr(red, "explained_variance_ratio_", [np.nan])))
    print(W.to_string(index=False), flush=True); guard.check("done")

if __name__ == "__main__":
    main()
