#!/usr/bin/env python
"""Global local-ZCA whitening: does it fix both detection and classification?

Kilosort-style whitening: each output channel c is a combination of its Kw
nearest channels, using the row of inv_sqrt(Sigma_local) for c.  Applied to
the whole processed stream once (Y @ W.T), and to the templates (linear, so
the existing bank is reused without a rebuild).

Measures, on 5 s of in-sample data against Kilosort 4's own spikes:
  1. oracle classification (perfect detection) on whitened data, common K=16
  2. full detect+classify accuracy: correct / labelled-as-neighbour / missed,
     unwhitened vs whitened, over several thresholds, with event rates
  3. cost of the whitening matmul per 1.9 s chunk

Memory-bounded: preflight warn, MemGuard cap 3.5 GB, 2 BLAS threads.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamsort.memguard import preflight, MemGuard  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import OpenEphysSession, FileStreamSource  # noqa: E402
from streamsort.sorting import TemplateBank  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_classifier import chunked_process, chunked_cov, inv_sqrt, ROOT, KSOUT  # noqa: E402

BUDGET_GB = 3.5
MAX_EL = 6_000_000


def local_zca(Sig, Dch, Kw=32):
    C = Sig.shape[0]
    W = np.zeros((C, C))
    for c in range(C):
        S = np.argsort(Dch[c], kind="stable")[:Kw]
        Wl = inv_sqrt(Sig[np.ix_(S, S)])
        q = int(np.flatnonzero(S == c)[0])
        W[c, S] = Wl[q]
    return W


def detect(Y, thr, nbr, dedup, block=20_000):
    """Local space-time minima below -thr, in time blocks (bounded memory)."""
    T_all, C_all = [], []
    n = len(Y)
    for b0 in range(0, n, block):
        lo_ = max(b0 - dedup, 0); hi_ = min(b0 + block + dedup, n)
        seg = Y[lo_:hi_]
        smin = seg[:, nbr].min(axis=2)
        stmin = minimum_filter1d(smin, size=2 * dedup + 1, axis=0, mode="nearest")
        t, c = np.nonzero((seg <= stmin) & (seg < -thr[None, :]))
        t = t + lo_
        k = (t >= b0) & (t < min(b0 + block, n))
        T_all.append(t[k]); C_all.append(c[k])
    return np.concatenate(T_all), np.concatenate(C_all)


def main(seconds_start=20.0, seconds=5.0):
    preflight(BUDGET_GB, margin_gb=0.0, strict=False)
    guard = MemGuard(BUDGET_GB, "exp_whitened")
    st = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1").stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts)
    src = FileStreamSource(st, 200)
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    rep = pd.read_csv("results/sorting_report.csv")
    usable = rep.usable.to_numpy(bool)
    uidx = {int(u): k for k, u in enumerate(bank.unit_ids)}
    T = bank.n_samples

    st_all = np.load(KSOUT + "/spike_times.npy", mmap_mode="r").ravel()
    cl_all = np.load(KSOUT + "/spike_clusters.npy", mmap_mode="r").ravel()
    S0, N, W0 = cfg.samples(seconds_start * 1e3), cfg.samples(seconds * 1e3), cfg.samples(15)
    lo, hi = S0 + 300, S0 + N - 300
    sel = (st_all >= lo) & (st_all < hi)
    t_win = np.asarray(st_all[sel]).astype(np.int64)
    c_win = np.asarray(cl_all[sel]).astype(np.int64)

    Yraw = chunked_process(src, cfg, S0 - W0, N + W0)
    Sig = chunked_cov(src, cfg, cfg.samples(40_000), cfg.samples(5_000))
    pos = bank.channel_locations.astype(float)
    Dch = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    Wg = local_zca(Sig, Dch, Kw=32)
    t = time.perf_counter()
    Yw = (Yraw @ Wg.T).astype(np.float32)
    dt_w = time.perf_counter() - t
    print(f"[exp] whitening matmul: {dt_w/seconds*1.9*1e3:.0f} ms per 1.9 s chunk | {guard.report()}", flush=True)
    guard.check("whitened")

    # whitened noise level per channel (robust), from a stim-free stretch
    nz = chunked_process(src, cfg, cfg.samples(45_000), cfg.samples(3_000))[cfg.samples(50):]
    sd_raw = (np.median(np.abs(nz - np.median(nz, 0)), 0) / 0.6745).astype(np.float32)
    nzw = nz @ Wg.T
    sd_w = (np.median(np.abs(nzw - np.median(nzw, 0)), 0) / 0.6745).astype(np.float32)
    del nz, nzw
    print(f"[exp] robust noise sd: raw median {np.median(sd_raw):.2f} uV | whitened median {np.median(sd_w):.3f}",
          flush=True)

    # whitened templates (dense, then common-set scoring)
    Td = bank.dense(200)
    Tdw = (Td @ Wg.T).astype(np.float32)          # (U,T,C)
    del Td

    # ground truth for usable units in window
    keep = np.isin(c_win, bank.unit_ids[usable])
    TRU = c_win[keep]; kk = np.array([uidx[int(u)] for u in TRU])
    Gt = t_win[keep]                                   # sorter-convention frames
    G = Gt - cfg.n_before + bank.trough_offset[kk]     # trough sample (unwhitened)
    CC = bank.peak_channel[kk]
    Ys = S0 - W0

    K = 16
    Sset = np.argsort(Dch, axis=1, kind="stable")[:, :K]
    dd = Dch[:, bank.peak_channel]
    cand = [np.flatnonzero((dd[c] <= 50.0) & usable & (bank.n_spikes_used > 0)) for c in range(200)]
    ar = np.arange(T); sh = np.arange(-2, 3); nS = len(sh)
    tro_all = bank.trough_offset

    def classify(Ydat, Tdat, Gev, Cev, amp=(0.5, 1.8)):
        lab = np.full(len(Gev), -1); scr = np.zeros(len(Gev)); shf = np.zeros(len(Gev), np.int64)
        for c in np.unique(Cev):
            idx_all = np.flatnonzero(Cev == c); cu = cand[c]
            if not len(cu):
                continue
            n = len(cu); tro = tro_all[cu]
            Ysub = Ydat[:, Sset[c]]
            T_ = Tdat[cu][:, :, Sset[c]]
            tt = (T_ * T_).sum(axis=(1, 2))
            B = max(1, MAX_EL // (n * nS * T * K))
            for b0 in range(0, len(idx_all), B):
                idx = idx_all[b0:b0 + B]
                st_ = (Gev[idx] - Ys)[:, None, None] - tro[None, :, None] + sh[None, None, :]
                X = Ysub[np.clip(st_[..., None] + ar, 0, len(Ydat) - 1)]
                xt = np.einsum("enstk,ntk->ens", X, T_, optimize=True)
                a = xt / tt[None, :, None]
                sc = np.where((a >= amp[0]) & (a <= amp[1]), xt * a, -np.inf)
                flat = sc.reshape(len(idx), -1); arg = flat.argmax(axis=1)
                lab[idx] = bank.unit_ids[cu][arg // nS]; scr[idx] = flat[np.arange(len(idx)), arg]
                shf[idx] = sh[arg % nS]
                guard.check(f"classify ch {c}")
        return lab, scr, shf

    # 1. oracle classification on whitened data
    lab, _, _ = classify(Yw, Tdw, G, CC)
    print(f"[exp] oracle, globally whitened, common K=16: correct unit {np.mean(lab == TRU):.3f}", flush=True)

    # 2. full pipeline, unwhitened (diag-normalised) vs whitened
    nb = [np.flatnonzero(Dch[c] <= cfg.dedup_radius_um) for c in range(200)]
    Kn = max(len(x) for x in nb)
    nbr = np.array([np.pad(x, (0, Kn - len(x)), constant_values=c) for c, x in enumerate(nb)])
    # unwhitened scoring uses diagonal noise normalisation of data and templates
    Ydiag = Yraw / sd_raw[None, :]
    Tdiag = (bank.dense(200) / sd_raw[None, None, :]).astype(np.float32)
    rows = []
    for name, Ydat, Tdat, sd in [("unwhitened", Ydiag, Tdiag, np.ones(200, np.float32)),
                                 ("whitened", Yw, Tdw, sd_w)]:
        for thr in (5.0, 4.5, 4.0, 3.5):
            tdet, cdet = detect(Ydat, thr * sd, nbr, cfg.dedup_window_samples)
            g = tdet + Ys
            ok = (g >= lo) & (g < hi); g, cdet = g[ok], cdet[ok]
            lab, scr, shf = classify(Ydat, Tdat, g, cdet)
            # convert to sorter frames
            assigned = lab >= 0
            tro = np.where(assigned, np.array([tro_all[uidx[int(u)]] if u >= 0 else cfg.n_before for u in lab]), cfg.n_before)
            fr = g + shf - tro + cfg.n_before
            # score against truth with spatial constraint (event channel within 75 um of unit peak)
            o = np.argsort(fr); F, L, PC = fr[o], lab[o], cdet[o]
            cnt = dict(correct=0, neighbour=0, missed=0)
            for u in np.unique(TRU):
                tr = Gt[TRU == u]; p = pos[bank.peak_channel[uidx[int(u)]]]
                m = np.linalg.norm(pos[PC] - p, axis=1) <= 75
                Fu, Lu = F[m], L[m]
                if len(Fu) == 0:
                    cnt["missed"] += len(tr); continue
                k = np.clip(np.searchsorted(Fu, tr), 1, max(len(Fu) - 1, 1)); k2 = np.minimum(k, len(Fu) - 1)
                j = np.where(np.abs(tr - Fu[k - 1]) <= np.abs(tr - Fu[k2]), k - 1, k2)
                hit = np.abs(tr - Fu[j]) <= 12
                cnt["missed"] += int((~hit).sum())
                cnt["correct"] += int((hit & (Lu[j] == u)).sum())
                cnt["neighbour"] += int((hit & (Lu[j] != u)).sum())
            totn = sum(cnt.values())
            # precision proxy: events labelled u that match a KS4 spike of u
            n_lab = int(np.isin(L, bank.unit_ids[usable]).sum())
            rows.append(dict(space=name, thr=thr, events_per_s=len(g) / seconds,
                             correct=cnt["correct"] / totn, neighbour=cnt["neighbour"] / totn,
                             missed=cnt["missed"] / totn,
                             precision=cnt["correct"] / max(n_lab, 1)))
            print(f"[exp] {name:10s} thr {thr:.1f}: events/s {len(g)/seconds:6.0f} | correct {rows[-1]['correct']:.3f} "
                  f"neighbour {rows[-1]['neighbour']:.3f} missed {rows[-1]['missed']:.3f} | precision {rows[-1]['precision']:.3f} "
                  f"| {guard.report()}", flush=True)
    pd.DataFrame(rows).to_csv("results/exp_whitened.csv", index=False)
    np.save("results/whitening_matrix_probeA.npy", Wg.astype(np.float32))
    print("[exp] DONE", flush=True)


if __name__ == "__main__":
    main()
