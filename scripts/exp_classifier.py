#!/usr/bin/env python
"""Oracle-detection classification experiment  (memory-bounded).

Give the classifier Kilosort 4's own spike times and peak channels (perfect
detection) and measure how often it picks the right unit.  Compares
own-channel scoring (each candidate on its own 12 local channels) against
common-channel scoring (all candidates on the detection channel's K nearest
channels) with diagonal or full spatial noise whitening.

Safety: an earlier form of this experiment crashed the 8 GB dev machine.
This version (1) refuses to start unless enough memory is available,
(2) aborts cleanly if its own peak RSS passes BUDGET_GB, (3) never imports
SpikeInterface, (4) filters in 1 s chunks, and (5) caps every candidate
tensor at MAX_EL floats by splitting events into blocks.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamsort.memguard import preflight, MemGuard  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import OpenEphysSession, FileStreamSource  # noqa: E402
from streamsort.preprocess import CausalPreprocessor  # noqa: E402
from streamsort.sorting import TemplateBank  # noqa: E402

BUDGET_GB = 3.5          # user limit: stay < 4 GB; macOS may swap a little
MAX_EL = 6_000_000          # floats per candidate tensor block (24 MB)
ROOT = "/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58"
KSOUT = "/Users/danieldenman/stim_scratch/jlh32_probeA/sort_kilosort4_seg600/sorter_output"


def chunked_process(src, cfg, start, n, chunk_s=1.0):
    pp = CausalPreprocessor(cfg, 200)
    step = cfg.samples(chunk_s * 1e3)
    out = np.empty((n, 200), np.float32)
    for s in range(0, n, step):
        m = min(step, n - s)
        out[s:s + m] = pp.process(np.asarray(src.read(start + s, m)))
    return out


def chunked_cov(src, cfg, start, n, chunk_s=1.0, skip_ms=50):
    pp = CausalPreprocessor(cfg, 200)
    step = cfg.samples(chunk_s * 1e3)
    sk = cfg.samples(skip_ms)
    S = np.zeros((200, 200)); mu = np.zeros(200); cnt = 0
    for s in range(0, n, step):
        y = pp.process(np.asarray(src.read(start + s, min(step, n - s)))).astype(np.float64)
        if s == 0:
            y = y[sk:]
        S += y.T @ y; mu += y.sum(0); cnt += len(y)
    mu /= cnt
    return S / cnt - np.outer(mu, mu)


def inv_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V / np.sqrt(np.maximum(w, 1e-9))) @ V.T


def main(seconds_start=20.0, seconds=5.0):
    preflight(BUDGET_GB, margin_gb=0.0, strict=False)   # warn only; swapping is acceptable
    guard = MemGuard(BUDGET_GB, "exp_classifier")
    st = OpenEphysSession(ROOT + "/Record Node 105/experiment1/recording1").stream("Neuropix-PXI-104.ProbeA-AP")
    cfg = StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts)
    src = FileStreamSource(st, 200)
    bank = TemplateBank.load("results/templates_jlh32_probeA.npz")
    rep = pd.read_csv("results/sorting_report.csv")
    usable = rep.usable.to_numpy(bool)
    uidx = {int(u): k for k, u in enumerate(bank.unit_ids)}

    st_all = np.load(KSOUT + "/spike_times.npy", mmap_mode="r").ravel()
    cl_all = np.load(KSOUT + "/spike_clusters.npy", mmap_mode="r").ravel()
    S0, N, W = cfg.samples(seconds_start * 1e3), cfg.samples(seconds * 1e3), cfg.samples(15)
    lo, hi = S0 + 300, S0 + N - 300
    sel = (st_all >= lo) & (st_all < hi)
    t_win = np.asarray(st_all[sel]).astype(np.int64)
    c_win = np.asarray(cl_all[sel]).astype(np.int64)
    tot = np.bincount(np.asarray(cl_all).astype(np.int64), minlength=int(bank.unit_ids.max()) + 1)
    chk = rep.set_index("unit_id").n_spikes_segment
    mism = int((tot[chk.index.to_numpy()] != chk.to_numpy()).sum())
    print(f"[exp] KS4 spike files vs sorting report: {mism} unit count mismatches", flush=True)

    t = time.perf_counter()
    Yw = chunked_process(src, cfg, S0 - W, N + W)
    Sig = chunked_cov(src, cfg, cfg.samples(40_000), cfg.samples(5_000))
    guard.check("data ready")
    print(f"[exp] data ready {time.perf_counter()-t:.1f}s | {guard.report()}", flush=True)

    pos = bank.channel_locations.astype(float)
    Dch = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    j1 = np.argsort(Dch, axis=1)[:, 1]
    corr = [Sig[c, j1[c]] / np.sqrt(Sig[c, c] * Sig[j1[c], j1[c]]) for c in range(200)]
    print(f"[exp] noise correlation with nearest site: median {np.median(corr):.2f}", flush=True)

    keep = np.isin(c_win, bank.unit_ids[usable])
    TRU = c_win[keep]
    kk = np.array([uidx[int(u)] for u in TRU])
    G = t_win[keep] - cfg.n_before + bank.trough_offset[kk]
    CC = bank.peak_channel[kk]
    print(f"[exp] oracle events {len(G)}", flush=True)

    ar = np.arange(bank.n_samples); Ys = S0 - W; T = bank.n_samples

    def run(K, whiten, own=False, radius=50.0, J=2, amp=(0.5, 1.8)):
        S = np.argsort(Dch, axis=1, kind="stable")[:, :K]
        dd = Dch[:, bank.peak_channel]
        cand = [np.flatnonzero((dd[c] <= radius) & usable & (bank.n_spikes_used > 0)) for c in range(200)]
        sh = np.arange(-J, J + 1); nS = len(sh)
        out = np.full(len(G), -1)
        for c in np.unique(CC):
            idx_all = np.flatnonzero(CC == c); cu = cand[c]
            if not len(cu):
                continue
            n = len(cu); tro = bank.trough_offset[cu]
            if own:
                Lw = bank.local_channels[cu]
                wts = 1.0 / bank.noise_uV[Lw] ** 2
                Tw = bank.templates[cu] * wts[:, None, :]
                tt = (bank.templates[cu] * Tw).sum(axis=(1, 2))
                per = n * nS * T * Lw.shape[1]
            else:
                Sc = Sig[np.ix_(S[c], S[c])]
                Wc = inv_sqrt(Sc) if whiten else np.diag(1 / np.sqrt(np.diag(Sc)))
                Ysub = (Yw[:, S[c]] @ Wc).astype(np.float32)
                Tc = np.zeros((n, T, K), np.float32)
                for j, u in enumerate(cu):
                    where = {ch: q for q, ch in enumerate(bank.local_channels[u])}
                    for q, ch in enumerate(S[c]):
                        if ch in where:
                            Tc[j, :, q] = bank.templates[u][:, where[ch]]
                T_ = (Tc @ Wc).astype(np.float32)
                tt = (T_ * T_).sum(axis=(1, 2))
                per = n * nS * T * K
            B = max(1, MAX_EL // per)
            for b0 in range(0, len(idx_all), B):
                idx = idx_all[b0:b0 + B]
                st_ = (G[idx] - Ys)[:, None, None] - tro[None, :, None] + sh[None, None, :]
                rows = np.clip(st_[..., None] + ar, 0, len(Yw) - 1)
                if own:
                    xt = np.empty((len(idx), n, nS), np.float32)
                    for j in range(n):
                        X = Yw[:, Lw[j]][rows[:, j]]
                        xt[:, j] = np.einsum("estl,tl->es", X, Tw[j])
                else:
                    X = Ysub[rows]
                    xt = np.einsum("enstk,ntk->ens", X, T_, optimize=True)
                a = xt / tt[None, :, None]
                sc = np.where((a >= amp[0]) & (a <= amp[1]), xt * a, -np.inf)
                best = sc.reshape(len(idx), -1).argmax(axis=1) // nS
                out[idx] = bank.unit_ids[cu][best]
                guard.check(f"channel {c}")
        return float(np.mean(out == TRU))

    for name, kw in [("own channels, diagonal (current)", dict(K=12, whiten=False, own=True)),
                     ("common K=12, diagonal", dict(K=12, whiten=False)),
                     ("common K=16, diagonal", dict(K=16, whiten=False)),
                     ("common K=16, whitened", dict(K=16, whiten=True)),
                     ("common K=24, whitened", dict(K=24, whiten=True)),
                     ("common K=32, whitened", dict(K=32, whiten=True))]:
        t = time.perf_counter(); acc = run(**kw)
        print(f"[exp] {name:34s} oracle correct unit {acc:.3f}  ({time.perf_counter()-t:.1f}s) | {guard.report()}",
              flush=True)


if __name__ == "__main__":
    main()
