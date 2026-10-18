"""Numba kernels for the decoder hot path.

Each kernel has a numpy reference implementation elsewhere in the package
(``CausalPreprocessor``, ``OnlineDecoder`` with ``use_numba=False``); the test
suite checks the two agree.  Kernels are compiled at decoder construction by
:func:`warmup` -- compiled caches do not carry across platforms, so on the
Windows acquisition rig a first-call compile inside ``process()`` would stall
the acquisition thread for seconds.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:          # pragma: no cover - numba optional
    nb = None
    HAVE_NUMBA = False


if HAVE_NUMBA:
    @nb.njit(parallel=True, cache=False)
    def sos_filter(x, scale, sos, zi):
        """Causal SOS cascade, direct form II transposed (scipy.sosfilt semantics).

        x : (N, C) int16 or float32 raw samples; scale multiplies on the way in.
        sos : (S, 6) float64; zi : (S, 2, C) float64 state, updated in place.
        Returns (N, C) float64 filtered samples.
        """
        N, C = x.shape
        S = sos.shape[0]
        y = np.empty((N, C), np.float64)
        for c in nb.prange(C):
            z0 = np.empty(S); z1 = np.empty(S)
            for s in range(S):
                z0[s] = zi[s, 0, c]; z1[s] = zi[s, 1, c]
            for n in range(N):
                v = x[n, c] * scale
                for s in range(S):
                    out = sos[s, 0] * v + z0[s]
                    z0[s] = sos[s, 1] * v - sos[s, 4] * out + z1[s]
                    z1[s] = sos[s, 2] * v - sos[s, 5] * out
                    v = out
                y[n, c] = v
            for s in range(S):
                zi[s, 0, c] = z0[s]; zi[s, 1, c] = z1[s]
        return y

    @nb.njit(parallel=True, cache=False)
    def median_reference(y):
        """Subtract the across-channel median from every sample; returns float32."""
        N, C = y.shape
        out = np.empty((N, C), np.float32)
        for n in nb.prange(N):
            row = y[n].copy()
            m = np.median(row)
            for c in range(C):
                out[n, c] = y[n, c] - m
        return out

    @nb.njit(parallel=True, cache=False)
    def detect_minima(Y, thr, nbr, dedup, lo, hi):
        """Mask of samples in [lo, hi) that are space-time local minima below -thr.

        A sample (n, c) qualifies if Y[n,c] < -thr[c] and Y[n,c] is <= every
        value in the neighbourhood {n-dedup..n+dedup} x nbr[c].  Only samples
        already below threshold pay for the neighbourhood scan.
        """
        N, C = Y.shape
        K = nbr.shape[1]
        mask = np.zeros((N, C), np.bool_)
        for c in nb.prange(C):
            for n in range(lo, hi):
                v = Y[n, c]
                if v >= -thr[c]:
                    continue
                a = max(n - dedup, 0); b = min(n + dedup + 1, N)
                ok = True
                for m in range(a, b):
                    for k in range(K):
                        if Y[m, nbr[c, k]] < v:
                            ok = False
                            break
                    if not ok:
                        break
                mask[n, c] = ok
        return mask

    @nb.njit(parallel=True, cache=False)
    def score_events(Y, g_rel, cc, cand_ptr, tro_cand, sset, Tcat, ttcat, shifts, amp_lo, amp_hi):
        """Best (candidate, shift) per event by explained energy on a shared channel set.

        Y : (N, C) whitened data; g_rel : event trough sample relative to Y.
        cand_ptr : (C+1,) CSR pointers; candidates of channel c are rows
        cand_ptr[c]..cand_ptr[c+1] of Tcat (M, T, K) / ttcat (M,) / tro_cand (M,).
        Returns best row index (-1 if none), score, amplitude, shift.
        """
        E = g_rel.shape[0]
        T = Tcat.shape[1]; K = Tcat.shape[2]; NS = shifts.shape[0]
        N = Y.shape[0]
        best_j = np.full(E, -1, np.int64)
        best_sc = np.zeros(E, np.float32)
        best_a = np.zeros(E, np.float32)
        best_s = np.zeros(E, np.int64)
        for e in nb.prange(E):
            c = cc[e]
            bsc = -1.0e30; bj = -1; ba = 0.0; bs = 0
            for j in range(cand_ptr[c], cand_ptr[c + 1]):
                tt = ttcat[j]
                for si in range(NS):
                    st = g_rel[e] - tro_cand[j] + shifts[si]
                    if st < 0 or st + T > N:
                        continue
                    dot = 0.0
                    for t in range(T):
                        row = st + t
                        for k in range(K):
                            dot += Y[row, sset[c, k]] * Tcat[j, t, k]
                    a = dot / tt
                    if a < amp_lo or a > amp_hi:
                        continue
                    sc = dot * a
                    if sc > bsc:
                        bsc = sc; bj = j; ba = a; bs = shifts[si]
            if bj >= 0:
                best_j[e] = bj; best_sc[e] = bsc; best_a[e] = ba; best_s[e] = bs
        return best_j, best_sc, best_a, best_s


def warmup() -> None:
    """Compile every kernel on tiny inputs (call before acquisition starts)."""
    if not HAVE_NUMBA:
        return
    x = np.zeros((64, 4), np.int16)
    sos = np.array([[1.0, 0, 0, 1.0, 0, 0]])
    zi = np.zeros((1, 2, 4))
    y = sos_filter(x, 1.0, sos, zi)
    Y = median_reference(y)
    nbr = np.tile(np.arange(4), (4, 1)).astype(np.int64)
    detect_minima(Y, np.ones(4, np.float32), nbr, 2, 0, 64)
    score_events(Y, np.array([10], np.int64), np.array([0], np.int64),
                 np.array([0, 1, 1, 1, 1], np.int64), np.array([2], np.int64),
                 np.zeros((4, 2), np.int64), np.ones((1, 8, 2), np.float32),
                 np.ones(1, np.float32), np.arange(-1, 2), 0.5, 1.8)
