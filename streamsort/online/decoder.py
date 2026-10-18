"""Push-based online spike decoder: detect, then classify against a template bank.

One object serves file replay and the Open Ephys plugin alike.  It owns all
state that must persist across buffers -- causal filter state, a raw
look-ahead for artifact blanking, the processed-sample history needed for
waveform windows -- so the caller simply pushes buffers in order:

    dec = OnlineDecoder(bank, cfg, whitening)
    for start, raw in stream.iter_buffers(1024):
        spikes = dec.push(raw)          # SpikeBatch of newly finalised spikes
    spikes = dec.flush()

Pipeline per buffer
-------------------
1. **Artifact blanking (causal, data-derived).**  Samples where >= 3 channels
   sit at the amplifier rail mark a stimulus.  Blanking covers 0.2 ms before
   the first clipped sample to 0.5 ms after the last (step-2 policy).  The
   pre-onset part needs a sample of the future, so raw data is held back by a
   1 ms look-ahead before filtering.  The TTL is *not* used: it jitters by up
   to +-0.8 ms against the artifact on this rig.
2. **Causal highpass + common median reference** -- bit-identical to the path
   the template bank was built on (enforced by config fingerprint).
3. **Spatial whitening** (local ZCA, see :mod:`.whitening`).  Removes the
   noise correlation between neighbouring sites that otherwise inflates
   matches to the wrong unit.
4. **Detection** on whitened data: negative threshold at ``detect_threshold``
   whitened-noise SDs; an event must be the minimum of its space-time
   neighbourhood (+-``dedup_window_samples``, channels within
   ``dedup_radius_um``).
5. **Classification.**  All candidate units (peak within
   ``classify_radius_um`` of the detection channel) are scored on one shared
   set of the ``score_channels`` channels nearest the detection channel, so
   their scores are comparable.  Score = explained energy of a least-squares
   amplitude fit, (x.t)^2 / (t.t), in whitened-noise units, maximised over
   +-``jitter`` samples.  The best unit is accepted if the fitted amplitude
   is plausible and the energy exceeds ``min_energy``; otherwise the event is
   reported as unit -1 (detected, unassigned).

Spike times are reported in the *offline sorter's* timing convention (the
frame Kilosort would assign), using each unit's stored trough offset, so they
compare directly with offline spike trains.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from ..config import StreamConfig
from ..preprocess.artifact import CLIP_COUNTS
from ..preprocess.filters import CausalPreprocessor
from ..sorting.templates import TemplateBank
from .whitening import Whitening
from . import kernels


@dataclass
class SpikeBatch:
    frame: np.ndarray         # int64, sample index in sorter convention
    unit: np.ndarray          # int64, bank unit id, -1 = unassigned
    score: np.ndarray         # float32, explained energy of best fit
    amp: np.ndarray           # float32, fitted amplitude scale
    peak_channel: np.ndarray  # int64, detection channel

    FIELDS = ("frame", "unit", "score", "amp", "peak_channel")

    @classmethod
    def empty(cls) -> "SpikeBatch":
        z = np.zeros(0, np.int64)
        return cls(z, z.copy(), np.zeros(0, np.float32), np.zeros(0, np.float32), z.copy())

    @classmethod
    def concat(cls, batches) -> "SpikeBatch":
        batches = [b for b in batches if len(b.frame)]
        if not batches:
            return cls.empty()
        return cls(*(np.concatenate([getattr(b, f) for b in batches]) for f in cls.FIELDS))

    def select(self, mask) -> "SpikeBatch":
        return SpikeBatch(*(getattr(self, f)[mask] for f in self.FIELDS))

    def __len__(self) -> int:
        return int(len(self.frame))


@dataclass
class DecoderParams:
    classify_radius_um: float = 50.0
    score_channels: int = 16        # shared channel set per event
    jitter: int = 2
    min_energy: float = 25.0        # explained energy, whitened-noise units
    amp_range: tuple = (0.5, 1.8)
    lookahead_ms: float = 1.0
    blank_pre_ms: float = 0.2
    blank_post_ms: float = 0.5
    clip_counts: int = CLIP_COUNTS
    clip_min_channels: int = 3
    max_block_floats: int = 6_000_000   # bounds the scoring tensor (24 MB)


class OnlineDecoder:
    def __init__(self, bank: TemplateBank, cfg: StreamConfig, whitening: Whitening | None,
                 params: DecoderParams | None = None, units_mask: np.ndarray | None = None,
                 check: bool = True, use_numba: bool = True, suppressor=None):
        self.suppressor = suppressor      # optional ArtifactSuppressor (raw-domain, causal)
        if check:
            bank.check_compatible(cfg)
            if whitening is not None and whitening.fingerprint != cfg.fingerprint():
                raise ValueError("whitening was estimated under a different signal path")
        self.bank, self.cfg = bank, cfg
        self.p = params or DecoderParams()
        C = len(bank.noise_uV)
        self.C = C
        self.pre = CausalPreprocessor(cfg, C)
        if whitening is None:     # diagonal normalisation only
            self.W = np.diag(1.0 / np.maximum(bank.noise_uV, 1e-3)).astype(np.float32)
            sd = np.ones(C, np.float32)
        else:
            self.W, sd = whitening.W.astype(np.float32), whitening.noise_sd.astype(np.float32)
        self.WT = np.ascontiguousarray(self.W.T)
        self.thr = (cfg.detect_threshold * sd).astype(np.float32)

        pos = bank.channel_locations[:C].astype(np.float64)
        D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        nb = [np.flatnonzero(D[c] <= cfg.dedup_radius_um) for c in range(C)]
        K = max(len(x) for x in nb)
        self.nbr = np.array([np.pad(x, (0, K - len(x)), constant_values=c)
                             for c, x in enumerate(nb)], dtype=np.int64)

        ok_unit = (bank.n_spikes_used > 0)
        if units_mask is not None:
            ok_unit = ok_unit & np.asarray(units_mask, bool)
        Dpk = D[:, bank.peak_channel]
        self.cand = [np.flatnonzero((Dpk[c] <= self.p.classify_radius_um) & ok_unit) for c in range(C)]
        Ks = self.p.score_channels
        self.sset = np.argsort(D, axis=1, kind="stable")[:, :Ks]
        Tdw = (bank.dense(C) @ self.WT).astype(np.float32)          # whitened templates
        self.Tc = [Tdw[self.cand[c]][:, :, self.sset[c]] for c in range(C)]
        self.ttc = [(t * t).sum(axis=(1, 2)) for t in self.Tc]

        self.T = bank.n_samples
        self.tro = bank.trough_offset.astype(np.int64)
        J = self.p.jitter
        self.shifts = np.arange(-J, J + 1)
        self.before_need = int(self.tro.max()) + J
        self.after_need = int((self.T - self.tro).max()) + J
        self.dedup = int(cfg.dedup_window_samples)
        self.lag = cfg.samples(self.p.lookahead_ms)
        self.bpre = cfg.samples(self.p.blank_pre_ms)
        self.bpost = cfg.samples(self.p.blank_post_ms)
        if self.bpre > self.lag:
            raise ValueError("look-ahead must cover the pre-onset blank")

        # numba fast path: flat (CSR) candidate tables + JIT warm-up
        self.use_numba = bool(use_numba and kernels.HAVE_NUMBA)
        if self.use_numba:
            if cfg.common_reference not in ("median",):
                raise ValueError("numba path implements median referencing only")
            sizes = np.array([len(x) for x in self.cand], np.int64)
            self.cand_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            self.cand_units = (np.concatenate(self.cand) if sizes.sum() else np.zeros(0)).astype(np.int64)
            self.Tcat = (np.concatenate(self.Tc) if sizes.sum() else
                         np.zeros((0, self.T, Ks))).astype(np.float32)
            self.ttcat = np.concatenate(self.ttc).astype(np.float32) if sizes.sum() else np.zeros(0, np.float32)
            self.tro_cand = self.tro[self.cand_units]
            self.sset64 = self.sset.astype(np.int64)
            self.sos = self.pre.sos.astype(np.float64)
            kernels.warmup()
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        self.pre.reset()
        if getattr(self, "use_numba", False):
            self._zi = np.zeros((self.sos.shape[0], 2, self.C), np.float64)
            self._primed = False
        self._pending = np.zeros((0, self.C), np.int16)
        self._pend_start = 0
        self._blank_until = -1
        self._hold = None
        self._Y = np.zeros((0, self.C), np.float32)
        self._Y_start = 0
        self._next_det = 0
        self.timing = dict(blank=0.0, filter=0.0, whiten=0.0, detect=0.0, classify=0.0)
        self.n_blanked = 0

    def start_at(self, sample: int) -> None:
        """Reset and declare the global index of the first sample to be pushed."""
        self.reset()
        self._pend_start = self._Y_start = self._next_det = int(sample)
        if self.suppressor is not None:
            self.suppressor.reset(int(sample))

    # -- public API -----------------------------------------------------------
    def push(self, raw: np.ndarray) -> SpikeBatch:
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != self.C:
            raise ValueError(f"expected (n, {self.C}) buffer, got {raw.shape}")
        if self.suppressor is not None:
            t0 = time.perf_counter()
            raw = self.suppressor.push(raw)
            self.timing["suppress"] = self.timing.get("suppress", 0.0) + time.perf_counter() - t0
            if not len(raw):
                return SpikeBatch.empty()
        self._pending = np.concatenate([self._pending, raw]) if len(self._pending) else np.array(raw)
        n_release = len(self._pending) - self.lag
        if n_release <= 0:
            return SpikeBatch.empty()
        return self._release(n_release)

    def flush(self) -> SpikeBatch:
        """Process the look-ahead tail and finalise every decodable sample."""
        out = []
        if self.suppressor is not None:
            tail = self.suppressor.flush()
            if len(tail):
                self._pending = np.concatenate([self._pending, tail]) if len(self._pending) else tail
        if len(self._pending):
            out.append(self._release(len(self._pending), final=True))
        out.append(self._detect_and_classify(final=True))
        return SpikeBatch.concat(out)

    # -- internals ------------------------------------------------------------
    def _release(self, n: int, final: bool = False) -> SpikeBatch:
        t0 = time.perf_counter()
        pend, g0 = self._pending, self._pend_start
        clipped = (np.abs(pend) >= self.p.clip_counts).sum(axis=1) >= self.p.clip_min_channels
        mask = np.zeros(len(pend), bool)
        if self._blank_until > g0:
            mask[:min(self._blank_until - g0, len(pend))] = True
        if clipped.any():
            idx = np.flatnonzero(clipped)
            diff = np.zeros(len(pend) + 1, np.int64)
            np.add.at(diff, np.maximum(idx - self.bpre, 0), 1)
            np.add.at(diff, np.minimum(idx + self.bpost + 1, len(pend)), -1)
            mask |= np.cumsum(diff[:-1]) > 0
            self._blank_until = max(self._blank_until, g0 + int(idx[-1]) + self.bpost + 1)
        rel = pend[:n].astype(np.float32)
        m = mask[:n]
        if m.any():
            self.n_blanked += int(m.sum())
            clean = ~m
            last = np.maximum.accumulate(np.where(clean, np.arange(n), -1))
            if self._hold is not None:
                hold = self._hold
            elif clean.any():
                hold = rel[np.argmax(clean)]
            else:
                hold = np.zeros(self.C, np.float32)
            filled = np.where(last[:, None] >= 0, rel[np.clip(last, 0, None)], hold[None, :])
            rel = np.where(m[:, None], filled, rel)
        if (~m).any():
            self._hold = rel[np.flatnonzero(~m)[-1]].copy()
        self._pending = pend[n:]
        self._pend_start = g0 + n
        self.timing["blank"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.use_numba:
            if not self._primed:   # same steady-state priming as CausalPreprocessor.prime
                x0 = rel[0].astype(np.float64) * self.cfg.bit_volts
                self._zi = self.pre._zi_unit[:, :, None] * x0[None, None, :]
                self._primed = True
            y = kernels.median_reference(kernels.sos_filter(rel, self.cfg.bit_volts, self.sos, self._zi))
        else:
            y = self.pre.process(rel, scale=True)
        self.timing["filter"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        y = y @ self.WT
        self.timing["whiten"] += time.perf_counter() - t0
        self._Y = np.concatenate([self._Y, y]) if len(self._Y) else y
        return self._detect_and_classify(final=final)

    def _detect_and_classify(self, final: bool = False) -> SpikeBatch:
        Y, Ys = self._Y, self._Y_start
        end = Ys + len(Y)
        fin_hi = end - self.after_need - (0 if final else self.dedup)
        fin_lo = self._next_det
        if fin_hi <= fin_lo:
            return SpikeBatch.empty()
        t0 = time.perf_counter()
        lo = max(fin_lo - self.dedup, Ys)
        hi = min(fin_hi + self.dedup, end)
        g_all, c_all = [], []
        if self.use_numba:
            a_rel = max(fin_lo, Ys + self.before_need) - Ys
            b_rel = fin_hi - Ys
            if b_rel > a_rel:
                mask = kernels.detect_minima(Y, self.thr, self.nbr, self.dedup, a_rel, b_rel)
                tt, cc = np.nonzero(mask)
                g_all.append(tt + Ys); c_all.append(cc)
        BLK = 20_000                        # bounds Y[:, nbr] to ~0.6 GB worst case
        for b0 in (range(lo, hi, BLK) if not self.use_numba else ()):
            a = max(b0 - self.dedup, lo); b = min(b0 + BLK + self.dedup, hi)
            seg = Y[a - Ys:b - Ys]
            smin = seg[:, self.nbr].min(axis=2)
            stmin = minimum_filter1d(smin, size=2 * self.dedup + 1, axis=0, mode="nearest")
            tt, cc = np.nonzero((seg <= stmin) & (seg < -self.thr[None, :]))
            g = tt + a
            k = (g >= max(b0, fin_lo)) & (g < min(b0 + BLK, fin_hi)) & (g - self.before_need >= Ys)
            g_all.append(g[k]); c_all.append(cc[k])
        g = np.concatenate(g_all); cc = np.concatenate(c_all)
        self.timing["detect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        batch = self._classify(g, cc)
        self.timing["classify"] += time.perf_counter() - t0

        self._next_det = fin_hi
        keep_from = max(fin_hi - self.before_need - self.dedup - 1, Ys)
        self._Y = Y[keep_from - Ys:]
        self._Y_start = keep_from
        return batch

    def _classify(self, g: np.ndarray, cc: np.ndarray) -> SpikeBatch:
        E = len(g)
        if E == 0:
            return SpikeBatch.empty()
        Y, Ys = self._Y, self._Y_start
        sh, nS, T = self.shifts, len(self.shifts), self.T
        ar = np.arange(T)
        best_u = np.full(E, -1, np.int64); best_sc = np.zeros(E, np.float32)
        best_a = np.zeros(E, np.float32); best_s = np.zeros(E, np.int64)
        if self.use_numba:
            bj, best_sc, best_a, best_s = kernels.score_events(
                Y, (g - Ys).astype(np.int64), cc.astype(np.int64), self.cand_ptr, self.tro_cand,
                self.sset64, self.Tcat, self.ttcat, sh.astype(np.int64),
                float(self.p.amp_range[0]), float(self.p.amp_range[1]))
            best_u = np.where(bj >= 0, self.cand_units[np.clip(bj, 0, None)], -1)
        for c in (np.unique(cc) if not self.use_numba else ()):
            idx_all = np.flatnonzero(cc == c)
            cu = self.cand[c]
            if not len(cu):
                continue
            Ysub = Y[:, self.sset[c]]
            T_, tt = self.Tc[c], self.ttc[c]
            tro = self.tro[cu]
            per = len(cu) * nS * T * T_.shape[2]
            B = max(1, self.p.max_block_floats // per)
            for b0 in range(0, len(idx_all), B):
                idx = idx_all[b0:b0 + B]
                st = (g[idx] - Ys)[:, None, None] - tro[None, :, None] + sh[None, None, :]
                X = Ysub[np.clip(st[..., None] + ar, 0, len(Y) - 1)]           # (E,n,S,T,K)
                xt = np.einsum("enstk,ntk->ens", X, T_, optimize=True)
                a = xt / tt[None, :, None]
                sc = np.where((a >= self.p.amp_range[0]) & (a <= self.p.amp_range[1]), xt * a, -np.inf)
                flat = sc.reshape(len(idx), -1)
                arg = flat.argmax(axis=1)
                v = flat[np.arange(len(idx)), arg]
                okv = np.isfinite(v)
                best_sc[idx] = np.where(okv, v, 0.0)
                best_u[idx] = np.where(okv, cu[arg // nS], -1)
                best_a[idx] = a.reshape(len(idx), -1)[np.arange(len(idx)), arg]
                best_s[idx] = sh[arg % nS]
        assigned = (best_u >= 0) & (best_sc >= self.p.min_energy)
        uu = np.clip(best_u, 0, None)
        unit = np.where(assigned, self.bank.unit_ids[uu], -1)
        tro = np.where(assigned, self.tro[uu], self.cfg.n_before)
        frame = g + np.where(assigned, best_s, 0) - tro + self.cfg.n_before
        return SpikeBatch(frame.astype(np.int64), unit.astype(np.int64),
                          best_sc.astype(np.float32), best_a.astype(np.float32), cc.astype(np.int64))


def decode_window(dec: OnlineDecoder, src, start: int, n: int,
                  warmup_ms: float = 15.0, tail_ms: float = 4.0, condition=None) -> SpikeBatch:
    """Pull-mode helper: decode samples [start, start+n) from a file source.

    Prepends ``warmup_ms`` so the causal filter has settled (validated: 15 ms
    gives < 1e-6 relative error) and appends ``tail_ms`` so spikes near the
    window end keep their full waveform.  Returns spikes with frames in
    [start, start+n).
    """
    W = dec.cfg.samples(warmup_ms); Tl = dec.cfg.samples(tail_ms)
    s0 = max(int(start) - W, 0)
    dec.start_at(s0)
    if dec.suppressor is not None:
        dec.suppressor.set_condition(condition)
    raw = np.asarray(src.read(s0, int(start) + n + Tl - s0))
    b = SpikeBatch.concat([dec.push(raw), dec.flush()])
    return b.select((b.frame >= start) & (b.frame < start + n))
