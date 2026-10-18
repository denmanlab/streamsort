"""Spatial noise whitening for the online decoder.

After common-median referencing, noise on neighbouring Neuropixels sites stays
correlated (median r = 0.31 with the nearest site on jlh32).  Scoring against
templates as if channels were independent therefore counts shared noise as
signal; with perfect detection, unit assignment rose from 0.68 to 0.82 once
the data were whitened.

The whitening is Kilosort-style and *local*: output channel c is a linear
combination of its ``k`` nearest channels, given by c's row of
``inv_sqrt(Sigma_local)``.  It is linear, so templates whiten with the same
matrix (no bank rebuild), and it is estimated once, offline, from stimulus-free
data on the same causal signal path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import StreamConfig
from ..preprocess.filters import CausalPreprocessor


def _inv_sqrt(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return (V / np.sqrt(np.maximum(w, 1e-9))) @ V.T


@dataclass
class Whitening:
    W: np.ndarray          # (C, C) float32; whitened = Y @ W.T
    noise_sd: np.ndarray   # (C,) robust sd of whitened noise
    k: int
    fingerprint: str       # StreamConfig fingerprint of the signal path

    def save(self, path) -> Path:
        path = Path(path); tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, W=self.W, noise_sd=self.noise_sd,
                            meta=np.array(json.dumps({"k": self.k, "fingerprint": self.fingerprint})))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path) -> "Whitening":
        z = np.load(Path(path), allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return cls(W=z["W"], noise_sd=z["noise_sd"], k=int(m["k"]), fingerprint=m["fingerprint"])


def estimate_whitening(src, cfg: StreamConfig, channel_locations: np.ndarray,
                       start: int, n: int, k: int = 32, chunk_s: float = 1.0,
                       skip_ms: float = 50.0) -> Whitening:
    """Estimate local ZCA whitening from stimulus-free data, in bounded memory.

    The covariance is accumulated chunk by chunk (never materialising the full
    sample matrix); the whitened noise level is then measured on the same span.
    """
    C = src.n_channels_used
    pp = CausalPreprocessor(cfg, C)
    step = cfg.samples(chunk_s * 1e3); sk = cfg.samples(skip_ms)
    S = np.zeros((C, C)); mu = np.zeros(C); cnt = 0
    for s in range(0, n, step):
        y = pp.process(np.asarray(src.read(start + s, min(step, n - s)))).astype(np.float64)
        if s == 0:
            y = y[sk:]
        S += y.T @ y; mu += y.sum(0); cnt += len(y)
    mu /= cnt
    Sig = S / cnt - np.outer(mu, mu)
    pos = np.asarray(channel_locations[:C], dtype=np.float64)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    W = np.zeros((C, C))
    for c in range(C):
        nb = np.argsort(D[c], kind="stable")[:k]
        W[c, nb] = _inv_sqrt(Sig[np.ix_(nb, nb)])[int(np.flatnonzero(nb == c)[0])]
    W = W.astype(np.float32)
    # robust sd of whitened noise, second pass in chunks
    pp.reset()
    med_abs = []
    for s in range(0, n, step):
        y = pp.process(np.asarray(src.read(start + s, min(step, n - s))))
        if s == 0:
            y = y[sk:]
        yw = y @ W.T
        med_abs.append(np.median(np.abs(yw - np.median(yw, 0)), 0))
    noise_sd = (np.median(np.stack(med_abs), 0) / 0.6744897501960817).astype(np.float32)
    return Whitening(W=W, noise_sd=noise_sd, k=k, fingerprint=cfg.fingerprint())
