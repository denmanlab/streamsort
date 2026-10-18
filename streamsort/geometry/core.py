"""Population geometry from decoded spikes: binning, swappable reducers, metrics.

Data convention: a trial's population activity is a matrix ``X`` of shape
``(n_bins, n_units)`` -- observations are time bins, features are neurons --
so a reducer maps it to a trajectory ``Z`` of shape ``(n_bins, k)``.

Reducers share one interface (:class:`BaseReducer`) so they can be swapped by
name; geometry can be computed either in a **fixed basis** (fit once on
reference data, project every trial -- stable for sparse single trials) or by
**per-trial refit** (each trial defines its own subspace).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import orthogonal_procrustes, subspace_angles
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import procrustes as _scipy_procrustes


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------
def bin_trial(frames: np.ndarray, units: np.ndarray, unit_ids: np.ndarray,
              window_start: int, n_samples: int, fs: float,
              bin_ms: float = 10.0, smooth_ms: float = 0.0) -> np.ndarray:
    """Spike counts (n_bins, n_units) for one trial, optional Gaussian smoothing.

    Parameters are in samples (frames, window) and ms (bin, smoothing), so the
    result does not depend on how the caller stores time.
    """
    bin_n = bin_ms * 1e-3 * fs
    n_bins = int(np.floor(n_samples / bin_n))
    col = {int(u): i for i, u in enumerate(unit_ids)}
    X = np.zeros((n_bins, len(unit_ids)), np.float32)
    rel = frames - window_start
    ok = (rel >= 0) & (rel < n_bins * bin_n)
    if ok.any():
        b = (rel[ok] / bin_n).astype(np.int64)
        c = np.array([col.get(int(u), -1) for u in units[ok]])
        keep = c >= 0
        np.add.at(X, (b[keep], c[keep]), 1.0)
    if smooth_ms > 0:
        X = gaussian_filter1d(X, sigma=smooth_ms / bin_ms, axis=0, mode="nearest")
    return X


# --------------------------------------------------------------------------
# reducers
# --------------------------------------------------------------------------
class BaseReducer:
    """fit(X) -> self; transform(X) -> Z.  X is (n_obs, n_units)."""
    name = "base"

    def __init__(self, k: int = 3):
        self.k = int(k)

    def fit(self, X: np.ndarray) -> "BaseReducer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def basis(self) -> np.ndarray:
        """(n_units, k) orthonormal basis of the reduced space (if linear)."""
        raise NotImplementedError


class PCAReducer(BaseReducer):
    name = "pca"

    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = min(self.k, Vt.shape[0])
        self.components_ = Vt[:k]                       # (k, n_units)
        var = S ** 2 / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = var[:k] / max(var.sum(), 1e-12)
        self.all_variance_ratio_ = var / max(var.sum(), 1e-12)
        return self

    def transform(self, X):
        return (np.asarray(X, np.float64) - self.mean_) @ self.components_.T

    @property
    def basis(self):
        return self.components_.T


class FAReducer(BaseReducer):
    """Factor analysis: separates shared variance from per-neuron noise --
    appropriate for sparse spike counts where private Poisson noise is large."""
    name = "fa"

    def fit(self, X):
        from sklearn.decomposition import FactorAnalysis
        X = np.asarray(X, np.float64)
        self.model_ = FactorAnalysis(n_components=self.k, random_state=0).fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, np.float64))

    @property
    def basis(self):
        Q, _ = np.linalg.qr(self.model_.components_.T)   # orthonormalised loadings
        return Q


class IsomapReducer(BaseReducer):
    """Non-linear option.  Has no linear basis, so only trajectory metrics apply."""
    name = "isomap"

    def __init__(self, k: int = 3, n_neighbors: int = 10):
        super().__init__(k)
        self.n_neighbors = n_neighbors

    def fit(self, X):
        from sklearn.manifold import Isomap
        X = np.asarray(X, np.float64)
        self.model_ = Isomap(n_components=self.k, n_neighbors=min(self.n_neighbors, len(X) - 1)).fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, np.float64))


REDUCERS = {"pca": PCAReducer, "fa": FAReducer, "isomap": IsomapReducer}


def make_reducer(name: str, k: int = 3, **kw) -> BaseReducer:
    """Construct a reducer by name: 'pca' | 'fa' | 'isomap'."""
    return REDUCERS[name](k=k, **kw)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> float:
    """Scale-free Procrustes disparity (scipy): 0 = identical shape, 1 = maximal.

    Both trajectories are centred and scaled to unit Frobenius norm before the
    optimal rotation/reflection is found.  Insensitive to overall amplitude.
    """
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    if np.allclose(A.std(), 0) or np.allclose(B.std(), 0):
        return float("nan")
    return float(_scipy_procrustes(A, B)[2])


def procrustes_distance(A: np.ndarray, B: np.ndarray, scale: bool = True) -> float:
    """Orthogonal Procrustes residual after centring.

    With ``scale=True`` the best isotropic scale is also fitted, and the
    residual is reported relative to ``||A||`` (amplitude-normalised but not
    norm-equalised, unlike :func:`procrustes_disparity`).  With ``scale=False``
    differences in amplitude count as differences in geometry.
    """
    A = np.asarray(A, np.float64) - np.mean(A, 0)
    B = np.asarray(B, np.float64) - np.mean(B, 0)
    R, s = orthogonal_procrustes(B, A)
    Bt = B @ R
    if scale:
        nb = (Bt * Bt).sum()
        Bt = Bt * ((Bt * A).sum() / nb if nb > 0 else 0.0)
    na = np.linalg.norm(A)
    return float(np.linalg.norm(A - Bt) / na) if na > 0 else float("nan")


def principal_angles_deg(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between column spaces of two bases."""
    return np.degrees(subspace_angles(Va, Vb))


def alignment_index(X_b: np.ndarray, basis_a: np.ndarray) -> float:
    """Fraction of X_b's variance captured by subspace A, normalised by the
    best possible k-dimensional capture of X_b (Elsayed et al. 2016).
    1 = A is X_b's own top-k subspace; ~k/n_units for a random subspace."""
    Xc = np.asarray(X_b, np.float64) - np.mean(X_b, 0)
    C = Xc.T @ Xc
    k = basis_a.shape[1]
    num = np.trace(basis_a.T @ C @ basis_a)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1][:k]
    return float(num / ev.sum()) if ev.sum() > 0 else float("nan")


def projection_score(X_trial: np.ndarray, basis: np.ndarray, mean: np.ndarray | None = None) -> float:
    """Fraction of a single trial's (centred) variance lying in ``basis``."""
    X = np.asarray(X_trial, np.float64)
    Xc = X - (mean if mean is not None else X.mean(0))
    tot = (Xc * Xc).sum()
    if tot <= 0:
        return float("nan")
    P = Xc @ basis
    return float((P * P).sum() / tot)


@dataclass
class TrialGeometry:
    trajectory: np.ndarray       # (n_bins, k)
    basis: np.ndarray | None     # (n_units, k) if linear
    n_spikes: int
    seconds: float               # wall time to compute
