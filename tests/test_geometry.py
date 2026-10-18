"""Geometry metrics: identities, invariances and bounds."""
import numpy as np, pytest
from streamsort.geometry import (bin_trial, make_reducer, procrustes_distance, procrustes_disparity,
                                 principal_angles_deg, alignment_index, projection_score)
rng = np.random.default_rng(0)

def rot(k):
    q, _ = np.linalg.qr(rng.normal(size=(k, k))); return q

def test_procrustes_identity_and_invariance():
    A = rng.normal(size=(50, 3))
    assert procrustes_distance(A, A) == pytest.approx(0, abs=1e-12)
    B = 2.5 * (A @ rot(3)) + 7.0                     # rotate, scale, translate
    assert procrustes_distance(A, B, scale=True) == pytest.approx(0, abs=1e-10)
    assert procrustes_disparity(A, B) == pytest.approx(0, abs=1e-10)
    assert procrustes_distance(A, B, scale=False) > 0.1   # amplitude counts when scale=False

def test_procrustes_unrelated_is_large():
    A, B = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    assert 0.8 < procrustes_distance(A, B) <= 1.0 + 1e-9
    assert 0 <= procrustes_disparity(A, B) <= 1

def test_subspace_metrics():
    X = rng.normal(size=(400, 20)) @ np.diag(np.r_[10, 8, 6, np.ones(17)])
    V = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:3].T
    assert np.allclose(principal_angles_deg(V, V), 0, atol=1e-5)
    assert alignment_index(X, V) == pytest.approx(1.0, abs=1e-9)
    Vr = np.linalg.qr(rng.normal(size=(20, 3)))[0]
    assert alignment_index(X, Vr) < 0.5
    ps = projection_score(X, V); assert 0 < ps <= 1

def test_bin_trial_counts_and_smoothing_preserve_mass():
    fs = 30000.0; units = np.array([3, 7, 9])
    fr = np.sort(rng.integers(1000, 1000 + 15000, 120)); un = rng.choice(units, 120)
    X = bin_trial(fr, un, units, 1000, 15000, fs, bin_ms=10)
    assert X.shape == (50, 3) and X.sum() == 120
    Xs = bin_trial(fr, un, units, 1000, 15000, fs, bin_ms=10, smooth_ms=20)
    assert Xs.sum() == pytest.approx(120, rel=0.05)

@pytest.mark.parametrize("name", ["pca", "fa", "isomap"])
def test_reducers_swappable(name):
    X = rng.normal(size=(60, 15))
    Z = make_reducer(name, 3).fit_transform(X)
    assert Z.shape == (60, 3) and np.isfinite(Z).all()

def test_pca_basis_orthonormal():
    r = make_reducer("pca", 3).fit(rng.normal(size=(80, 12)))
    assert np.allclose(r.basis.T @ r.basis, np.eye(3), atol=1e-10)
