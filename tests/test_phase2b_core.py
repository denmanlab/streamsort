"""Phase 2b: C++ decoder core (streamsort_core) vs the Python decoder; table export; pulse pairing."""
import sys
from pathlib import Path
import numpy as np, pytest
from conftest import needs_data, needs_results, ROOT_REPO

REPO = Path(ROOT_REPO)
core = pytest.importorskip("streamsort.core_cpp")


def _H():
    sys.path.insert(0, str(REPO / "scripts"))
    import replay_harness as H
    return H


def _run(cd, src, s0, n, pts, buf):
    from streamsort.online.decoder import SpikeBatch
    edges = [0] + [s for s, _ in pts] + [n]; kk = [None] + [k for _, k in pts]; out = []
    cd.start_at(0)
    for a, b, key in zip(edges[:-1], edges[1:], kk):
        if key is not None:
            cd.set_condition(key)
        for c in range(a, b, buf):
            out.append(cd.push(np.asarray(src.read(s0 + c, min(buf, b - c))).astype(np.float32)))
    out.append(cd.flush()); bb = SpikeBatch.concat(out); m = bb.unit >= 0
    return bb.frame[m], bb.unit[m]


@needs_data
@needs_results
@pytest.mark.parametrize("threads,buf", [(1, 1024), (2, 512)])
def test_cpp_core_identical_to_python(threads, buf):
    H = _H()
    sys.path.insert(0, str(REPO / "scripts")); import replay_harness_phase2b as P
    from streamsort.io import FileStreamSource
    st, dec = P.py_decoder("A"); D = 6.0; pts = H.cond_points(st, H.T0, D)
    ref = H.reference("A", H.T0, D, pts)
    s0 = int(st.time_to_sample(np.array([H.T0]))[0])
    got = _run(core.CoreDecoder(core.tables_from_decoder(dec), threads), FileStreamSource(st, H.NCH), s0, int(D * st.effective_rate), pts, buf)
    assert H.compare(got, ref)["identical"]


@needs_data
@needs_results
def test_exported_tables_roundtrip(tmp_path):
    sys.path.insert(0, str(REPO / "scripts")); import replay_harness_phase2b as P
    _, dec = P.py_decoder("A"); t = core.tables_from_decoder(dec)
    core.export_tables(t, tmp_path, extra=dict(sample_rate=29970.0757, n_units=1, n_usable=1))
    t2 = core.load_exported(tmp_path)
    for k in core.ARRAYS:
        assert np.array_equal(np.asarray(t[k]), np.asarray(t2[k])), k
    C, nu, nus, fs, _ = core._module().load_tables_check(str(tmp_path))
    assert C == t["C"] and abs(fs - 29970.0757) < 1e-3
    assert not list(tmp_path.glob("*.tmp*"))


def test_pulse_pairer_rules():
    m = core._module()
    p = m.PulsePairer(True, 0.05)
    p.add_ttl(0, 10.000); p.add_artifact(300, 10.010)
    assert p.pair(10.02) == [(300, 10.010, 0)]
    p.add_artifact(9000, 11.0)                       # clip without TTL ...
    assert p.pair(11.01) == [] and p.pair(11.2) == []  # ... dropped after tol
    p.add_ttl(1, 12.0)                               # TTL without clip -> emitted alone after tol
    assert p.pair(12.01) == [] and p.pair(12.2) == [(-1, 12.0, 1)]
    q = m.PulsePairer(False, 0.05); q.add_artifact(5, 1.0)
    assert q.pair(1.0) == [(5, 1.0, -1)]


def test_pool_threads_do_not_change_results_on_synthetic_noise():
    rng = np.random.default_rng(1); C = 8
    t = dict(C=C, bit_volts=0.195, sos=np.array([[1., 0, 0, 1., 0, 0]]), zi_unit=np.zeros((1, 2)),
             w_ptr=np.arange(C + 1, dtype=np.int64), w_row=np.arange(C, dtype=np.int64), w_val=np.ones(C, np.float32),
             thr=np.full(C, 3.0, np.float32), nbr=np.tile(np.arange(C), (C, 1)).astype(np.int64), K=C, dedup=5, before_need=12,
             after_need=12, lag=30, bpre=6, bpost=15, clip_counts=6190, clip_min_channels=3, T=10, Ks=4, n_before=5,
             cand_ptr=np.arange(C + 1, dtype=np.int64), cand_units=np.arange(C, dtype=np.int64), tro_cand=np.full(C, 5, np.int64),
             Tcat=-np.abs(rng.standard_normal((C, 10, 4))).astype(np.float32), ttcat=np.ones(C, np.float32),
             sset=np.tile(np.arange(4), (C, 1)).astype(np.int64), shifts=np.arange(-2, 3, dtype=np.int64), amp_lo=0.5, amp_hi=1.8,
             min_energy=0.0, unit_ids=np.arange(C, dtype=np.int64), has_suppressor=False, sup_taper=np.zeros(0, np.float32))
    t["ttcat"] = (t["Tcat"] ** 2).sum(axis=(1, 2)).astype(np.float32)
    x = (rng.standard_normal((20000, C)) * 30).astype(np.float32)
    outs = []
    for nt in (1, 3):
        d = core.CoreDecoder(t, nt); d.start_at(0); b = [d.push(x[i:i + 700]) for i in range(0, len(x), 700)] + [d.flush()]
        outs.append(np.concatenate([q.frame for q in b]))
    assert len(outs[0]) > 0 and np.array_equal(outs[0], outs[1])
