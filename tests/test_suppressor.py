"""ArtifactSuppressor: causality, chunk invariance, bridging, template subtraction (synthetic)."""
import numpy as np, pytest
from streamsort.preprocess import ArtifactSuppressor
FS, C = 30000.0, 16
rng = np.random.default_rng(1)

def synth(n_events=12, gap=6000, seed=1):
    r = np.random.default_rng(seed)
    n = gap * (n_events + 1); x = r.normal(0, 20, size=(n, C)).astype(np.float32)
    tail = (np.exp(-np.arange(600) / 120.0)[:, None] * r.uniform(800, 2000, C)[None]).astype(np.float32)
    onsets = []
    for e in range(n_events):
        on = gap * (e + 1) + int(r.integers(0, 3)); onsets.append(on)
        x[on:on + 8, :6] = 6369                         # rail-clipped samples on 6 channels
        x[on + 8:on + 608] += tail
    return x, onsets

def run(x, onsets, chunks, cond=("a",)):
    sup = ArtifactSuppressor(FS, C); sup.reset(0); out = []
    # the stimulus controller announces each pulse 200 samples ahead; buffers are split there
    ann = [o - 200 for o in onsets]
    edges = sorted(set([0, len(x)] + [int(c) for c in chunks] + ann))
    for a, b in zip(edges[:-1], edges[1:]):
        if a in ann:
            sup.set_condition(cond)
        out.append(sup.push(x[a:b]))
    out.append(sup.flush())
    return np.concatenate(out), sup

def test_length_and_chunk_invariance():
    x, on = synth()
    a, _ = run(x, on, [])
    b, _ = run(x, on, rng.choice(np.arange(1, len(x)), 300, replace=False))
    assert len(a) == len(x) and np.array_equal(a, b)

def test_bridge_removes_clipping_and_is_linear():
    x, on = synth()
    y, sup = run(x, on, [])
    assert (np.abs(y) >= 6190).sum() == 0
    e = sup.events[0]; b0 = e["onset"] - sup.pre; b1 = e["onset"] + e["span"] + sup.post
    seg = y[b0:b1, 0]; assert np.allclose(np.diff(seg, 2), 0, atol=1e-2)   # straight line

def test_subtraction_starts_after_min_history_and_reduces_tail():
    x, on = synth()
    y, sup = run(x, on, [])
    flags = [e["subtracted"] for e in sup.events]
    assert flags[:5] == [False] * 5 and all(flags[5:])
    res = lambda k: np.abs(y[on[k] + 40:on[k] + 400, 6:]).mean()
    assert res(10) < 0.5 * res(2)

def test_causal_future_events_do_not_change_past_output():
    x, on = synth()
    full, _ = run(x, on, [])
    cut = on[7] - 500
    part, _ = run(x[:cut], [o for o in on if o < cut], [])
    assert np.array_equal(part[:cut - 200], full[:cut - 200])

def test_condition_none_bridges_but_keeps_no_history():
    x, on = synth()
    sup = ArtifactSuppressor(FS, C); sup.reset(0)
    y = np.concatenate([sup.push(x), sup.flush()])
    assert (np.abs(y) >= 6190).sum() == 0 and len(sup.history) == 0
