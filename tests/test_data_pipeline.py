"""End-to-end invariants on the real recording (skipped when the data are not mounted)."""
import os, numpy as np, pytest
from conftest import needs_data, needs_results, RESULTS
pytestmark = [needs_data, needs_results]

def test_time_map_round_trip(stream):
    i = np.array([0, 12345, 10_000_000, stream.n_samples - 1]) if hasattr(stream, "n_samples") else np.array([0, 12345, 10_000_000])
    assert np.array_equal(stream.time_to_sample(stream.sample_to_time(i)), i)

def test_live_stream_refuses_unwritten_samples(src):
    from streamsort.io import SimulatedLiveStream, NotYetWritten
    live = SimulatedLiveStream(src, head=100_000)
    live.read(99_000, 1000)
    with pytest.raises(NotYetWritten):
        live.read(99_500, 1000)

def _dec(cfg, decoder_parts, sup=None, numba=True):
    from streamsort.online import OnlineDecoder, DecoderParams
    bank, usable, wh = decoder_parts
    return OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False,
                         use_numba=numba, suppressor=sup)

def _window(cfg, trials, k, stream):
    off = int(stream.time_to_sample(np.array([trials.ttl_offset[k]]))[0])
    S = lambda ms: int(round(ms * 1e-3 * cfg.sample_rate)); return off, S

def test_push_equals_pull_with_suppressor(cfg, src, trials, stream, decoder_parts):
    from streamsort.online import decode_window, SpikeBatch
    from streamsort.preprocess import ArtifactSuppressor
    k = 1500; off, S = _window(cfg, trials, k, stream); key = (int(trials.amplitude[k]), trials.polarity[k], int(trials.pulse_duration[k]))
    w0, n = off + S(1), S(400)
    d1 = _dec(cfg, decoder_parts, ArtifactSuppressor(cfg.sample_rate, 200)); a = decode_window(d1, src, w0, n, condition=key)
    d2 = _dec(cfg, decoder_parts, ArtifactSuppressor(cfg.sample_rate, 200))
    s0 = w0 - S(15); d2.start_at(s0); d2.suppressor.set_condition(key)
    raw = np.asarray(src.read(s0, w0 + n + S(4) - s0)); rng = np.random.default_rng(3)
    cuts = np.r_[0, np.sort(rng.choice(np.arange(1, len(raw)), 60, replace=False)), len(raw)]
    b = SpikeBatch.concat([d2.push(raw[x:y]) for x, y in zip(cuts[:-1], cuts[1:])] + [d2.flush()])
    b = b.select((b.frame >= w0) & (b.frame < w0 + n))
    assert np.array_equal(a.frame, b.frame) and np.array_equal(a.unit, b.unit)

def test_numba_equals_numpy(cfg, src, trials, stream, decoder_parts):
    from streamsort.online import decode_window
    k = 900; off, S = _window(cfg, trials, k, stream)
    a = decode_window(_dec(cfg, decoder_parts, numba=True), src, off + S(500), S(300))
    b = decode_window(_dec(cfg, decoder_parts, numba=False), src, off + S(500), S(300))
    assert np.array_equal(a.frame, b.frame) and np.array_equal(a.unit, b.unit)

@pytest.mark.skipif(not os.path.isfile(os.path.join(RESULTS, "online_spikes_all_trials_sup.npz")), reason="validation output absent")
def test_reproduces_saved_validation(cfg, src, trials, stream, decoder_parts):
    """Re-decoding (with fresh suppressor history) reproduces the saved run for a trial with no subtraction yet."""
    from streamsort.online import decode_window
    from streamsort.preprocess import ArtifactSuppressor
    v = np.load(os.path.join(RESULTS, "online_spikes_all_trials_sup.npz"))
    k = 0; off = int(v["meta_pulse_offset"][k]); w0 = int(v["meta_window_start"][k]); n = int(v["meta_window_len"])
    key = (int(trials.amplitude[k]), trials.polarity[k], int(trials.pulse_duration[k]))
    b = decode_window(_dec(cfg, decoder_parts, ArtifactSuppressor(cfg.sample_rate, 200)), src, w0, n, condition=key)
    m = v["trial"] == k; b = b.select(b.unit >= 0)
    assert set(zip(b.frame.tolist(), b.unit.tolist())) == set(zip(v["frame"][m].tolist(), v["unit"][m].tolist()))
