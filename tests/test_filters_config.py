"""Chunked causal filtering equals whole-block filtering; config fingerprint semantics."""
import dataclasses, numpy as np
from streamsort.config import StreamConfig
from streamsort.preprocess import CausalPreprocessor

def test_chunked_equals_whole():
    cfg = StreamConfig(sample_rate=30000.0, bit_volts=0.195)
    x = (np.random.default_rng(2).normal(0, 50, size=(20000, 24))).astype(np.int16)
    p = CausalPreprocessor(cfg, 24); p.prime(x[0]); whole = p.process(x)
    q = CausalPreprocessor(cfg, 24); q.prime(x[0])
    cuts = np.r_[0, 7, 1024, 1500, 9000, 9001, 15000, 20000]
    chunked = np.concatenate([q.process(x[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    assert np.array_equal(whole, chunked)

def test_fingerprint_stable_and_sensitive():
    a = StreamConfig(sample_rate=30000.0, bit_volts=0.195)
    assert a.fingerprint() == StreamConfig(sample_rate=30000.0, bit_volts=0.195).fingerprint()
    assert a.fingerprint() != dataclasses.replace(a, highpass_hz=a.highpass_hz + 1).fingerprint()
