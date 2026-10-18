"""Phase 2b: plugin message protocol, worker v2 engine (pooling, analyses, deadline hold), ZMQ inproc."""
import time
import numpy as np, pytest

zmq = pytest.importorskip("zmq")
from streamsort.worker2 import protocol  # noqa: E402
from streamsort.worker2.engine import Engine, ZmqWorker  # noqa: E402
from streamsort.worker2.analyses import ImageDecoderAnalysis, export_linear_readout  # noqa: E402


def test_protocol_roundtrip():
    env, h, d = protocol.decode(protocol.spikes("S", 1, 7, 1234, 30000.0, [5, 9], [3, 4]))
    assert env == "SPIKES" and h["head"] == 1234 and d["frames"].tolist() == [5, 9] and d["units"].tolist() == [3, 4]
    env, h, d = protocol.decode(protocol.spikes("S", 1, 8, 2000, 30000.0, [], []))
    assert h["n"] == 0 and len(d["frames"]) == 0
    assert protocol.decode(protocol.hello("S", 1, 30000.0, 200, [1, 2]))[1]["unit_ids"] == [1, 2]
    assert protocol.decode(protocol.pulse("S", -1, 4))[1]["approx"] is True
    assert protocol.decode(protocol.condition(3))[1] == {"type": "condition", "id": 3}


def _feed(e, streams, n_trials, fs=30000.0, iti=2.0, ttl_first=True, rng=None, gap_s=0.03):
    rng = rng or np.random.default_rng(0)
    for s in streams:
        e.on_message("STATUS", dict(type="hello", stream=s, sample_rate=fs, unit_ids=list(range(5))), {})
    for k in range(n_trials):
        on = int((1 + k * iti) * fs)
        if ttl_first:
            e.on_message("EVENT", dict(type="event", ttl_index=k, t_wall_us=protocol.wall_us()), {})
        for s in streams:
            e.on_message("PULSE", dict(type="pulse", stream=s, onset_sample=on + (3 if s == "B" else 0), ttl_index=k, t_wall_us=protocol.wall_us()), {})
        for s in streams:
            cls = (k % 2)
            f = np.sort(rng.integers(on, on + int(0.6 * fs), 300)); u = rng.integers(0, 3, 300) + 2 * cls
            e.on_message("SPIKES", dict(type="spikes", stream=s, head=on + int(0.6 * fs)), dict(frames=f, units=u))
        time.sleep(gap_s)                       # let the stimulator thread pick the next condition (ITI on the rig)


def test_engine_pools_streams_and_follows_schedule():
    sent = []
    e = Engine(dict(analysis_kw=dict(n_fit=4), stimulator_kw=dict(reprog_ms=0.0), conditions=[dict(a=0), dict(a=1)], schedule=[0, 1]),
               send_condition=sent.append)
    e.arm(); _feed(e, ["A", "B"], 8); time.sleep(0.2); e.close()
    assert len(e.rows) == 8 and all(r["n_streams_pulse"] == 2 and r["n_units"] == 10 for r in e.rows)
    assert [r["cond"] for r in e.rows] == [0, 1] * 4 and sum(r["loss"] is not None for r in e.rows) == 4
    assert sent[0] == 0 and all(r["ref"] == "ttl" for r in e.rows)


def test_engine_holds_when_deadline_cannot_be_met():
    e = Engine(dict(iti_s=0.05, stimulator_kw=dict(reprog_ms=100.0), conditions=[dict(a=0), dict(a=1)], schedule=[0, 1]))
    _feed(e, ["A"], 2); time.sleep(0.5); e.close()
    assert [r["stim"] for r in e.stim_rows] == ["hold", "hold"]


def test_image_decoder_placeholder_and_linear_export(tmp_path):
    an = ImageDecoderAnalysis(images={0: "a.png", 1: "b.png"}, target_class=1, n_fit=6)
    rng = np.random.default_rng(0); mus = {0: np.r_[5, 0, 5, 0.], 1: np.r_[0, 5, 0, 5.]}
    for i in range(6):
        c = i % 2; assert an(np.tile(mus[c], (10, 1)) + rng.normal(0, .3, (10, 4)), c)[0] is None
    loss, d = an(np.tile(mus[1], (10, 1)), 1)
    assert d["decoded"] == 1 and d["decoded_image"] == "b.png" and d["target_image"] == "b.png" and loss < 0.7
    p = export_linear_readout(an, tmp_path / "r.npz")
    lin = ImageDecoderAnalysis(images={0: "a.png", 1: "b.png"}, target_class=1, model=str(p))
    assert lin(np.tile(mus[0], (10, 1)), 0)[1]["decoded"] == 0 and lin(np.tile(mus[1], (10, 1)), 1)[1]["decoded"] == 1


def test_zmq_inproc_roundtrip_and_condition_backchannel():
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB); pub.bind("inproc://t-pub")
    csub = ctx.socket(zmq.SUB); csub.setsockopt(zmq.SUBSCRIBE, b"")
    e = Engine(dict(analysis_kw=dict(n_fit=2), stimulator_kw=dict(reprog_ms=0.0), conditions=[dict(a=0), dict(a=1)], schedule=[0, 1]))
    zw = ZmqWorker(e, pub="inproc://t-pub", cond="inproc://t-cond", context=ctx).start()
    csub.connect("inproc://t-cond"); time.sleep(0.2)
    pub.send_multipart(protocol.hello("A", 1, 30000.0, 200, [0, 1, 2])); time.sleep(0.2)
    e.arm(); time.sleep(0.1)
    got = []
    while csub.poll(100):
        got.append(protocol.decode(csub.recv_multipart())[1])
    zw.stop(); e.close(); pub.close(0); csub.close(0); ctx.term()
    assert zw.n_msgs == 1 and "A" in e.S and {"type": "condition", "id": 0} in got
