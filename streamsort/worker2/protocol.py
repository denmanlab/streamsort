"""Messages published by the streamsort Open Ephys plugin (and by `streamsort emulate`).

Framing follows the Open Ephys ZMQ Interface: multipart [envelope][JSON header][binary payload?].

  envelope  header["type"]  fields                                                  payload
  STATUS    hello           stream, stream_id, sample_rate, n_channels, bank,        -
                            fingerprint, unit_ids[], t_wall_us
  SPIKES    spikes          stream, stream_id, message_num, n, head, sample_rate,    n x int64 sample numbers,
                            t_wall_us   (n = 0: heartbeat, head advanced)            then n x int32 unit ids
  PULSE     pulse           stream, onset_sample (-1 = TTL without artifact),        -
                            ttl_index (-1 = no TTL source), t_wall_us, approx
  EVENT     event           stream, line (1-based), state, sample_num, timestamp_s,  -
                            ttl_index, t_wall_us          (stim TTL rising edges)

Worker -> plugin (worker binds PUB on the condition port, plugin SUB connects):
  envelope CONDITION, header {"type": "condition", "id": <int>}   condition id of the NEXT pulse
t_wall_us = microseconds since the Unix epoch on the acquisition machine (Python: time.time()).
"""
from __future__ import annotations

import json
import time

import numpy as np

DEFAULT_PUB = "tcp://127.0.0.1:5557"      # worker connects here (plugin binds tcp://*:5557)
DEFAULT_COND = "tcp://*:5558"             # worker binds here (plugin connects tcp://127.0.0.1:5558)


def wall_us() -> int:
    return int(time.time() * 1e6)


def _hdr(d: dict) -> bytes:
    return json.dumps(d, separators=(",", ":")).encode()


def hello(stream, stream_id, sample_rate, n_channels, unit_ids, bank="", fingerprint=""):
    return [b"STATUS", _hdr(dict(type="hello", stream=stream, stream_id=int(stream_id), sample_rate=float(sample_rate),
                                  n_channels=int(n_channels), bank=str(bank), fingerprint=str(fingerprint),
                                  t_wall_us=wall_us(), unit_ids=[int(u) for u in unit_ids]))]


def spikes(stream, stream_id, message_num, head, sample_rate, frames, units, t_wall=None):
    f = np.ascontiguousarray(frames, "<i8"); u = np.ascontiguousarray(units, "<i4")
    h = _hdr(dict(type="spikes", stream=stream, stream_id=int(stream_id), message_num=int(message_num), n=int(len(f)),
                  head=int(head), sample_rate=float(sample_rate), t_wall_us=int(t_wall if t_wall is not None else wall_us())))
    return [b"SPIKES", h, f.tobytes() + u.tobytes()]


def pulse(stream, onset_sample, ttl_index, t_wall=None):
    return [b"PULSE", _hdr(dict(type="pulse", stream=stream, onset_sample=int(onset_sample), ttl_index=int(ttl_index),
                                 t_wall_us=int(t_wall if t_wall is not None else wall_us()), approx=bool(onset_sample < 0)))]


def event(stream, line, sample_num, ttl_index, timestamp_s=-1.0, t_wall=None):
    return [b"EVENT", _hdr(dict(type="event", stream=stream, line=int(line), state=1, sample_num=int(sample_num),
                                 timestamp_s=float(timestamp_s), ttl_index=int(ttl_index),
                                 t_wall_us=int(t_wall if t_wall is not None else wall_us())))]


def condition(cid: int):
    return [b"CONDITION", _hdr(dict(type="condition", id=int(cid)))]


def decode(frames):
    """multipart -> (envelope, header dict, payload dict); SPIKES payload -> frames/units arrays."""
    env = bytes(frames[0]).decode(); hdr = json.loads(bytes(frames[1]))
    data = {}
    if hdr.get("type") == "spikes":
        n = int(hdr["n"]); buf = bytes(frames[2]) if len(frames) > 2 else b""
        data["frames"] = np.frombuffer(buf, "<i8", count=n, offset=0) if n else np.zeros(0, np.int64)
        data["units"] = np.frombuffer(buf, "<i4", count=n, offset=8 * n).astype(np.int64) if n else np.zeros(0, np.int64)
    return env, hdr, data
