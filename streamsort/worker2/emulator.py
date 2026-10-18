"""`streamsort emulate`: replay a recorded session through the C++ core and publish exactly what the
Open Ephys plugin publishes (protocol.py), including the condition back-channel. For testing the
worker / worker GUI without the GUI; also usable on the rig to rehearse a session.

Mirrors StreamsortProcessor + StreamDecoder: one decoder per stream, stim TTLs paired with artifact
onsets by block arrival time (C++ PulsePairer), spikes published once per block, hello on start.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..core_cpp import CoreDecoder, load_exported, _module
from ..io.openephys import OpenEphysSession
from ..io.stream import FileStreamSource
from . import protocol


class Emulator:
    def __init__(self, session_dir, banks: dict, t0: float, seconds: float, buffer: int = 1024, pace: float = 1.0,
                 ttl_stream: str | None = "NI-DAQmx-107.PXI-6133", ttl_line: int = 5, threads: int = 1,
                 pub: str = "tcp://*:5557", cond: str = "tcp://127.0.0.1:5558", context=None, startup_wait_s: float = 0.5, record: bool = False):
        import zmq
        self.zmq = zmq
        self.ctx = context or zmq.Context.instance()
        sess = OpenEphysSession(session_dir)
        self.streams = {}
        for i, (name, folder) in enumerate(banks.items()):
            st = sess.stream(name); tab = load_exported(Path(folder) / "streamsort" if (Path(folder) / "streamsort").exists() else folder)
            C = int(tab["C"]); s0 = int(st.time_to_sample(np.array([t0]))[0])
            used = np.zeros(len(tab["unit_ids"]), bool); used[np.asarray(tab["cand_units"], int)] = True
            self.streams[name] = dict(id=i + 1, st=st, src=FileStreamSource(st, C), s0=s0, n=int(seconds * st.effective_rate),
                                      dec=CoreDecoder(tab, threads), C=C, units=np.asarray(tab["unit_ids"])[used],
                                      pairer=_module().PulsePairer(ttl_stream is not None, 0.05), n_ev=0, arrivals=[], msg=0,
                                      bank=str(folder), fp=str(tab.get("fingerprint", "")))
        first = next(iter(self.streams.values()))
        self.ttl = np.zeros(0, np.int64)
        if ttl_stream:
            up = sess.ttl_edges(ttl_stream, ttl_line, True)
            self.ttl = np.sort(first["st"].time_to_sample(np.asarray(up, float)) - first["s0"])
            self.ttl = self.ttl[(self.ttl >= 0) & (self.ttl < first["n"])]
        self.ttl_stream, self.ttl_line = ttl_stream, ttl_line
        self.buffer, self.pace, self.wait = int(buffer), float(pace), float(startup_wait_s)
        self.pub = self.ctx.socket(zmq.PUB); self.pub.setsockopt(zmq.SNDHWM, 100000); self.pub.bind(pub)
        self.sub = self.ctx.socket(zmq.SUB); self.sub.setsockopt(zmq.SUBSCRIBE, b""); self.sub.connect(cond)
        self.block_s = []; self.stop_ev = threading.Event(); self.n_ttl = 0
        self.record = record; self.recorded = {k: [] for k in self.streams}

    def _send(self, frames):
        self.pub.send_multipart(frames)

    def _poll_conditions(self):
        while True:
            try:
                fr = self.sub.recv_multipart(self.zmq.NOBLOCK)
            except self.zmq.Again:
                return
            env, hdr, _ = protocol.decode(fr)
            if hdr.get("type") == "condition":
                for s in self.streams.values():
                    s["dec"]._dec.set_condition(int(hdr["id"]))

    def run(self):
        time.sleep(self.wait)                         # let subscribers connect (ZMQ slow joiner)
        for name, s in self.streams.items():
            s["dec"].start_at(0)
            self._send(protocol.hello(name, s["id"], s["st"].effective_rate, s["C"], s["units"], s["bank"], s["fp"]))
        time.sleep(0.2); self._poll_conditions()
        n = min(s["n"] for s in self.streams.values()); B = self.buffer
        fs = next(iter(self.streams.values()))["st"].effective_rate
        t_start = time.perf_counter(); ti = 0
        with ThreadPoolExecutor(len(self.streams)) as ex:
            for b0 in range(0, n, B):
                if self.stop_ev.is_set():
                    break
                m = min(B, n - b0); now = time.monotonic(); wall = protocol.wall_us()
                self._poll_conditions()
                while ti < len(self.ttl) and self.ttl[ti] < b0 + m:          # stim TTLs of this block
                    self._send(protocol.event(self.ttl_stream, self.ttl_line, int(self.ttl[ti]), ti, t_wall=wall))
                    for s in self.streams.values():
                        s["pairer"].add_ttl(ti, now)
                    ti += 1; self.n_ttl = ti
                blocks = {k: np.asarray(s["src"].read(s["s0"] + b0, m)).astype(np.float32) for k, s in self.streams.items()}
                t0 = time.perf_counter()
                outs = dict(zip(self.streams, ex.map(lambda k: self.streams[k]["dec"].push(blocks[k]), self.streams)))
                self.block_s.append(time.perf_counter() - t0)
                for k, s in self.streams.items():
                    head = b0 + m; s["arrivals"].append((head, now, wall)); s["arrivals"] = s["arrivals"][-256:]
                    ev = s["dec"]._dec.events()
                    for e in ev[s["n_ev"]:]:
                        on = int(e[0]); t = next((a[1] for a in s["arrivals"] if on < a[0]), now)
                        s["pairer"].add_artifact(on, t)
                    s["n_ev"] = len(ev)
                    for on, t, kk in s["pairer"].pair(now):
                        w = next((a[2] for a in s["arrivals"] if 0 <= on < a[0]), wall)
                        self._send(protocol.pulse(k, on, kk, t_wall=w))
                    b = outs[k]; sel = b.unit >= 0
                    if self.record:
                        self.recorded[k].append((b.frame[sel], b.unit[sel]))
                    self._send(protocol.spikes(k, s["id"], s["msg"], head, s["st"].effective_rate, b.frame[sel], b.unit[sel], t_wall=wall)); s["msg"] += 1
                if (b0 // B) % max(1, int(2.0 * fs / B)) == 0 and b0:      # periodic hello, as the plugin
                    for k, s in self.streams.items():
                        self._send(protocol.hello(k, s["id"], s["st"].effective_rate, s["C"], s["units"], s["bank"], s["fp"]))
                if self.pace > 0:
                    lag = t_start + (b0 + m) / fs / self.pace - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
        for k, s in self.streams.items():
            b = s["dec"].flush(); sel = b.unit >= 0
            if self.record:
                self.recorded[k].append((b.frame[sel], b.unit[sel]))
            self._send(protocol.spikes(k, s["id"], s["msg"], n, s["st"].effective_rate, b.frame[sel], b.unit[sel]))
        time.sleep(0.3)

    def close(self):
        self.pub.close(0); self.sub.close(0)
