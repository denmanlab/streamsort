"""Replay a recorded Open Ephys session through the plugin code path, GUI-style.

Blocks of ``buffer`` samples are converted to float32 microvolts (n_channels, n) exactly as the GUI
hands them to a Python Processor, stim TTLs are injected on each probe stream (the required rig
wiring), and every probe's DecoderProcessor runs serially in one thread (the GUI shares one
interpreter between processors).  Used by ``streamsort replay`` and scripts/replay_harness.py.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from multiprocessing.connection import Client

import numpy as np

from ..io.openephys import OpenEphysSession
from ..io.stream import FileStreamSource
from .processor import DecoderProcessor
from . import protocol, worker as worker_mod


def load_probe_cfg(path) -> dict:
    return json.loads(Path(path).read_text())


class GuiFeeder:
    """One probe stream as the GUI would deliver it: float32 uV blocks plus stim-TTL events."""
    def __init__(self, sess: OpenEphysSession, cfg: dict, t0: float, dur: float, buffer: int,
                 ttl_times=None):
        st = sess.stream(cfg["stream"])
        self.src = FileStreamSource(st)               # all channels on disk, as in the GUI
        self.bit_volts = float(cfg.get("bit_volts", st.bit_volts))
        self.s0 = int(st.time_to_sample(np.array([t0]))[0]); self.n = int(dur * st.effective_rate)
        self.buffer, self.pos = int(buffer), 0
        tt = np.asarray(ttl_times if ttl_times is not None else [], float)
        self.ttl = np.sort(st.time_to_sample(tt) - self.s0) if len(tt) else np.zeros(0, np.int64)
        self.ttl = self.ttl[(self.ttl >= 0) & (self.ttl < self.n)]
        self.line = cfg.get("ttl_line")

    def blocks(self):
        """yield (ttl_events_in_block, block_uV, counts)"""
        j = 0
        while self.pos < self.n:
            m = min(self.buffer, self.n - self.pos)
            counts = np.asarray(self.src.read(self.s0 + self.pos, m))
            k = j
            while k < len(self.ttl) and self.ttl[k] < self.pos + m:
                k += 1
            ev, j = list(range(j, k)), k
            self.pos += m
            yield ev, (counts.T.astype(np.float32) * np.float32(self.bit_volts)), counts


def run(session_dir, probe_cfgs: dict, t0: float, dur: float, buffer: int = 1024,
        ttl_times=None, worker_cfg: dict | None = None, pace: float = 1.0, swap_at_s: float | None = None,
        swap_fn=None, schedule_fn=None, transport: str = "listener"):
    """Replay; returns dict(spikes={probe: (frames, units)}, buffer_s={probe: [...]}, worker=log, cores).
    worker_cfg None: no worker (deterministic); schedule_fn(core, block_start, block_end) may set conditions.
    pace: 1.0 = real time, 0 = as fast as possible."""
    t_call = time.perf_counter()
    sess = OpenEphysSession(session_dir)
    feeders = {p: GuiFeeder(sess, c, t0, dur, buffer, ttl_times) for p, c in probe_cfgs.items()}
    proc, conns = None, {}
    if worker_cfg is not None:
        # transport="listener": the rig path (TCP / AF_UNIX / named pipe); "pipe": multiprocessing.Pipe,
        # same Connection API, for sandboxes that forbid binding sockets.
        ctx = mp.get_context("spawn"); ready = ctx.Event(); wconns = None
        if transport == "pipe":
            pairs = {p: ctx.Pipe() for p in probe_cfgs}; conns = {p: a for p, (a, _) in pairs.items()}
            wconns = [b for _, b in pairs.values()]
        proc = ctx.Process(target=_worker_entry, args=(worker_cfg, ready, wconns), daemon=True); proc.start()
        if not ready.wait(60):
            raise RuntimeError("worker did not start")
        if wconns is not None:
            for c in wconns:
                c.close()                     # child holds its copies
    cores = {}
    for p, c in probe_cfgs.items():
        f = feeders[p]
        cores[p] = DecoderProcessor(c, f.src.n_channels, f.src.stream.effective_rate)
        if worker_cfg is not None:
            if transport != "pipe":
                conns[p] = Client(protocol.address(worker_cfg), family=protocol.family(worker_cfg),
                                  authkey=protocol.authkey(worker_cfg))
            cores[p].conn = conns[p]
        cores[p].start(0)
    out = {p: [] for p in cores}; buf_s = {p: [] for p in cores}; swapped = False
    gens = {p: feeders[p].blocks() for p in cores}
    fs = next(iter(feeders.values())).src.stream.effective_rate
    t_wall0, n_done = time.perf_counter(), 0
    startup_s = t_wall0 - t_call
    while gens:
        for p in list(gens):
            try:
                ev, x, _ = next(gens[p])
            except StopIteration:
                del gens[p]; continue
            core = cores[p]
            if schedule_fn is not None:
                schedule_fn(core, core.head, core.head + x.shape[1])
            for k in ev:
                core.ttl(feeders[p].line if feeders[p].line is not None else -1, True)
            t = time.perf_counter(); b = core.process(x); buf_s[p].append(time.perf_counter() - t)
            out[p].append(b)
        n_done += buffer
        if swap_fn is not None and not swapped and swap_at_s is not None and n_done / fs >= swap_at_s:
            swap_fn(); swapped = True
        if pace > 0:
            lag = t_wall0 + n_done / fs / pace - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    stream_s = time.perf_counter() - t_wall0
    for p, core in cores.items():
        out[p].append(core.stop())
    log = None
    if proc is not None:
        for c in conns.values():
            c.close()
        proc.join(60)
        if worker_cfg.get("log") and Path(worker_cfg["log"]).exists():
            log = json.loads(Path(worker_cfg["log"]).read_text())
    from ..online.decoder import SpikeBatch
    spikes = {}
    for p, bs in out.items():
        b = SpikeBatch.concat(bs); m = b.unit >= 0
        spikes[p] = (b.frame[m], b.unit[m])
    return dict(spikes=spikes, buffer_s=buf_s, worker=log, cores=cores, feeders=feeders, startup_s=startup_s, stream_s=stream_s)


def _worker_entry(cfg, ready, conns=None):
    worker_mod.main(cfg, ready=ready, conns=conns)
