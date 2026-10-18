"""Worker v2 engine: consumes plugin messages, pools streams per trial, runs the analysis,
drives the stimulator on its own thread, sends the next condition back to the plugin.

Config (dict / JSON):
  streams        list of stream names to pool (default: every stream that says hello)
  iti_s 2.0, w0_ms 2, win_ms 500, bin_ms 10, smooth_ms 20, pair_tol_ms 5, deadline_margin_ms 50
  analysis       "procrustes" | "image_decoder", analysis_kw {...}
  stimulator     "dummy" | "serial", stimulator_kw {...}
  conditions     list of condition dicts (id = index); schedule: list of condition ids (cycled)
  log            path of the per-trial JSON log
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry.core import bin_trial
from .analyses import ANALYSES
from .stimulator import STIMULATORS
from . import protocol


@dataclass
class _Stream:
    fs: float = 30000.0
    units: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    head: int = 0
    frames: list = field(default_factory=list)
    unitl: list = field(default_factory=list)


class Engine:
    def __init__(self, cfg: dict, send_condition=None, on_trial=None, on_status=None, stimulator=None):
        self.cfg = dict(cfg)
        g = lambda k, d: float(self.cfg.get(k, d))
        self.iti, self.bin_ms, self.sm_ms = g("iti_s", 2.0), g("bin_ms", 10), g("smooth_ms", 20)
        self.w0, self.win = g("w0_ms", 2) * 1e-3, g("win_ms", 500) * 1e-3
        self.tol, self.margin = g("pair_tol_ms", 5) * 1e-3, g("deadline_margin_ms", 50) * 1e-3
        self.want = list(self.cfg.get("streams", []))
        self.S: dict[str, _Stream] = {}
        self.analysis = ANALYSES[self.cfg.get("analysis", "procrustes")](**self.cfg.get("analysis_kw", {}))
        self.stim = stimulator or STIMULATORS[self.cfg.get("stimulator", "dummy")](**self.cfg.get("stimulator_kw", {}))
        self.conditions = list(self.cfg.get("conditions", []))
        self.schedule = [int(c) for c in self.cfg.get("schedule", range(len(self.conditions)))]
        self.sched_i = 0
        self.send_condition, self.on_trial, self.on_status = send_condition, on_trial, on_status
        self.trials, self.rows, self.stim_rows, self.ttl_wall = [], [], [], {}
        self.cond_now = None
        self.stim_q, self.lock = queue.Queue(), threading.Lock()
        self._stim_th = threading.Thread(target=self._stim_loop, daemon=True); self._stim_th.start()

    # -- schedule -------------------------------------------------------------------------
    def _next_condition(self):
        if not self.schedule:
            return None
        c = self.schedule[self.sched_i % len(self.schedule)]; self.sched_i += 1
        return c

    def arm(self):
        """Condition of the first pulse; call once before (or at) acquisition start."""
        if self.cond_now is None:
            self.cond_now = self._next_condition()
            self._program(self.cond_now)
            if self.send_condition and self.cond_now is not None:
                self.send_condition(self.cond_now)

    def streams(self):
        return self.want or sorted(self.S)

    # -- messages -------------------------------------------------------------------------
    def on_message(self, env: str, hdr: dict, data: dict, t_recv: float | None = None):
        t = hdr.get("type")
        if t == "hello":
            s = self.S.setdefault(hdr["stream"], _Stream())
            units = np.asarray(hdr["unit_ids"], np.int64); changed = not np.array_equal(units, s.units)
            s.fs = float(hdr["sample_rate"]); s.units = units
            if not changed:                         # periodic hello (late-joining workers): nothing to do
                return
            self.analysis.reset()                   # unit set (re)defined (start or hot-swap): refit
            if self.on_status:
                self.on_status(dict(kind="hello", stream=hdr["stream"], n_units=len(s.units)))
            if self.send_condition and self.cond_now is not None:
                self.send_condition(self.cond_now)
        elif t == "spikes":
            s = self.S.get(hdr["stream"])
            if s is None:
                return
            s.head = int(hdr["head"])
            if len(data.get("frames", ())):
                s.frames.append(np.asarray(data["frames"])); s.unitl.append(np.asarray(data["units"]))
            self._try_trials()
        elif t == "event":
            k = int(hdr["ttl_index"]); tw = hdr["t_wall_us"] * 1e-6
            self.ttl_wall[k] = tw
            if not any(tr["k"] == k for tr in self.trials):      # TTL arrives before the artifact: reprogram now
                tr = dict(k=k, t=None, on={}, t_pulse_wall=tw, cond=self.cond_now, done=False)
                self.trials.append(tr); self.stim_q.put(tr)
        elif t == "pulse":
            self._add_pulse(hdr["stream"], int(hdr["onset_sample"]), int(hdr["ttl_index"]), hdr["t_wall_us"] * 1e-6)

    def _add_pulse(self, stream, onset, k, t_wall):
        s = self.S.get(stream)
        if s is None:
            return
        ts = onset / s.fs if onset >= 0 else None
        for tr in self.trials:
            if (k >= 0 and tr["k"] == k) or (k < 0 and ts is not None and tr["t"] is not None and abs(tr["t"] - ts) <= self.tol):
                if onset >= 0:
                    tr["on"].setdefault(stream, onset)
                    if tr["t"] is None:
                        tr["t"] = ts
                tr["t_pulse_wall"] = min(tr["t_pulse_wall"], t_wall)
                return
        tr = dict(k=k, t=ts, on=({stream: onset} if onset >= 0 else {}), t_pulse_wall=t_wall, cond=self.cond_now, done=False)
        self.trials.append(tr)
        self.stim_q.put(tr)                             # reprogram for the next pulse now, in parallel with the response

    def _onset(self, tr, name):
        s = self.S[name]
        if name in tr["on"]:
            return tr["on"][name]
        for o, v in tr["on"].items():                   # stream without a detected artifact: use another stream's onset time
            return int(round(v / self.S[o].fs * s.fs))
        return None

    def _try_trials(self):
        names = self.streams()
        for tr in self.trials:
            if tr["done"]:
                continue
            ok = True
            for n in names:
                if n not in self.S:
                    ok = False; break
                on = self._onset(tr, n)
                if on is None or self.S[n].head < on + int((self.w0 + self.win + 2e-3) * self.S[n].fs):
                    ok = False; break
            if ok:
                self._analyse(tr, names)
        kmax = max((t["k"] for t in self.trials), default=-1)     # TTL never matched by an artifact on any stream: drop
        self.trials = [t for t in self.trials if not t["done"] and not (not t["on"] and t["k"] >= 0 and t["k"] < kmax - 5)] \
            + [t for t in self.trials if t["done"]][-3:]

    def _analyse(self, tr, names):
        t0 = time.time(); Xs = []
        for n in names:
            s = self.S[n]
            f = np.concatenate(s.frames) if s.frames else np.zeros(0, np.int64)
            u = np.concatenate(s.unitl) if s.unitl else np.zeros(0, np.int64)
            on = self._onset(tr, n); a = on + int(self.w0 * s.fs)
            Xs.append(bin_trial(f, u, s.units, a, int(self.win * s.fs), s.fs, bin_ms=self.bin_ms, smooth_ms=self.sm_ms) / (self.bin_ms * 1e-3))
            keep = f >= on - int(1.0 * s.fs)
            s.frames, s.unitl = ([f[keep]], [u[keep]]) if len(f) else ([], [])
        nb = min(len(x) for x in Xs); X = np.concatenate([x[:nb] for x in Xs], axis=1)
        loss, disp = self.analysis(X, tr["cond"])
        t1 = time.time(); tr["done"] = True
        t_ttl = self.ttl_wall.get(tr["k"])
        ref = t_ttl if t_ttl is not None else tr["t_pulse_wall"]
        row = dict(k=tr["k"], cond=tr["cond"], n_units=int(X.shape[1]), n_streams_pulse=len(tr["on"]), loss=loss,
                   geometry_ms=(t1 - t0) * 1e3, ttl_to_loss_ms=(t1 - ref) * 1e3, ref="ttl" if t_ttl is not None else "pulse_block")
        with self.lock:
            self.rows.append(row)
        if row["ttl_to_loss_ms"] > self.iti * 1e3:
            print(f"[worker] WARNING trial {tr['k']}: TTL->loss {row['ttl_to_loss_ms']:.0f} ms exceeds the ITI", flush=True)
        if self.on_trial:
            self.on_trial(row, disp)

    # -- stimulator thread ------------------------------------------------------------------
    def _program(self, cid):
        if cid is None or not self.conditions:
            return
        self.stim.program(dict(self.conditions[cid], id=cid))

    def _stim_loop(self):
        while True:
            tr = self.stim_q.get()
            if tr is None:
                return
            nxt = self._next_condition()
            t_ttl = self.ttl_wall.get(tr["k"], tr["t_pulse_wall"])
            deadline = t_ttl + self.iti - self.margin
            t0 = time.time()
            if nxt is None or t0 + getattr(self.stim, "reprog_s", 0.0) > deadline:
                state = "hold"
            else:
                self._program(nxt); self.cond_now = nxt; state = "programmed"
                if self.send_condition:
                    self.send_condition(nxt)
            r = dict(k=tr["k"], stim=state, next=nxt, ttl_to_stim_ms=(time.time() - t_ttl) * 1e3)
            with self.lock:
                self.stim_rows.append(r)
            if self.on_status:
                self.on_status(dict(kind="stim", **r))

    def close(self):
        self.stim_q.put(None); self._stim_th.join(5)
        try:
            self.stim.close()
        except Exception:
            pass
        if self.cfg.get("log"):
            with open(self.cfg["log"], "w") as f:
                json.dump(dict(trials=self.rows, stim=self.stim_rows), f, default=float)


class ZmqWorker:
    """Engine + ZMQ: SUB to the plugin's publisher, PUB for conditions. Runs in a background thread."""
    def __init__(self, engine: Engine, pub=protocol.DEFAULT_PUB, cond=protocol.DEFAULT_COND, context=None):
        import zmq
        self.zmq, self.engine = zmq, engine
        self.ctx = context or zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB); self.sub.setsockopt(zmq.SUBSCRIBE, b""); self.sub.connect(pub)
        self.pubc = self.ctx.socket(zmq.PUB); self.pubc.bind(cond)
        self._clock = threading.Lock()
        engine.send_condition = self.send_condition
        self.stop_ev = threading.Event(); self.n_msgs = 0
        self.th = threading.Thread(target=self._loop, daemon=True)

    def send_condition(self, cid):
        with self._clock:
            self.pubc.send_multipart(protocol.condition(cid))

    def start(self):
        self.th.start(); return self

    def _loop(self):
        poller = self.zmq.Poller(); poller.register(self.sub, self.zmq.POLLIN)
        while not self.stop_ev.is_set():
            if not dict(poller.poll(20)):
                continue
            while True:
                try:
                    fr = self.sub.recv_multipart(self.zmq.NOBLOCK)
                except self.zmq.Again:
                    break
                env, hdr, data = protocol.decode(fr); self.n_msgs += 1
                self.engine.on_message(env, hdr, data)

    def stop(self):
        self.stop_ev.set(); self.th.join(2)
        self.sub.close(0); self.pubc.close(0)
