"""Out-of-GUI worker: pools spikes from every probe's decoder, computes per-trial population
geometry and a loss, and drives the stimulator on its own thread.

Runs as a separate process (``streamsort worker --config worker.json``), so analysis and stimulator
I/O never hold the GUI's shared interpreter.  Analyses and reducers are swappable:
  analysis = "procrustes": reducer (pca|fa|isomap) fitted causally on the first ``n_fit`` trials'
             condition means; loss = Procrustes distance of the trial trajectory to the target
             (mean trajectory of ``target_condition`` over the fit trials, or a loaded ``target``).
  analysis = "decoder":    pre-loaded linear readout (npz: W (n_units, n_classes), b, classes);
             loss = -log p(target_class) of the trial's mean rate vector.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Listener

import numpy as np

from ..geometry.core import bin_trial, make_reducer, procrustes_distance
from . import protocol


# ---------------------------------------------------------------- analyses
class ProcrustesAnalysis:
    def __init__(self, reducer="pca", k=3, n_fit=50, target_condition=None, target=None, **kw):
        self.red_name, self.k, self.n_fit, self.tc = reducer, int(k), int(n_fit), target_condition
        self.red, self.target = None, (np.asarray(target) if target is not None else None)
        self.fit_X, self.fit_c = [], []

    def reset(self):                       # unit set changed (template hot-swap): refit
        self.red, self.fit_X, self.fit_c = None, [], []

    def __call__(self, X, cond):
        if self.red is None:
            self.fit_X.append(X); self.fit_c.append(cond)
            if len(self.fit_X) < self.n_fit:
                return None, None
            c = np.array([str(x) for x in self.fit_c]); Xs = np.stack(self.fit_X)
            self.red = make_reducer(self.red_name, self.k).fit(np.concatenate([Xs[c == u].mean(0) for u in np.unique(c)]))
            if self.target is None:
                sel = (c == str(self.tc)) if self.tc is not None and (c == str(self.tc)).any() else np.ones(len(c), bool)
                self.target = self.red.transform(Xs[sel].mean(0))
            return None, None
        z = self.red.transform(X)
        return float(procrustes_distance(self.target, z)), z


class DecoderAnalysis:
    def __init__(self, model, target_class=None, **kw):
        m = np.load(model, allow_pickle=False)
        self.W, self.b, self.classes = m["W"], m["b"], [str(c) for c in m["classes"]]
        self.t = self.classes.index(str(target_class)) if target_class is not None else 0

    def reset(self):
        pass

    def __call__(self, X, cond):
        r = X.mean(0)
        if len(r) != self.W.shape[0]:
            return None, None              # unit set does not match the loaded readout
        a = r @ self.W + self.b; a = a - a.max(); p = np.exp(a) / np.exp(a).sum()
        return float(-np.log(max(p[self.t], 1e-12))), p


ANALYSES = {"procrustes": ProcrustesAnalysis, "decoder": DecoderAnalysis}


# ---------------------------------------------------------------- stimulator
class DummyStimulator:
    """Stand-in for the rig's stimulator: reprogramming takes ``reprog_ms``."""
    def __init__(self, reprog_ms=100.0):
        self.reprog_s, self.log = float(reprog_ms) * 1e-3, []

    def program(self, condition):
        time.sleep(self.reprog_s); self.log.append(condition)


class SchedulePolicy:
    """Next condition from a fixed list (cycled).  Closed-loop policies replace ``__call__``."""
    def __init__(self, schedule):
        self.s, self.i = list(schedule), 0

    def __call__(self, history):
        if not self.s:
            return None
        c = self.s[self.i % len(self.s)]; self.i += 1
        return c


# ---------------------------------------------------------------- worker
@dataclass
class _Probe:
    fs: float = 30000.0
    units: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    head: int = 0
    frames: list = field(default_factory=list)
    unitl: list = field(default_factory=list)
    conn: object = None


class Worker:
    """cfg keys: probes (list), address, authkey, iti_s (2.0), bin_ms (10), smooth_ms (20),
    w0_ms (2), win_ms (500), pair_tol_ms (5), refractory_ms (300), analysis (+ its kwargs under
    "analysis_kw"), reprog_ms (100), deadline_margin_ms (50), schedule (list of conditions), log (path)."""

    def __init__(self, cfg: dict, stimulator=None, policy=None):
        self.cfg = cfg
        g = lambda k, d: float(cfg.get(k, d))
        self.iti, self.bin_ms, self.sm_ms = g("iti_s", 2.0), g("bin_ms", 10), g("smooth_ms", 20)
        self.w0, self.win, self.tol = g("w0_ms", 2) * 1e-3, g("win_ms", 500) * 1e-3, g("pair_tol_ms", 5) * 1e-3
        self.refr, self.margin = g("refractory_ms", 300) * 1e-3, g("deadline_margin_ms", 50) * 1e-3
        self.names = [str(p) for p in cfg["probes"]]
        self.P = {p: _Probe() for p in self.names}
        self.analysis = ANALYSES[cfg.get("analysis", "procrustes")](**cfg.get("analysis_kw", {}))
        self.stim = stimulator or DummyStimulator(cfg.get("reprog_ms", 100.0))
        sched = cfg.get("schedule", [])
        self.policy = policy or SchedulePolicy([tuple(c) if isinstance(c, list) else c for c in sched])
        self.trials, self.rows, self.stim_rows, self.cond_now = [], [], [], None
        self.inbox, self.stim_q = queue.Queue(), queue.Queue()
        self.lock, self.done = threading.Lock(), threading.Event()

    # -- connection threads --------------------------------------------------------
    def _reader(self, conn):
        try:
            while True:
                self.inbox.put((time.perf_counter(), conn, conn.recv()))
        except (EOFError, OSError):
            self.inbox.put((time.perf_counter(), conn, ("closed",)))

    def serve(self, n_clients=None, ready=None, conns=None):
        """Accept one connection per probe on a Listener (rig), or use given connections (pipes)."""
        lst = None
        if conns is None:
            lst = Listener(protocol.address(self.cfg) or protocol.DEFAULT_ADDRESS,
                           family=protocol.family(self.cfg), authkey=protocol.authkey(self.cfg))
        if ready is not None:
            ready.set()
        threading.Thread(target=self._stim_loop, daemon=True).start()
        n = len(conns) if conns is not None else (n_clients or len(self.names))
        for i in range(n):
            c = conns[i] if conns is not None else lst.accept()
            threading.Thread(target=self._reader, args=(c,), daemon=True).start()
        open_ = n
        while open_:
            t, conn, msg = self.inbox.get()
            if msg[0] in ("closed", "bye"):
                open_ -= msg[0] == "closed"
                continue
            self.handle(msg, conn, t)
        self.stim_q.put(None); self.done.wait(5.0)
        if lst is not None:
            lst.close()
        self._write_log()
        return dict(trials=self.rows, stim=self.stim_rows)

    # -- message handling --------------------------------------------------------------
    def handle(self, msg, conn=None, t_recv=None):
        kind, p = msg[0], str(msg[1])
        P = self.P[p]
        if kind == "hello":
            _, _, P.fs, units, _ = msg; P.units, P.conn = np.asarray(units, np.int64), conn
            self.analysis.reset()
            if self.cond_now is None:
                self.cond_now = self.policy([])
            if conn is not None:
                conn.send(("condition", self.cond_now))
        elif kind == "buf":
            _, _, P.head, f, u = msg
            if len(f):
                P.frames.append(np.asarray(f)); P.unitl.append(np.asarray(u))
            self._try_trials()
        elif kind == "pulse":
            on, k = msg[2], msg[3]
            t_arr = msg[4] if len(msg) > 4 else (t_recv or time.perf_counter())   # block-arrival stamp
            self._add_pulse(p, int(on), k, t_arr)

    def _add_pulse(self, p, on, k, t):
        ts = on / self.P[p].fs
        for tr in self.trials:                      # same stimulus seen on another probe
            if (k is not None and tr["k"] == k) or (k is None and abs(tr["t"] - ts) <= self.tol):
                tr["on"].setdefault(p, on); tr["t_pulse"] = min(tr["t_pulse"], t); return
        if any(abs(tr["t"] - ts) < self.refr and p in tr["on"] for tr in self.trials[-3:]):
            return                                  # second clip within one stimulus
        self.trials.append(dict(k=k, t=ts, on={p: on}, t_pulse=t, cond=self.cond_now, done=False))
        self.stim_q.put(("next", self.trials[-1]))  # reprogram now, in parallel with the response

    def _try_trials(self):
        for tr in self.trials:
            if tr["done"]:
                continue
            ready = all(self.P[p].head / self.P[p].fs >= self._on_s(tr, p) + self.w0 + self.win + 2e-3
                        for p in self.names)
            if ready:
                self._analyse(tr)
        self.trials = [tr for tr in self.trials if not tr["done"]] + [tr for tr in self.trials if tr["done"]][-3:]

    def _on_s(self, tr, p):
        return tr["on"][p] / self.P[p].fs if p in tr["on"] else tr["t"]

    def _analyse(self, tr):
        t0 = time.perf_counter(); Xs = []
        for p in self.names:
            P = self.P[p]
            f = np.concatenate(P.frames) if P.frames else np.zeros(0, np.int64)
            u = np.concatenate(P.unitl) if P.unitl else np.zeros(0, np.int64)
            on = int(round(self._on_s(tr, p) * P.fs)); a = on + int(self.w0 * P.fs); n = int(self.win * P.fs)
            Xs.append(bin_trial(f, u, P.units, a, n, P.fs, bin_ms=self.bin_ms, smooth_ms=self.sm_ms) / (self.bin_ms * 1e-3))
            keep = f >= on - int(1.0 * P.fs)        # bounded memory: drop spikes > 1 s before this pulse
            P.frames, P.unitl = ([f[keep]], [u[keep]]) if len(f) else ([], [])
        nb = min(len(x) for x in Xs); X = np.concatenate([x[:nb] for x in Xs], axis=1)
        loss, _ = self.analysis(X, tr["cond"])
        t1 = time.perf_counter()
        tr["done"] = True
        row = dict(k=tr["k"], t=tr["t"], cond=str(tr["cond"]), n_units=int(X.shape[1]), loss=loss,
                   geometry_ms=(t1 - t0) * 1e3, pulse_to_loss_ms=(t1 - tr["t_pulse"]) * 1e3,
                   n_probes_pulse=len(tr["on"]))
        with self.lock:
            self.rows.append(row)
        if row["pulse_to_loss_ms"] > self.iti * 1e3:
            print(f"[worker] WARNING trial {tr['k']}: pulse->loss {row['pulse_to_loss_ms']:.0f} ms exceeds ITI", flush=True)

    # -- stimulator thread --------------------------------------------------------------
    def _stim_loop(self):
        while True:
            item = self.stim_q.get()
            if item is None:
                self.done.set(); return
            _, tr = item
            deadline = tr["t_pulse"] + self.iti - self.margin
            with self.lock:
                hist = list(self.rows)
            nxt = self.policy(hist)
            t0 = time.perf_counter()
            if nxt is None or t0 + getattr(self.stim, "reprog_s", 0.0) > deadline:
                tr["stim"] = "hold"                 # deadline cannot be met: keep current parameters
            else:
                self.stim.program(nxt); self.cond_now = nxt; tr["stim"] = "programmed"
                self._broadcast()
            self.stim_rows.append(dict(k=tr["k"], t=tr["t"], stim=tr["stim"], next=str(nxt),
                                       pulse_to_stim_ms=(time.perf_counter() - tr["t_pulse"]) * 1e3))

    def _broadcast(self):
        for P in self.P.values():
            if P.conn is not None:
                try:
                    P.conn.send(("condition", self.cond_now))
                except OSError:
                    pass

    def _write_log(self):
        path = self.cfg.get("log")
        if path:
            with open(path, "w") as f:
                json.dump(dict(trials=self.rows, stim=self.stim_rows), f, default=str)


def main(cfg: dict, ready=None, conns=None):
    return Worker(cfg).serve(ready=ready, conns=conns)
