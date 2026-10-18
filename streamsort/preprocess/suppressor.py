"""Streaming, causal stimulus-artifact suppression (raw domain, before filtering).

Two operations per stimulus event, both validated on jlh32 (Phase 1, step 6):

1. **Template subtraction.**  For a stimulus condition with at least
   ``min_history`` past events, subtract the median raw artifact of the last
   ``n_history`` events of that condition (onset-aligned, each referenced to its
   own 1 ms pre-onset mean, cosine-tapered over the final ``taper_ms`` so that
   the evoked-LFP offset it carries does not leave a step).  Templates use
   *past* events only, so the stage is causal and works live: the condition key
   comes from the stimulus controller via :meth:`set_condition`.
2. **Bridging.**  The clipped span (+ ``pre_ms`` / ``post_ms`` pads) is
   replaced by a straight line between the 3-sample means just outside it.
   Unlike sample-and-hold this leaves no step at release, so the highpass does
   not ring.

Output is float32 raw counts, delayed by a bounded look-ahead: ``pre`` pad +
1 sample normally, and while an event is unresolved until the whole
``max_span_ms`` search window has arrived (about 5.5 ms worst case).  Clips can
recur after a >1 ms clean gap (seen on jlh32), so an earlier decision would make
the span depend on buffer timing.  Concatenated
output is independent of how the input is chunked.
"""
from __future__ import annotations
import collections
import numpy as np
from .artifact import CLIP_COUNTS


class ArtifactSuppressor:
    def __init__(self, sample_rate: float, n_channels: int, *, clip_counts: int = CLIP_COUNTS,
                 clip_min_channels: int = 3, template_ms: float = 20.0, taper_ms: float = 5.0,
                 n_history: int = 20, min_history: int = 5, pre_ms: float = 0.2, post_ms: float = 0.5,
                 max_span_ms: float = 5.0, baseline_ms: float = 1.0,
                 subtract: bool = True, bridge: bool = True):
        S = lambda ms: int(round(ms * 1e-3 * sample_rate))
        self.C, self.clip, self.kmin = n_channels, clip_counts, clip_min_channels
        self.L, self.pre, self.post = S(template_ms), S(pre_ms), S(post_ms)
        self.maxspan, self.nb = S(max_span_ms), S(baseline_ms)
        self.n_history, self.min_history = n_history, min_history
        self.subtract, self.bridge = subtract, bridge
        k = S(taper_ms)
        self.taper = np.ones(self.L, np.float32)
        self.taper[-k:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
        self.H = self.nb + self.pre + 3            # raw history kept behind the emit point
        self.history: dict = collections.defaultdict(lambda: collections.deque(maxlen=self.n_history))
        self._tpl_cache: dict = {}
        self.events: list[dict] = []               # log: onset, span, condition, n_history used
        self._cond = None
        self.reset(0)

    # -- control --------------------------------------------------------------
    def reset(self, start_sample: int = 0) -> None:
        """Reset stream state (keeps per-condition template history)."""
        self._x = np.zeros((0, self.C), np.float32)  # working buffer (raw, modified in place)
        self._raw = np.zeros((0, self.C), np.float32)  # untouched copy, same span
        self._x0 = int(start_sample)                 # global index of _x[0]
        self._emit = int(start_sample)               # next global sample to emit
        self._scan = int(start_sample)               # next sample to scan for onset
        self._ev = None                              # active event

    def set_condition(self, key) -> None:
        """Condition of the *next* stimulus event (hashable; None = bridge only)."""
        self._cond = key

    def clear_history(self) -> None:
        self.history.clear(); self._tpl_cache.clear()

    def template(self, key):
        h = self.history.get(key)
        if key is None or h is None or len(h) < self.min_history:
            return None
        if key not in self._tpl_cache:
            self._tpl_cache[key] = (np.median(np.stack(h), 0) * self.taper[:, None]).astype(np.float32)
        return self._tpl_cache[key]

    # -- streaming --------------------------------------------------------------
    def push(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        self._x = np.concatenate([self._x, raw]); self._raw = np.concatenate([self._raw, raw])
        return self._advance(final=False)

    def flush(self) -> np.ndarray:
        return self._advance(final=True)

    def _advance(self, final: bool) -> np.ndarray:
        end = self._x0 + len(self._x)
        while True:
            if self._ev is None:
                lo = self._scan - self._x0
                seg = self._x[lo:]
                hit = np.flatnonzero((np.abs(seg) >= self.clip).sum(1) >= self.kmin) if len(seg) else []
                if len(hit) == 0:
                    self._scan = end
                    break
                on = self._x0 + lo + int(hit[0])
                self._start_event(on)
            if not self._resolve(end, final):
                break
            if not self._finish_template(end, final):
                break
        # emit: everything that can no longer change
        if final:
            limit = end
        elif self._ev is None:
            limit = end - (self.pre + 1)            # a clip at `end` may still pad back `pre`
        elif not self._ev["resolved"]:
            limit = self._ev["b0"]                  # wait for the bridge / span decision
        else:
            limit = min(end, self._ev["on"] + self.L - (self.pre + 1))
        limit = max(limit, self._emit)
        out = self._x[self._emit - self._x0:limit - self._x0].copy()
        self._emit = limit
        keep = max(self._emit - self.H, self._x0)
        if self._ev is not None:
            keep = min(keep, self._ev["on"] - self.nb)
        self._x = self._x[keep - self._x0:]; self._raw = self._raw[keep - self._x0:]; self._x0 = keep
        return out

    def _start_event(self, on: int) -> None:
        cond = self._cond
        self._ev = dict(on=on, cond=cond, tpl=self.template(cond) if self.subtract else None,
                        last_clip=on, resolved=False, bridged=False, sub_upto=on,
                        b0=on - self.pre, b1=None, n_hist=len(self.history.get(cond, ())))

    def _resolve(self, end: int, final: bool) -> bool:
        ev = self._ev
        if not ev["resolved"]:
            on = ev["on"]
            a = ev["last_clip"] + 1 - self._x0
            hi = min(end, on + self.maxspan) - self._x0
            if hi > a:
                cl = np.flatnonzero((np.abs(self._x[a:hi]) >= self.clip).any(1))
                if len(cl):
                    ev["last_clip"] = self._x0 + a + int(cl[-1])
            done_scan = end >= on + self.maxspan   # full search window: chunk-invariant, = offline clipped_span
            span = ev["last_clip"] + 1 - on
            b1 = on + span + self.post
            if not ((done_scan and end >= b1 + 3) or final):
                ev["scan_upto"] = end
                return False
            ev.update(resolved=True, span=span, b1=b1)
        self._apply(end)
        return True

    def _apply(self, end: int) -> None:
        ev = self._ev
        # subtraction on [sub_upto, min(end, on+L))
        if ev["tpl"] is not None:
            a, b = ev["sub_upto"], min(end, ev["on"] + self.L)
            if b > a:
                self._x[a - self._x0:b - self._x0] -= ev["tpl"][a - ev["on"]:b - ev["on"]]
            ev["sub_upto"] = max(a, b)
        if self.bridge and not ev["bridged"] and end >= ev["b1"] + 3 and ev["b0"] - 3 >= self._x0:
            b0, b1 = ev["b0"] - self._x0, ev["b1"] - self._x0
            l, r = self._x[b0 - 3:b0].mean(0), self._x[b1:b1 + 3].mean(0)
            w = np.linspace(0, 1, b1 - b0, dtype=np.float32)[:, None]
            self._x[b0:b1] = (1 - w) * l + w * r
            ev["bridged"] = True
        elif not self.bridge:
            ev["bridged"] = True

    def _finish_template(self, end: int, final: bool) -> bool:
        ev = self._ev
        on = ev["on"]
        if end < on + self.L and not final:
            return False
        if end >= on + self.L and ev["cond"] is not None and on - self.nb >= self._x0:
            r = self._raw[on - self._x0:on + self.L - self._x0]
            base = self._raw[on - self.nb - self._x0:on - self._x0].mean(0)
            self.history[ev["cond"]].append((r - base).astype(np.float32))
            self._tpl_cache.pop(ev["cond"], None)
        self.events.append(dict(onset=on, span=ev.get("span"), condition=ev["cond"], n_history=ev["n_hist"],
                                subtracted=ev["tpl"] is not None))
        self._scan = on + self.L
        self._ev = None
        self._cond = None
        return True
