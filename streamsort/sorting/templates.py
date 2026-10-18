"""Template bank: the hand-off between the offline sort and the online decoder.

A :class:`TemplateBank` is *signal-path specific*.  It is built by running raw
samples through exactly the same :class:`~streamsort.preprocess.CausalPreprocessor`
the online decoder uses, and it records that preprocessor's
:meth:`StreamConfig.fingerprint`.  The decoder refuses a bank whose fingerprint
differs from its own configuration.

Why not take the sorter's templates directly?  Two reasons measured on jlh32:

* SpikeInterface's ``highpass_filter`` defaults to zero-phase
  (forward-backward) filtering, which a live decoder cannot do.  A causal
  filter shapes waveforms differently, so templates from a zero-phase cache
  would be systematically mismatched online.
* Kilosort's own templates live in its whitened space.

Templates are stored sparsely on the ``n_local`` channels nearest each unit's
peak, which is what the online matcher compares against.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import StreamConfig
from ..preprocess.filters import CausalPreprocessor, robust_sigma


@dataclass
class TemplateBank:
    unit_ids: np.ndarray            # (U,) int
    templates: np.ndarray           # (U, T, L) float32, uV, causal signal path
    local_channels: np.ndarray      # (U, L) int, channel index for each local slot
    peak_channel: np.ndarray        # (U,) int
    trough_offset: np.ndarray       # (U,) int, trough sample index within T
    amplitude_uV: np.ndarray        # (U,) float, |trough| on peak channel
    n_spikes_used: np.ndarray       # (U,) int
    noise_uV: np.ndarray            # (C,) float, robust sd per channel
    channel_locations: np.ndarray   # (C, 2) um
    config: dict
    fingerprint: str
    unit_info: dict = field(default_factory=dict)   # per-unit extras (quality, match)
    source: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(len(self.unit_ids))

    @property
    def n_samples(self) -> int:
        return int(self.templates.shape[1])

    @property
    def n_local(self) -> int:
        return int(self.templates.shape[2])

    def dense(self, n_channels: int | None = None) -> np.ndarray:
        """(U, T, C) array with zeros off the local channels."""
        C = n_channels or len(self.noise_uV)
        out = np.zeros((self.n_units, self.n_samples, C), dtype=np.float32)
        for u in range(self.n_units):
            out[u][:, self.local_channels[u]] = self.templates[u]
        return out

    def check_compatible(self, cfg: StreamConfig) -> None:
        if cfg.fingerprint() != self.fingerprint:
            raise ValueError(
                "template bank was built under a different signal path "
                f"(bank {self.fingerprint}, decoder {cfg.fingerprint()}); "
                "rebuild the bank with this StreamConfig")

    # -- persistence -------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path); tmp = path.with_name(path.stem + ".tmp.npz")   # atomic: readers never see a partial file
        np.savez_compressed(
            tmp, unit_ids=self.unit_ids, templates=self.templates,
            local_channels=self.local_channels, peak_channel=self.peak_channel,
            trough_offset=self.trough_offset, amplitude_uV=self.amplitude_uV,
            n_spikes_used=self.n_spikes_used, noise_uV=self.noise_uV,
            channel_locations=self.channel_locations,
            meta=np.array(json.dumps({"config": self.config,
                                      "fingerprint": self.fingerprint,
                                      "unit_info": self.unit_info,
                                      "source": self.source}, default=float)),
        )
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path) -> "TemplateBank":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(unit_ids=z["unit_ids"], templates=z["templates"],
                   local_channels=z["local_channels"], peak_channel=z["peak_channel"],
                   trough_offset=z["trough_offset"], amplitude_uV=z["amplitude_uV"],
                   n_spikes_used=z["n_spikes_used"], noise_uV=z["noise_uV"],
                   channel_locations=z["channel_locations"], config=meta["config"],
                   fingerprint=meta["fingerprint"], unit_info=meta.get("unit_info", {}),
                   source=meta.get("source", {}))


def estimate_noise(src, cfg: StreamConfig, start: int, n: int,
                   chunk: int = 30_000) -> np.ndarray:
    """Per-channel robust noise sd (uV) on the causal signal path."""
    pp = CausalPreprocessor(cfg, src.n_channels_used)
    outs = [pp.process(np.asarray(src.read(s, min(chunk, start + n - s))))
            for s in range(start, start + n, chunk)]
    y = np.concatenate(outs, axis=0)[cfg.samples(50):]   # drop startup
    return robust_sigma(y)


def build_template_bank(src, cfg: StreamConfig, spike_frames: dict,
                        channel_locations: np.ndarray, noise_uV: np.ndarray,
                        exclude_windows: np.ndarray | None = None,
                        max_spikes: int = 300, n_local: int = 12,
                        warmup_ms: float = 15.0, seed: int = 0,
                        unit_info: dict | None = None,
                        source: dict | None = None) -> TemplateBank:
    """Average causally-preprocessed waveforms at given spike sample indices.

    Each spike is processed in its own short raw window with ``warmup_ms`` of
    lead-in so the causal filter has settled; the processed snippet then
    matches what a continuously running online preprocessor would emit (the
    residual is checked by :func:`windowed_filter_error`).  Spikes inside any
    ``exclude_windows`` (sample ranges around stimuli) are skipped, so stimulus
    artifact never enters a template.
    """
    rng = np.random.default_rng(seed)
    nb, na = cfg.n_before, cfg.n_after
    T = nb + na
    W = cfg.samples(warmup_ms)
    pp = CausalPreprocessor(cfg, src.n_channels_used)
    C = src.n_channels_used
    ex = None
    if exclude_windows is not None and len(exclude_windows):
        ex = np.asarray(exclude_windows, dtype=np.int64)
        ex = ex[np.argsort(ex[:, 0])]

    unit_ids = np.array(sorted(spike_frames), dtype=np.int64)
    U = len(unit_ids)
    tmpl = np.zeros((U, T, n_local), np.float32)
    loc = np.zeros((U, n_local), np.int64)
    pk = np.zeros(U, np.int64); tro = np.zeros(U, np.int64)
    amp = np.zeros(U, np.float32); nused = np.zeros(U, np.int64)

    for ui, u in enumerate(unit_ids):
        fr = np.asarray(spike_frames[u], dtype=np.int64)
        fr = fr[(fr - nb - W >= 0) & (fr + na < src.n_samples)]
        if ex is not None and len(fr):
            k = np.searchsorted(ex[:, 0], fr, side="right") - 1
            inside = (k >= 0) & (fr - nb - W <= ex[np.clip(k, 0, None), 1])
            # also reject if the window's lead-in overlaps the next blank
            k2 = np.searchsorted(ex[:, 0], fr - nb - W, side="right")
            nxt = ex[np.clip(k2, 0, len(ex) - 1), 0]
            overlap_next = (k2 < len(ex)) & (nxt <= fr + na)
            fr = fr[~(inside | overlap_next)]
        if len(fr) > max_spikes:
            fr = np.sort(rng.choice(fr, max_spikes, replace=False))
        if len(fr) == 0:
            continue
        acc = np.zeros((T, C), np.float64)
        for f in fr:
            pp.reset()
            y = pp.process(np.asarray(src.read(int(f) - nb - W, W + T)))
            acc += y[W:]
        mean = (acc / len(fr)).astype(np.float32)
        p = int(np.argmax(np.ptp(mean, axis=0)))
        d = np.linalg.norm(channel_locations[:C] - channel_locations[p], axis=1)
        lc = np.sort(np.argsort(d, kind="stable")[:n_local])
        tmpl[ui] = mean[:, lc]; loc[ui] = lc; pk[ui] = p
        tro[ui] = int(np.argmin(mean[:, p])); amp[ui] = float(-mean[:, p].min())
        nused[ui] = len(fr)

    return TemplateBank(unit_ids=unit_ids, templates=tmpl, local_channels=loc,
                        peak_channel=pk, trough_offset=tro, amplitude_uV=amp,
                        n_spikes_used=nused, noise_uV=np.asarray(noise_uV, np.float32),
                        channel_locations=np.asarray(channel_locations[:C], np.float32),
                        config=cfg.to_dict(), fingerprint=cfg.fingerprint(),
                        unit_info=unit_info or {}, source=source or {})


def windowed_filter_error(src, cfg: StreamConfig, frames, warmup_ms: float = 15.0) -> float:
    """Max relative error of windowed causal filtering vs a continuous run.

    Validates that the per-spike windows in :func:`build_template_bank` emit
    the same samples a continuously running online preprocessor would.
    """
    nb, na = cfg.n_before, cfg.n_after
    W = cfg.samples(warmup_ms)
    worst = 0.0
    for f in frames:
        lead = cfg.samples(500)          # long continuous run as the reference
        ref = CausalPreprocessor(cfg, src.n_channels_used).process(
            np.asarray(src.read(int(f) - nb - lead, lead + nb + na)))[lead:]
        win = CausalPreprocessor(cfg, src.n_channels_used).process(
            np.asarray(src.read(int(f) - nb - W, W + nb + na)))[W:]
        worst = max(worst, float(np.abs(win - ref).max() / (np.abs(ref).max() + 1e-9)))
    return worst
