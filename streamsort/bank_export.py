"""Decoder bank from ANY Kilosort output folder (``streamsort bank --ks-folder``).

Reads spike_times / spike_clusters (+ cluster_KSLabel.tsv, channel_map / channel_positions) with
numpy only, re-extracts data-mean templates through the causal streamsort signal path (Phase 1
recipe: <=300 spikes/unit, 12 local channels, stimulus windows excluded), estimates noise and local
whitening from a stimulus-free stretch, applies the Phase 1 usability rule, builds the decoder and
writes, into ``<ks_folder>/streamsort/`` (or ``--out``):

  decoder tables for the C++ core / Open Ephys plugin  (*.npy + decoder_tables.txt)
  templates.npz, whitening.npz, sorting_report.csv     (Python paths and the worker's unit list)
  bank_info.json                                       (provenance)
"""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from .config import StreamConfig
from .io import OpenEphysSession, FileStreamSource
from .online.decoder import OnlineDecoder, DecoderParams
from .online.whitening import estimate_whitening
from .preprocess.suppressor import ArtifactSuppressor
from .sorting.templates import build_template_bank, estimate_noise
from .core_cpp import tables_from_decoder, export_tables


def read_ks_folder(ks: Path, offset_samples: int = 0) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Spike frames per cluster, KS labels, channel map, channel positions."""
    ks = Path(ks)
    st = np.load(ks / "spike_times.npy").astype(np.int64).ravel() + int(offset_samples)
    cl = np.load(ks / ("spike_clusters.npy" if (ks / "spike_clusters.npy").exists() else "spike_templates.npy")).astype(np.int64).ravel()
    order = np.argsort(cl, kind="stable"); st, cl = st[order], cl[order]
    u, start = np.unique(cl, return_index=True); stops = np.r_[start[1:], len(cl)]
    frames = {int(k): np.sort(st[a:b]) for k, a, b in zip(u, start, stops)}
    labels = {}
    for f in ("cluster_KSLabel.tsv", "cluster_group.tsv"):
        if (ks / f).exists():
            with open(ks / f) as fh:
                r = csv.reader(fh, delimiter="\t"); next(r, None)
                for row in r:
                    if len(row) >= 2:
                        labels[int(row[0])] = row[1]
            break
    chan_map = np.load(ks / "channel_map.npy").astype(np.int64).ravel() if (ks / "channel_map.npy").exists() else None
    pos = np.load(ks / "channel_positions.npy") if (ks / "channel_positions.npy").exists() else None
    return frames, labels, chan_map, pos


def usable_mask(bank) -> np.ndarray:
    """Phase 1 rule: >=50 template spikes, trough offset 20-40, negative-dominant on the peak channel."""
    pk = [list(bank.local_channels[k]).index(bank.peak_channel[k]) for k in range(bank.n_units)]
    negdom = np.array([abs(bank.templates[k][:, p].min()) >= bank.templates[k][:, p].max() for k, p in enumerate(pk)])
    return (bank.n_spikes_used >= 50) & (bank.trough_offset >= 20) & (bank.trough_offset <= 40) & negdom


def build_bank(ks_folder, rec_dir, stream: str, nch: int, out=None, ttl_stream: str | None = "NI-DAQmx-107.PXI-6133",
               ttl_line: int = 5, offset_samples: int = 0, noise_s=(10.0, 30.0), whiten_s=(60.0, 20.0),
               detect_threshold: float = 4.0, min_energy: float = 50.0, positions=None, log=print) -> Path:
    ks = Path(ks_folder); out = Path(out) if out else ks / "streamsort"
    sess = OpenEphysSession(rec_dir); st = sess.stream(stream)
    base = StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts)
    cfg = dataclasses.replace(base, n_channels_used=int(nch))
    src = FileStreamSource(st, n_channels_used=int(nch))
    frames, labels, chan_map, kpos = read_ks_folder(ks, offset_samples)
    if positions is not None:
        pos = np.asarray(positions, float)[:nch]
    elif kpos is not None and chan_map is not None:
        full = np.full((max(int(chan_map.max()) + 1, nch), 2), np.nan); full[chan_map] = kpos[:, :2]
        pos = full[:nch]
    elif (st.path / "channel_positions.npy").exists():
        pos = np.load(st.path / "channel_positions.npy")[:nch]
    else:
        raise ValueError("no channel positions: KS folder lacks channel_map/channel_positions and none given")
    if np.isnan(pos).any():
        raise ValueError(f"channel positions missing for some of the first {nch} channels (KS sorted a subset?)")
    # stimulus windows (TTL), exact clock
    excl, first_pulse_s = None, float("inf")
    if ttl_stream:
        try:
            up = sess.ttl_edges(ttl_stream, ttl_line, True)
            on = st.time_to_sample(np.asarray(up, float))
            on = on[(on >= 0) & (on < st.n_samples)]
            excl = np.stack([on - cfg.samples(1.0), on + cfg.samples(8.0)], axis=1) if len(on) else None
            first_pulse_s = float(on.min() / st.effective_rate) if len(on) else float("inf")
        except Exception as e:                       # no TTL folder on this recording
            log(f"[bank] no stim TTL ({e}); templates use all spikes")
    t0 = time.perf_counter()
    noise = estimate_noise(src, cfg, start=cfg.samples(noise_s[0] * 1e3), n=cfg.samples(noise_s[1] * 1e3))
    info = {str(u): dict(ks_label=labels.get(u, "")) for u in frames}
    bank = build_template_bank(src, cfg, frames, pos, noise, exclude_windows=excl, max_spikes=300, n_local=12,
                               warmup_ms=15.0, seed=0, unit_info=info,
                               source=dict(sorter="kilosort", ks_folder=str(ks.resolve()), offset_samples=int(offset_samples),
                                           note="spike times from the Kilosort folder; waveforms re-extracted through "
                                                "the streamsort causal signal path"))
    usable = usable_mask(bank)
    log(f"[bank] {bank.n_units} units, usable {int(usable.sum())} ({time.perf_counter() - t0:.1f} s)")
    if whiten_s[0] + whiten_s[1] >= first_pulse_s - 1:
        raise ValueError(f"whitening window {whiten_s} s overlaps stimulation (first pulse {first_pulse_s:.1f} s); "
                         "pass --whiten-start/--whiten-dur inside a stimulus-free stretch")
    wcfg = dataclasses.replace(cfg, detect_threshold=4.0)
    wh = estimate_whitening(src, wcfg, pos, start=wcfg.samples(whiten_s[0] * 1e3), n=wcfg.samples(whiten_s[1] * 1e3), k=32)
    dcfg = dataclasses.replace(cfg, detect_threshold=float(detect_threshold))
    dec = OnlineDecoder(bank, dcfg, wh, DecoderParams(min_energy=float(min_energy)), units_mask=usable, check=False,
                        suppressor=ArtifactSuppressor(dcfg.sample_rate, int(nch)))
    out.mkdir(parents=True, exist_ok=True)
    bank.save(out / "templates.npz"); wh.save(out / "whitening.npz")
    with open(out / "sorting_report.tmp", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["unit_id", "peak_channel", "trough_offset", "n_spikes_template", "ks_label", "usable"])
        for k, u in enumerate(bank.unit_ids):
            w.writerow([int(u), int(bank.peak_channel[k]), int(bank.trough_offset[k]), int(bank.n_spikes_used[k]),
                        labels.get(int(u), ""), bool(usable[k])])
    (out / "sorting_report.tmp").replace(out / "sorting_report.csv")
    export_tables(tables_from_decoder(dec), out, extra=dict(
        sample_rate=st.effective_rate, nominal_rate=st.sample_rate, n_units=bank.n_units, n_usable=int(usable.sum()),
        fingerprint=dcfg.fingerprint(), stream=stream, nch=int(nch)))
    (out / "bank_info.json").write_text(json.dumps(dict(
        ks_folder=str(ks.resolve()), rec_dir=str(Path(rec_dir).resolve()), stream=stream, nch=int(nch),
        n_units=int(bank.n_units), n_usable=int(usable.sum()), noise_s=list(noise_s), whiten_s=list(whiten_s),
        ttl_stream=ttl_stream, ttl_line=ttl_line, offset_samples=int(offset_samples), sample_rate=st.effective_rate,
        detect_threshold=detect_threshold, min_energy=min_energy, fingerprint=dcfg.fingerprint()), indent=1))
    return out


def export_from_bank_files(bank_dir, rec_dir, stream: str, nch: int, folders, detect_threshold: float = 4.0,
                           min_energy: float = 50.0) -> list:
    """Plugin tables from an existing templates.npz / whitening.npz / sorting_report.csv (``streamsort fit`` output)."""
    import pandas as pd
    from .online.whitening import Whitening
    from .sorting.templates import TemplateBank
    bank_dir = Path(bank_dir)
    st = OpenEphysSession(rec_dir).stream(stream)
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts),
                              detect_threshold=float(detect_threshold), n_channels_used=int(nch))
    bank = TemplateBank.load(bank_dir / "templates.npz"); wh = Whitening.load(bank_dir / "whitening.npz")
    usable = pd.read_csv(bank_dir / "sorting_report.csv").usable.to_numpy(bool)
    dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=float(min_energy)), units_mask=usable, check=False,
                        suppressor=ArtifactSuppressor(cfg.sample_rate, int(nch)))
    tab = tables_from_decoder(dec); out = []
    for f in folders:
        out.append(export_tables(tab, f, extra=dict(sample_rate=st.effective_rate, nominal_rate=st.sample_rate,
                                                    n_units=bank.n_units, n_usable=int(usable.sum()),
                                                    fingerprint=cfg.fingerprint(), stream=stream, nch=int(nch))))
    return out
