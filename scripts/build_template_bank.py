#!/usr/bin/env python
"""Build the online template bank from the offline initial sort.

Spike *times and identities* come from the Kilosort 4 initial sort.  Waveforms
are re-extracted from raw data through streamsort's causal preprocessor, so the
bank lives in exactly the signal space the online decoder sees.

Outputs: results/templates_jlh32_probeA.npz, results/sorting_report.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from streamsort.io import fsguard  # noqa: E402
fsguard.install()

import spikeinterface.full as si  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import OpenEphysSession, FileStreamSource  # noqa: E402
from streamsort.sorting import build_template_bank, estimate_noise  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx  # noqa: E402
ROOT, REC, SCRATCH = ctx.ROOT, ctx.REC_DIR, ctx.SCRATCH
TAG, SORTER = os.environ.get("SEG_TAG", "seg600"), os.environ.get("SORTER", "kilosort4")


def main() -> int:
    sess = OpenEphysSession(REC)
    st = sess.stream(ctx.STREAM)
    import dataclasses
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), n_channels_used=ctx.NCH)
    src = FileStreamSource(st, n_channels_used=cfg.n_channels_used)
    pos = np.load(st.path / "channel_positions.npy")[:cfg.n_channels_used]

    an = si.load_sorting_analyzer(str(SCRATCH / f"analyzer_{SORTER}_{TAG}"))
    sorting = an.sorting
    n_frames = an.get_num_samples()
    frames = {int(u): sorting.get_unit_spike_train(u).astype(np.int64)
              for u in sorting.unit_ids}

    # stimulus exclusion windows inside the segment (exact clock)
    trials = ctx.load_trials(sess)
    on = st.time_to_sample(trials["ttl_onset"].to_numpy())
    on = on[(on >= 0) & (on < n_frames)]
    # TTL-to-artifact jitter reaches +-0.8 ms and clipping lasts <= 2.4 ms;
    # 1 ms before to 8 ms after the TTL keeps every artifact out of templates.
    excl = np.stack([on - cfg.samples(1.0), on + cfg.samples(8.0)], axis=1)

    # noise on the causal path, from the stimulus-free lead-in
    t0 = time.perf_counter()
    noise = estimate_noise(src, cfg, start=cfg.samples(10_000), n=cfg.samples(30_000))
    print(f"[bank] noise: median {np.median(noise):.2f} uV "
          f"({time.perf_counter()-t0:.1f} s)", flush=True)

    # per-unit info: KS4 quality, KS label, NWB match
    qm = pd.read_csv(SCRATCH / f"quality_metrics_{SORTER}_{TAG}.csv", index_col=0)
    lab_path = SCRATCH / f"sort_{SORTER}_{TAG}" / "sorter_output" / "cluster_KSLabel.tsv"
    ks_lab = (pd.read_csv(lab_path, sep="\t").set_index("cluster_id")["KSLabel"]
              if lab_path.exists() else pd.Series(dtype=str))
    rep = (pd.read_csv(ctx.R("sort_agreement_report.csv"), index_col=0) if ctx.R("sort_agreement_report.csv").exists()
           else pd.DataFrame(columns=["new_id", "nwb_unit_id", "agreement", "group"]))   # no reference sort on a live rig
    m = rep[rep.new_id != -1]
    match = {int(r.new_id): (str(r.nwb_unit_id), float(r.agreement), str(r.group))
             for _, r in m.iterrows()}

    info = {}
    for u in frames:
        nwb = match.get(u)
        info[str(u)] = dict(
            ks4_label=str(ks_lab.get(u, "")),
            snr=float(qm.loc[u, "snr"]) if u in qm.index else float("nan"),
            firing_rate=float(qm.loc[u, "firing_rate"]) if u in qm.index else float("nan"),
            isi_violations_ratio=float(qm.loc[u, "isi_violations_ratio"]) if u in qm.index else float("nan"),
            nwb_unit_id=nwb[0] if nwb else None,
            nwb_agreement=nwb[1] if nwb else 0.0,
            nwb_group=nwb[2] if nwb else None,
        )

    t0 = time.perf_counter()
    bank = build_template_bank(
        src, cfg, frames, pos, noise, exclude_windows=excl,
        max_spikes=300, n_local=12, warmup_ms=15.0, seed=0, unit_info=info,
        source=dict(sorter=SORTER, tag=TAG, segment_frames=[0, int(n_frames)],
                    note="spike times from Kilosort 4; waveforms re-extracted "
                         "through streamsort CausalPreprocessor"))
    dt = time.perf_counter() - t0
    out = ctx.R("templates.npz")
    bank.save(out)
    print(f"[bank] {bank.n_units} units, T={bank.n_samples}, L={bank.n_local}, "
          f"built in {dt:.1f} s -> {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"[bank] fingerprint {bank.fingerprint}", flush=True)

    rows = []
    for k, u in enumerate(bank.unit_ids):
        d = info[str(int(u))]
        rows.append(dict(unit_id=int(u), n_spikes_segment=int(len(frames[int(u)])),
                         n_spikes_template=int(bank.n_spikes_used[k]),
                         peak_channel=int(bank.peak_channel[k]),
                         x_um=float(pos[bank.peak_channel[k], 0]),
                         depth_um=float(pos[bank.peak_channel[k], 1]),
                         trough_uV=float(bank.amplitude_uV[k]),
                         snr_causal=float(bank.amplitude_uV[k] / noise[bank.peak_channel[k]]),
                         **d))
    rep_out = pd.DataFrame(rows)
    # usable = enough spikes for a template, trough where the decoder expects it, negative-dominant
    pk_local = [list(bank.local_channels[k]).index(bank.peak_channel[k]) for k in range(bank.n_units)]
    negdom = np.array([abs(bank.templates[k][:, p].min()) >= bank.templates[k][:, p].max() for k, p in enumerate(pk_local)])
    rep_out["usable"] = (bank.n_spikes_used >= 50) & (bank.trough_offset >= 20) & (bank.trough_offset <= 40) & negdom
    rep_out["trough_offset"] = bank.trough_offset
    rep_out.to_csv(ctx.R("sorting_report.csv"), index=False)
    print(f"[bank] usable {int(rep_out.usable.sum())} / {bank.n_units}", flush=True)
    print("[bank] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
