#!/usr/bin/env python
"""Step 6: validate the composite pipeline (KS4 initial sort -> online decoder)
against the lab's full-session Kilosort 2.5 sort (NWB), on every stimulus trial.

For each of the 1923 trials, decode the window [pulse offset + 1 ms, + 1.9 s)
with the online decoder (pull mode, data-derived blanking).  Units are compared
through the Hungarian KS4 -> KS2.5 map from the gate step.

Outputs
  results/online_spikes_all_trials.npz   decoded spikes (all trials) for steps 7-8
  results/reference_spikes_all_trials.npz NWB KS2.5 spikes in the same windows
  results/validation_jlh32_probeA.csv     per matched unit: precision/recall/F1,
                                          counts, early/late, in-sample/held-out
  results/validation_by_latency.csv       accuracy vs time since pulse offset
  results/validation_trial_counts.csv     per (trial, unit) spike counts, both sorts
Memory-bounded (MemGuard 3.5 GB).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamsort.memguard import preflight, MemGuard  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import (OpenEphysSession, FileStreamSource, load_nwb_trials,  # noqa: E402
                           load_nwb_units, reconcile_with_ttl)
from streamsort.sorting import TemplateBank  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window  # noqa: E402
from streamsort.preprocess import ArtifactSuppressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx  # noqa: E402
ROOT = ctx.ROOT
TOL = 12                      # samples, 0.4 ms
WIN_START_MS, WIN_S = 1.0, 1.9
SEG_END_S = 600.0             # initial-sort segment (in-sample boundary)
LAT_EDGES_MS = [1, 3, 5, 10, 20, 50, 100, 300, 1000, 1901]


def match_count(a, b, tol=TOL):
    """Greedy one-to-one count of a-b coincidences within tol (both sorted)."""
    i = j = m = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol:
            m += 1; i += 1; j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return m


def main():
    preflight(3.5, margin_gb=0.0, strict=False)
    guard = MemGuard(3.5, "validate_online")
    sess = OpenEphysSession(ctx.REC_DIR)
    st = sess.stream(ctx.STREAM)
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts),
                              detect_threshold=4.0, n_channels_used=ctx.NCH)
    src = FileStreamSource(st, ctx.NCH)
    print(f"[val] {ctx.describe()}", flush=True)
    bank = TemplateBank.load(ctx.R("templates.npz"))
    rep = pd.read_csv(ctx.R("sorting_report.csv"))
    usable = rep.usable.to_numpy(bool)
    wh = Whitening.load(ctx.R("whitening.npz"))
    SUP = os.environ.get("SUPPRESS", "0") == "1"
    SUF = "_sup" if SUP else ""
    sup = ArtifactSuppressor(cfg.sample_rate, ctx.NCH) if SUP else None
    dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False,
                        suppressor=sup)
    print(f"[val] artifact suppressor: {SUP}", flush=True)

    trials = pd.read_csv(ctx.R("trials.csv"))        # pulse offsets already corrected (jlh34 TTL gate)
    keys = ctx.cond_keys(trials)
    limit = int(os.environ.get("VAL_LIMIT", "0"))
    if limit:
        trials = trials.iloc[:limit].copy()
    off = st.time_to_sample(trials["ttl_offset"].to_numpy())
    w0 = off + cfg.samples(WIN_START_MS)
    nW = cfg.samples(WIN_S * 1e3)
    seg_end_frame = st.time_to_sample(st.t_start + SEG_END_S)

    # unit map from the gate: KS4 unit -> NWB unit (agreement > 0.5)
    gate = pd.read_csv(ctx.R("sort_agreement_report.csv"), index_col=0)
    gate = gate[(gate.new_id != -1) & (gate.agreement > 0.5)]
    ks4_ids = set(bank.unit_ids[usable].tolist())
    pairs = {int(r.new_id): (int(i), str(r.nwb_unit_id), float(r.agreement), float(r.accuracy))
             for i, r in gate.iterrows() if int(r.new_id) in ks4_ids}
    print(f"[val] matched usable unit pairs: {len(pairs)}", flush=True)

    # reference spikes (NWB KS2.5, probe A) -> exact sample indices
    U = load_nwb_units(ctx.NWB, probe=ctx.PROBE)
    ref_tr = {}
    for ks4_u, (ref_row, nwb_id, _, _) in pairs.items():
        ref_tr[ks4_u] = np.sort(st.time_to_sample(np.asarray(U.loc[ref_row, "spike_times"])))
    guard.check("reference loaded")

    # decode all trials
    on_f, on_u, on_t, on_sc = [], [], [], []
    rf_f, rf_u, rf_t = [], [], []
    lat = []
    t_all = time.perf_counter()
    for k in range(len(trials)):
        t0 = time.perf_counter()
        ck = keys[k]
        b = decode_window(dec, src, int(w0[k]), nW, condition=ck)
        lat.append(time.perf_counter() - t0)
        m = b.unit >= 0
        on_f.append(b.frame[m]); on_u.append(b.unit[m]); on_t.append(np.full(m.sum(), k, np.int32))
        on_sc.append(b.score[m])
        for u, tr in ref_tr.items():
            a = np.searchsorted(tr, w0[k]); z = np.searchsorted(tr, w0[k] + nW)
            if z > a:
                rf_f.append(tr[a:z]); rf_u.append(np.full(z - a, u)); rf_t.append(np.full(z - a, k, np.int32))
        if k % 200 == 0:
            guard.check(f"trial {k}")
            print(f"[val] trial {k}/{len(trials)} | median decode {np.median(lat)*1e3:.0f} ms | {guard.report()}",
                  flush=True)
    print(f"[val] decoded {len(trials)} trials in {time.perf_counter()-t_all:.0f} s; per-trial decode median "
          f"{np.median(lat)*1e3:.0f} ms, p99 {np.percentile(lat,99)*1e3:.0f} ms, max {np.max(lat)*1e3:.0f} ms",
          flush=True)
    on = dict(frame=np.concatenate(on_f), unit=np.concatenate(on_u), trial=np.concatenate(on_t),
              score=np.concatenate(on_sc))
    rf = dict(frame=np.concatenate(rf_f), unit=np.concatenate(rf_u), trial=np.concatenate(rf_t))
    meta = dict(window_start=w0, window_len=nW, pulse_offset=off, trial_run=trials["run"].to_numpy(),
                fs=cfg.sample_rate)
    np.savez_compressed(ctx.R(f"online_spikes_all_trials{SUF}.npz"), **on, **{f"meta_{k}": v for k, v in meta.items()})
    np.savez_compressed(ctx.R(f"reference_spikes_all_trials{SUF}.npz"), **rf,
                        ks4_to_nwb=np.array([[u, int(p[1])] for u, p in pairs.items()]))
    if sup is not None:
        pd.DataFrame(sup.events).to_csv(ctx.R(f"suppressor_events{SUF}.csv"), index=False)
    guard.check("saved spikes")

    # per-unit validation ----------------------------------------------------
    early_q = np.quantile(w0, 0.25); late_q = np.quantile(w0, 0.75)
    rows, lat_rows, cnt_rows = [], [], []
    lat_edges = np.array(LAT_EDGES_MS) * 1e-3 * cfg.sample_rate
    lat_tp = np.zeros(len(LAT_EDGES_MS) - 1); lat_on = np.zeros_like(lat_tp); lat_rf = np.zeros_like(lat_tp)
    o_order = np.lexsort((on["frame"], on["unit"])); of, ou, ot = on["frame"][o_order], on["unit"][o_order], on["trial"][o_order]
    r_order = np.lexsort((rf["frame"], rf["unit"])); rff, rfu, rft = rf["frame"][r_order], rf["unit"][r_order], rf["trial"][r_order]
    for u, (ref_row, nwb_id, gate_agr, gate_acc) in pairs.items():
        a0, a1 = np.searchsorted(ou, u), np.searchsorted(ou, u, side="right")
        b0, b1 = np.searchsorted(rfu, u), np.searchsorted(rfu, u, side="right")
        fo, to = of[a0:a1], ot[a0:a1]
        fr, tr_ = rff[b0:b1], rft[b0:b1]
        res = dict(ks4_unit=u, nwb_unit=nwb_id, gate_agreement=gate_agr, gate_accuracy=gate_acc)
        for name, so, sr in [("all", slice(None), slice(None)),
                             ("insample", fo < seg_end_frame, fr < seg_end_frame),
                             ("heldout", fo >= seg_end_frame, fr >= seg_end_frame),
                             ("early", fo < early_q, fr < early_q),
                             ("late", fo >= late_q, fr >= late_q)]:
            A_, B_ = np.sort(fo[so]), np.sort(fr[sr])
            m = match_count(A_, B_)
            res[f"n_online_{name}"] = len(A_); res[f"n_ref_{name}"] = len(B_); res[f"tp_{name}"] = m
            res[f"precision_{name}"] = m / max(len(A_), 1); res[f"recall_{name}"] = m / max(len(B_), 1)
            res[f"f1_{name}"] = 2 * m / max(len(A_) + len(B_), 1)
        # latency bins (time since pulse offset)
        lo_on = fo - off[to]; lo_rf = fr - off[tr_]
        for i in range(len(lat_tp)):
            sa = np.sort(fo[(lo_on >= lat_edges[i]) & (lo_on < lat_edges[i + 1])])
            sb = np.sort(fr[(lo_rf >= lat_edges[i]) & (lo_rf < lat_edges[i + 1])])
            lat_tp[i] += match_count(sa, sb); lat_on[i] += len(sa); lat_rf[i] += len(sb)
        # per-trial counts
        co = np.bincount(to, minlength=len(trials)); cr = np.bincount(tr_, minlength=len(trials))
        for k in np.flatnonzero((co > 0) | (cr > 0)):
            cnt_rows.append((k, u, int(co[k]), int(cr[k])))
        res["count_corr_trials"] = float(np.corrcoef(co, cr)[0, 1]) if co.std() > 0 and cr.std() > 0 else np.nan
        rows.append(res)
    V = pd.DataFrame(rows)
    amp = pd.Series(bank.amplitude_uV, index=bank.unit_ids)
    V["template_trough_uV"] = V.ks4_unit.map(amp)
    V["ref_rate_hz_in_windows"] = V.n_ref_all / (len(trials) * WIN_S)
    V.to_csv(ctx.R(f"validation_units{SUF}.csv"), index=False)
    LT = pd.DataFrame(dict(lat_lo_ms=LAT_EDGES_MS[:-1], lat_hi_ms=LAT_EDGES_MS[1:], tp=lat_tp,
                           n_online=lat_on, n_ref=lat_rf))
    LT["precision"] = LT.tp / LT.n_online.clip(lower=1); LT["recall"] = LT.tp / LT.n_ref.clip(lower=1)
    LT.to_csv(ctx.R(f"validation_by_latency{SUF}.csv"), index=False)
    pd.DataFrame(cnt_rows, columns=["trial", "ks4_unit", "n_online", "n_ref"]).to_csv(
        ctx.R(f"validation_trial_counts{SUF}.csv"), index=False)

    tot = {k: int(V[f"tp_{k}"].sum()) for k in ("all", "insample", "heldout", "early", "late")}
    summ = {}
    for k in tot:
        summ[k] = dict(precision=float(V[f"tp_{k}"].sum() / max(V[f"n_online_{k}"].sum(), 1)),
                       recall=float(V[f"tp_{k}"].sum() / max(V[f"n_ref_{k}"].sum(), 1)),
                       median_unit_f1=float(V[f"f1_{k}"].median()))
    summ["n_pairs"] = len(V)
    summ["units_f1_ge_0.8"] = int((V.f1_all >= 0.8).sum()); summ["units_f1_ge_0.9"] = int((V.f1_all >= 0.9).sum())
    summ["median_count_corr"] = float(V.count_corr_trials.median())
    summ["decode_latency_ms"] = dict(median=float(np.median(lat) * 1e3), p99=float(np.percentile(lat, 99) * 1e3),
                                     max=float(np.max(lat) * 1e3))
    json.dump(summ, open(ctx.R(f"validation_summary{SUF}.json"), "w"), indent=2)
    for k in ("all", "insample", "heldout", "early", "late"):
        s = summ[k]
        print(f"[val] {k:9s}: precision {s['precision']:.3f} recall {s['recall']:.3f} median unit F1 {s['median_unit_f1']:.3f}",
              flush=True)
    print(f"[val] units F1>=0.8: {summ['units_f1_ge_0.8']} / {len(V)} | F1>=0.9: {summ['units_f1_ge_0.9']} | "
          f"median per-trial count correlation {summ['median_count_corr']:.3f}", flush=True)
    print(LT.round(3).to_string(), flush=True)
    print(f"[val] DONE | {guard.report()}", flush=True)


if __name__ == "__main__":
    main()
