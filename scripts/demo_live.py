"""Phase 1 step 9: end-to-end simulated-live demo (push mode, wall-clock pace).

Stream: jlh32 probe A (ch 0-199) replayed through SimulatedLiveStream in real
time (speed 1.0), handed to one uninterrupted OnlineDecoder in 1024-sample
buffers (how the Open Ephys GUI calls a plugin's process()).  The stimulus
controller is emulated: ~100 ms before each pulse the condition key is passed to
the ArtifactSuppressor.  When the decoder's finalised frontier passes a trial's
window end (pulse offset + 1 ms + 1.9 s) the trial is binned (2-502 ms, 10 ms,
sigma 20 ms), baseline-referenced (1.0-1.9 s of the same chunk), projected on a
fixed PCA basis fitted ONLY on earlier (online-decoded) runs, and compared with
the running past-only mean of its own condition.

Latency = wall time from the moment the window's last sample is 'written' to
the moment the geometry result exists.

Phase B (not paced): pull-mode decode of 0.5 / 1.0 / 1.9 s windows to build
the inter-trial-interval budget.
Outputs: results/demo_live_log.csv, demo_live_buffers.csv, demo_live_trials.npz,
         results/iti_bench.csv, demo_live_summary.json
"""
import os, sys, time, json, dataclasses, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from streamsort.config import StreamConfig
from streamsort.io import OpenEphysSession, FileStreamSource, SimulatedLiveStream
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window, SpikeBatch
from streamsort.preprocess import ArtifactSuppressor
from streamsort.geometry import bin_trial, make_reducer, procrustes_distance, projection_score
from streamsort.memguard import MemGuard, preflight
import geometry_step7 as g7
import ctx

ROOT = ctx.ROOT
RUNS = [int(r) for r in os.environ["DEMO_RUNS"].split(",")] if os.environ.get("DEMO_RUNS") else None
SPEED = float(os.environ.get("DEMO_SPEED", "1.0")); BUF = 1024
B_MS, SM_MS, W0_MS, W_MS, BASE = 10, 20, 2.0, 500.0, (1.0, 1.9)
WIN_START_MS, WIN_S = 1.0, 1.9; MIN_HIST = 5

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "demo_live")
    sess = OpenEphysSession(ctx.REC_DIR)
    st = sess.stream(ctx.STREAM)
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=ctx.NCH)
    fs = cfg.sample_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
    src = FileStreamSource(st, ctx.NCH)
    bank = TemplateBank.load(ctx.R("templates.npz"))
    usable = pd.read_csv(ctx.R("sorting_report.csv")).usable.to_numpy(bool)
    wh = Whitening.load(ctx.R("whitening.npz"))
    tr = pd.read_csv(ctx.R("trials.csv"))
    global RUNS
    if RUNS is None:   # adjacent pair of runs after the sort window with the highest charge
        rr = tr.groupby("run").agg(t=("ttl_onset", "min"), q=("amplitude", lambda a: 0))
        rr["q"] = tr.assign(q=tr.amplitude.abs() * tr.pulse_duration).groupby("run").q.first()
        t_seg = st.sample_to_time(0) + 600.0 + 30.0
        cand = [(min(rr.q[r], rr.q[r + 1]), -r, r) for r in rr.index[:-1] if rr.t[r] > t_seg and (r + 1) in rr.index]
        RUNS = [max(cand)[2], max(cand)[2] + 1]
    print(f"[demo] {ctx.describe()} | runs {RUNS}", flush=True)
    val = dict(np.load(ctx.R("online_spikes_all_trials_sup.npz")))
    units = np.unique(np.load(ctx.R("reference_spikes_all_trials_sup.npz"))["ks4_to_nwb"][:, 0])
    off_all = val["meta_pulse_offset"]
    vo = np.argsort(val["trial"], kind="stable"); vbnd = np.searchsorted(val["trial"][vo], np.arange(len(off_all) + 1))
    def val_spikes(k):
        sl = vo[vbnd[k]:vbnd[k + 1]]; return val["frame"][sl], val["unit"][sl]
    on_all = st.time_to_sample(tr.ttl_onset.to_numpy())
    cond_all = ctx.cond_labels(tr)
    key_all = ctx.cond_keys(tr)

    # ---- causal basis: online-decoded runs before the demo ----------------------
    pre = np.flatnonzero(tr.run.to_numpy() < min(RUNS))
    sub = {k: v for k, v in val.items() if not k.startswith("meta_")}
    Xpre = g7.bin_ref(sub, units, off_all, fs, B_MS, SM_MS, len(off_all))[pre]
    cp = cond_all[pre]
    M = np.concatenate([Xpre[cp == c].mean(0) for c in np.unique(cp)])
    red = make_reducer("pca", 3).fit(M); V = red.basis
    print(f"[demo] basis from runs < {min(RUNS)}: {len(pre)} trials, {len(np.unique(cp))} conditions", flush=True)
    del Xpre

    # ---- live loop -----------------------------------------------------------------
    demo = np.flatnonzero(tr.run.isin(RUNS).to_numpy())
    s_begin = int(on_all[demo[0]] - S(5000)); s_end = int(off_all[demo[-1]] + S(WIN_START_MS) + S(WIN_S * 1e3) + S(50))
    sup = ArtifactSuppressor(fs, ctx.NCH)
    dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False, suppressor=sup)
    # warm-up (JIT, caches) on data before the stream start, then reset state
    decode_window(dec, src, s_begin - S(3000), S(1900)); sup.clear_history(); sup.events.clear()
    dec.start_at(s_begin)
    live = SimulatedLiveStream(src, head=s_begin, speed=SPEED)
    nb = S(W_MS); s0off = S(W0_MS); nbase = (S(BASE[0] * 1e3), S(BASE[1] * 1e3))
    pending = list(demo); cond_set = set()
    spk_f, spk_u = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    hist = {}; logs, blogs, Zs = [], [], {}
    live.start_realtime(); t_wall0 = live._t0_wall
    wall_of = lambda sample: t_wall0 + (sample - s_begin) * st.slope / SPEED   # when a sample is 'written'
    for i0, buf in live.iter_buffers(BUF, start=s_begin, stop=s_end):
        # emulated stim controller: announce the next condition ~100 ms ahead
        for k in pending:
            if k not in cond_set and i0 + BUF >= on_all[k] - S(100) and i0 < on_all[k] - S(5):
                sup.set_condition(key_all[k]); cond_set.add(k)
            if on_all[k] > i0 + S(200):
                break
        t0 = time.perf_counter(); b = dec.push(buf); t1 = time.perf_counter()
        if len(b):
            m = b.unit >= 0; spk_f.append(b.frame[m]); spk_u.append(b.unit[m])
        blogs.append(dict(start=i0, proc_ms=(t1 - t0) * 1e3, behind_ms=(live.head - (i0 + BUF)) / fs * 1e3))
        frontier = dec._next_det
        while pending and frontier >= off_all[pending[0]] + S(WIN_START_MS) + S(WIN_S * 1e3):
            k = pending.pop(0); tg0 = time.perf_counter()
            f = np.concatenate(spk_f); u = np.concatenate(spk_u); spk_f, spk_u = [f], [u]
            off = int(off_all[k]); w_end = off + S(WIN_START_MS) + S(WIN_S * 1e3)
            sel = (f >= off + S(WIN_START_MS)) & (f < w_end)
            fk, uk = f[sel], u[sel]
            mu = np.isin(uk, units)
            x = bin_trial(fk[mu], uk[mu], units, off + s0off, nb, fs, bin_ms=B_MS, smooth_ms=SM_MS) / (B_MS * 1e-3)
            bsel = (fk[mu] >= off + nbase[0]) & (fk[mu] < off + nbase[1])
            base = np.bincount(np.searchsorted(units, uk[mu][bsel]), minlength=len(units)) / (BASE[1] - BASE[0])
            x = x - base[None, :]
            z = red.transform(x)
            c = cond_all[k]; h = hist.setdefault(c, dict(X=[], Z=[]))
            if len(h["X"]) >= MIN_HIST:
                Mp = np.mean(h["X"], 0); gain = float((x * Mp).sum() / (Mp * Mp).sum())
                Up, Sp, Vt = np.linalg.svd(Mp - Mp.mean(0), full_matrices=False)
                ps = projection_score(x, Vt[:3].T); dP = procrustes_distance(np.mean(h["Z"], 0), z)
            else:
                gain = ps = dP = np.nan
            h["X"].append(x); h["Z"].append(z); Zs[k] = z
            tg1 = time.perf_counter()
            # keep the spike buffer bounded
            keep = f >= off + S(WIN_START_MS); spk_f, spk_u = [f[keep]], [u[keep]]
            vf, vu = val_spikes(k)
            A = set(zip(fk.tolist(), uk.tolist())); Bv = set(zip(vf.tolist(), vu.tolist()))
            logs.append(dict(match_vs_pull=len(A & Bv) / max(len(A | Bv), 1), trial=int(k), run=int(tr.run[k]), cond=c, rep=len(h["X"]) - 1, n_spikes=int(sel.sum()),
                             n_spikes_units=int(mu.sum()), latency_ms=(tg1 - wall_of(w_end)) * 1e3,
                             spikes_final_after_end_ms=(tg0 - wall_of(w_end)) * 1e3, geometry_ms=(tg1 - tg0) * 1e3,
                             gain_past=gain, projscore_past=ps, procrustes_past=dP))
            L = logs[-1]
            if len(logs) % 25 == 0 or len(logs) == 1:
                print(f"[demo] trial {k} ({c}) rep {L['rep']}: latency {L['latency_ms']:.1f} ms, geometry {L['geometry_ms']:.2f} ms, "
                      f"gain {L['gain_past']:.2f} | {guard.report()}", flush=True)
        if len(blogs) % 20000 == 0:
            guard.check("stream")
    b = dec.flush()
    pd.DataFrame(logs).to_csv(ctx.R("demo_live_log.csv"), index=False)
    pd.DataFrame(blogs).to_csv(ctx.R("demo_live_buffers.csv"), index=False)
    f = np.concatenate(spk_f + ([b.frame[b.unit >= 0]] if len(b) else []))
    # ---- agreement with the per-trial (pull-mode) validation decode ---------------
    D = pd.DataFrame(logs)
    np.savez_compressed(ctx.R("demo_live_trials.npz"), trials=np.array(sorted(Zs)), Z=np.stack([Zs[k] for k in sorted(Zs)]),
                        basis=V, mean=red.mean_)
    BL = pd.DataFrame(blogs)
    summ = dict(runs=RUNS, speed=SPEED, buffer=BUF, n_trials=len(D), wall_s=float(time.perf_counter() - t_wall0),
                latency_ms_median=float(D.latency_ms.median()), latency_ms_p99=float(D.latency_ms.quantile(.99)),
                latency_ms_max=float(D.latency_ms.max()), match_vs_pull_median=float(D.match_vs_pull.median()),
                match_vs_pull_min=float(D.match_vs_pull.min()), geometry_ms_p99=float(D.geometry_ms.quantile(.99)),
                buffer_proc_ms_median=float(BL.proc_ms.median()), buffer_proc_ms_p99=float(BL.proc_ms.quantile(.99)),
                buffer_budget_ms=BUF / fs * 1e3, behind_ms_max=float(BL.behind_ms.max()),
                rtf=float(BL.proc_ms.sum() / (len(BL) * BUF / fs * 1e3)),
                n_supp_events=len(sup.events), n_supp_subtracted=int(sum(e["subtracted"] for e in sup.events)))
    print("[demo] " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)
    guard.check("live done")

    # ---- Phase B: pull-mode latency vs window length (not paced) -----------------
    rows = []
    sup2 = ArtifactSuppressor(fs, ctx.NCH)
    dec2 = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False, suppressor=sup2)
    decode_window(dec2, src, s_begin - S(3000), S(1900))
    for k in demo[::2]:
        for wl in [0.5, 1.0, 1.9]:
            t0 = time.perf_counter(); bb = decode_window(dec2, src, int(off_all[k]) + S(WIN_START_MS), S(wl * 1e3), condition=key_all[k])
            rows.append(dict(trial=int(k), window_s=wl, decode_ms=(time.perf_counter() - t0) * 1e3, n=len(bb)))
    IB = pd.DataFrame(rows); IB.to_csv(ctx.R("iti_bench.csv"), index=False)
    print(IB.groupby("window_s").decode_ms.describe(percentiles=[.5, .99]).round(1).to_string(), flush=True)
    summ["pull_decode_ms"] = {str(w): dict(median=float(g.decode_ms.median()), p99=float(g.decode_ms.quantile(.99)), max=float(g.decode_ms.max()))
                              for w, g in IB.groupby("window_s")}
    json.dump(summ, open(ctx.R("demo_live_summary.json"), "w"), indent=2)
    print("[demo] DONE |", guard.report(), flush=True)

if __name__ == "__main__":
    main()
