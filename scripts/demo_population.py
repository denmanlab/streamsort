"""Recording-level real-time demo: all chosen probes streamed at once, pooled geometry per trial.

One SimulatedLiveStream per probe (each on its own sample clock, all started at the same wall
time), one ArtifactSuppressor + OnlineDecoder per probe, serviced round-robin in ONE process
(numba kernels are already parallel across cores; one process keeps memory to a single copy of
the libraries).  When every probe's decoder has finalised past a trial's window end, the pooled
population is binned, baseline-referenced and projected on a basis fitted only on earlier runs.
Latency = wall time from when the LAST probe's window end was written to pooled geometry ready.

  REC=jlh32 PROBES=A,B,C [DEMO_RUNS=18,19] python scripts/demo_population.py
"""
import os, sys, time, json, dataclasses, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx, geometry_step7 as g7
from streamsort.config import StreamConfig
from streamsort.io import OpenEphysSession, FileStreamSource, SimulatedLiveStream
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window
from streamsort.preprocess import ArtifactSuppressor
from streamsort.geometry import bin_trial, make_reducer, procrustes_distance
from streamsort.memguard import MemGuard, preflight

BUF, SPEED = 1024, float(os.environ.get("DEMO_SPEED", "1.0"))
B_MS, SM_MS, W0_MS, W_MS, BASE = 10, 20, 2.0, 500.0, (1.0, 1.9); WIN_START_MS, WIN_S = 1.0, 1.9; MIN_HIST = 5
SHORT_WINDOWS = (0.5, 1.0)   # also timed live (pre-pulse 0.5 s baseline); 1.9 s uses same-chunk baseline
PULL_EVERY = int(os.environ.get("PULL_EVERY", "2"))

def main():
    preflight(3.5, margin_gb=0.0, strict=False); guard = MemGuard(3.5, "demo_population")
    rec = ctx.REC; probes = os.environ["PROBES"].split(",") if os.environ.get("PROBES") else ctx.available_probes(rec)
    root = ctx.RECS[rec]["root"]; nch = ctx.RECS[rec]["nch"]; sess = OpenEphysSession(root + "/Record Node 105/experiment1/recording1")
    tr = pd.read_csv(ctx.R_probe(rec, probes[0], "trials.csv")); cond_all = ctx.cond_labels(tr); key_all = ctx.cond_keys(tr)
    if os.environ.get("DEMO_RUNS"):
        RUNS = [int(r) for r in os.environ["DEMO_RUNS"].split(",")]
    else:
        st0 = sess.stream(f"Neuropix-PXI-104.Probe{probes[0]}-AP"); rr = tr.assign(q=tr.amplitude.abs() * tr.pulse_duration).groupby("run").agg(t=("ttl_onset", "min"), q=("q", "first"))
        t_seg = st0.sample_to_time(0) + 630.0
        cand = [(min(rr.q[r], rr.q[r + 1]), -r, r) for r in rr.index[:-1] if rr.t[r] > t_seg and (r + 1) in rr.index]; RUNS = [max(cand)[2], max(cand)[2] + 1]
    demo = np.flatnonzero(tr.run.isin(RUNS).to_numpy()); pre = np.flatnonzero(tr.run.to_numpy() < min(RUNS))
    print(f"[popdemo] {rec} probes {probes} runs {RUNS} ({len(demo)} trials)", flush=True)
    PR = []
    for p in probes:
        R = lambda n, p=p: ctx.R_probe(rec, p, n)
        st = sess.stream(f"Neuropix-PXI-104.Probe{p}-AP")
        cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=nch)
        src = FileStreamSource(st, nch); bank = TemplateBank.load(R("templates.npz"))
        usable = pd.read_csv(R("sorting_report.csv")).usable.to_numpy(bool); wh = Whitening.load(R("whitening.npz"))
        val = dict(np.load(R("online_spikes_all_trials_sup.npz"))); units = np.unique(np.load(R("reference_spikes_all_trials_sup.npz"))["ks4_to_nwb"][:, 0])
        sup = ArtifactSuppressor(cfg.sample_rate, nch)
        dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False, suppressor=sup)
        PR.append(dict(p=p, st=st, cfg=cfg, src=src, dec=dec, sup=sup, val=val, units=units, fs=cfg.sample_rate,
                       off=val["meta_pulse_offset"], on=st.time_to_sample(tr.ttl_onset.to_numpy())))
        guard.check(f"probe {p} loaded")
    # causal pooled basis: online-decoded runs before the demo
    Xpre = []
    for P in PR:
        sub = {k: v for k, v in P["val"].items() if not k.startswith("meta_")}
        Xpre.append(g7.bin_ref(sub, P["units"], P["off"], P["fs"], B_MS, SM_MS, len(P["off"]))[pre])
    Xpre = ctx.cat_units(Xpre, axis=2); cp = cond_all[pre]
    red = make_reducer("pca", 3).fit(np.concatenate([Xpre[cp == c].mean(0) for c in np.unique(cp)])); del Xpre
    # common wall-clock start: every stream begins 5 s before the first demo pulse (global time)
    T_begin = tr.ttl_onset[demo[0]] - 5.0; T_end = tr.ttl_onset[demo[-1]] + 2.0
    for P in PR:
        S = lambda ms, P=P: int(round(ms * 1e-3 * P["fs"]))
        P["S"] = S; P["s_begin"] = int(P["st"].time_to_sample(np.array([T_begin]))[0]); P["s_end"] = int(P["st"].time_to_sample(np.array([T_end]))[0])
        decode_window(P["dec"], P["src"], P["s_begin"] - S(3000), S(1900)); P["sup"].clear_history(); P["sup"].events.clear()
        P["dec"].start_at(P["s_begin"]); P["live"] = SimulatedLiveStream(P["src"], head=P["s_begin"], speed=SPEED)
        P["i"] = P["s_begin"]; P["announced"] = set(); P["spk_f"] = []; P["spk_u"] = []; P["blog"] = []; P["proc"] = 0.0
    for P in PR: P["live"].start_realtime()
    t_wall0 = min(P["live"]._t0_wall for P in PR)
    wall_of = lambda P, sample: P["live"]._t0_wall + (sample - P["s_begin"]) * P["st"].slope / SPEED
    pending = list(demo); hist = {}; logs = []; nb = {P["p"]: P["S"](W_MS) for P in PR}
    pend_w = {w: list(demo) for w in SHORT_WINDOWS}; wlogs = []
    while pending:
        progressed = False
        for P in PR:
            if P["i"] + BUF > P["s_end"] or P["live"].head < P["i"] + BUF:
                continue
            i0 = P["i"]; buf = P["src"].read(i0, BUF)
            for k in pending[:3]:
                if k not in P["announced"] and i0 + BUF >= P["on"][k] - P["S"](100) and i0 < P["on"][k] - P["S"](5):
                    P["sup"].set_condition(key_all[k]); P["announced"].add(k)
            t0 = time.perf_counter(); b = P["dec"].push(buf); dt = time.perf_counter() - t0; P["proc"] += dt
            if len(b):
                m = b.unit >= 0; P["spk_f"].append(b.frame[m]); P["spk_u"].append(b.unit[m])
            P["blog"].append(dict(probe=P["p"], start=i0, proc_ms=dt * 1e3, behind_ms=(P["live"].head - (i0 + BUF)) / P["fs"] * 1e3))
            P["i"] += BUF; progressed = True
        # short analysis windows (0.5 / 1.0 s): pooled geometry as soon as every probe has decoded past the window end
        for w in SHORT_WINDOWS:
            while pend_w[w]:
                k = pend_w[w][0]; we = {P["p"]: int(P["off"][k]) + P["S"](WIN_START_MS) + P["S"](w * 1e3) for P in PR}
                if not all(P["dec"]._next_det >= we[P["p"]] for P in PR):
                    break
                pend_w[w].pop(0); tg0 = time.perf_counter(); xs = []
                for P in PR:
                    f = np.concatenate(P["spk_f"]) if P["spk_f"] else np.zeros(0, np.int64); u = np.concatenate(P["spk_u"]) if P["spk_u"] else np.zeros(0, np.int64)
                    off = int(P["off"][k]); on = int(P["on"][k]); mu = np.isin(u, P["units"])
                    sel = mu & (f >= off + P["S"](W0_MS)) & (f < we[P["p"]])
                    x = bin_trial(f[sel], u[sel], P["units"], off + P["S"](W0_MS), min(nb[P["p"]], we[P["p"]] - off - P["S"](W0_MS)), P["fs"], bin_ms=B_MS, smooth_ms=SM_MS) / (B_MS * 1e-3)
                    bs = mu & (f >= on - P["S"](500)) & (f < on)
                    xs.append(x - (np.bincount(np.searchsorted(P["units"], u[bs]), minlength=len(P["units"])) / 0.5)[None, :])
                z = red.transform(ctx.cat_units(xs, axis=1)); tg1 = time.perf_counter()
                wlogs.append(dict(trial=int(k), window_s=w, latency_ms=(tg1 - max(wall_of(P, we[P["p"]]) for P in PR)) * 1e3, geometry_ms=(tg1 - tg0) * 1e3,
                                  **{f"lag_{P['p']}_ms": (tg1 - wall_of(P, we[P['p']])) * 1e3 for P in PR}))
        # pooled trial completion
        while pending:
            k = pending[0]
            w_end = {P["p"]: int(P["off"][k]) + P["S"](WIN_START_MS) + P["S"](WIN_S * 1e3) for P in PR}
            if not all(P["dec"]._next_det >= w_end[P["p"]] for P in PR):
                break
            pending.pop(0); tg0 = time.perf_counter(); xs = []; jac = []
            for P in PR:
                f = np.concatenate(P["spk_f"]) if P["spk_f"] else np.zeros(0, np.int64); u = np.concatenate(P["spk_u"]) if P["spk_u"] else np.zeros(0, np.int64)
                off = int(P["off"][k]); sel = (f >= off + P["S"](WIN_START_MS)) & (f < w_end[P["p"]]); fk, uk = f[sel], u[sel]
                mu = np.isin(uk, P["units"])
                x = bin_trial(fk[mu], uk[mu], P["units"], off + P["S"](W0_MS), nb[P["p"]], P["fs"], bin_ms=B_MS, smooth_ms=SM_MS) / (B_MS * 1e-3)
                bs = (fk[mu] >= off + P["S"](BASE[0] * 1e3)) & (fk[mu] < off + P["S"](BASE[1] * 1e3))
                x = x - (np.bincount(np.searchsorted(P["units"], uk[mu][bs]), minlength=len(P["units"])) / (BASE[1] - BASE[0]))[None, :]
                xs.append(x); jac.append((fk, uk))   # validation vs pull decode is done after the timed span
                keep = f >= off + P["S"](WIN_START_MS); P["spk_f"], P["spk_u"] = [f[keep]], [u[keep]]
            x = ctx.cat_units(xs, axis=1); z = red.transform(x); c = cond_all[k]; h = hist.setdefault(c, dict(X=[], Z=[]))
            if len(h["X"]) >= MIN_HIST:
                Mp = np.mean(h["X"], 0); gain = float((x * Mp).sum() / (Mp * Mp).sum()); dP = procrustes_distance(np.mean(h["Z"], 0), z)
            else:
                gain = dP = np.nan
            h["X"].append(x); h["Z"].append(z); tg1 = time.perf_counter()
            for i_, P in enumerate(PR):   # validation only (not part of the live pipeline, not timed)
                fk, uk = jac[i_]; v = P["val"]; vm = v["trial"] == k; A_ = set(zip(fk.tolist(), uk.tolist())); B_ = set(zip(v["frame"][vm].tolist(), v["unit"][vm].tolist()))
                jac[i_] = len(A_ & B_) / max(len(A_ | B_), 1)
            last_written = max(wall_of(P, w_end[P["p"]]) for P in PR)
            logs.append(dict(trial=int(k), run=int(tr.run[k]), cond=c, rep=len(h["X"]) - 1, latency_ms=(tg1 - last_written) * 1e3,
                             geometry_ms=(tg1 - tg0) * 1e3, gain_past=gain, procrustes_past=dP, min_jaccard_vs_pull=min(jac),
                             **{f"lag_{P['p']}_ms": (time.perf_counter() - wall_of(P, w_end[P['p']])) * 1e3 for P in PR}))
            if len(logs) % 25 == 0 or len(logs) == 1:
                print(f"[popdemo] trial {k} ({c}): latency {logs[-1]['latency_ms']:.1f} ms, geometry {logs[-1]['geometry_ms']:.2f} ms, "
                      f"min Jaccard {min(jac):.3f} | {guard.report()}", flush=True)
        if not progressed:
            if all(P["i"] + BUF > P["s_end"] for P in PR): break
            time.sleep(0.0005)
        if len(logs) and len(logs) % 50 == 0: guard.check("stream")
    wall = time.perf_counter() - t_wall0
    L = pd.DataFrame(logs)
    for P in PR:   # position of each buffer in the trial cycle: ms from the most recent pulse onset to the buffer's last sample
        b = pd.DataFrame(P["blog"]); ons = np.sort(P["on"]); e = b.start.to_numpy() + BUF; j = np.searchsorted(ons, e, side="right") - 1
        P["blog"] = b.assign(since_pulse_ms=np.where(j >= 0, (e - ons[np.clip(j, 0, None)]) / P["fs"] * 1e3, np.nan))
    BL = pd.concat([P["blog"] for P in PR])
    PU = pd.concat([pd.DataFrame(wlogs), L[["trial", "latency_ms", "geometry_ms"] + [c for c in L.columns if c.startswith("lag_")]].assign(window_s=WIN_S)], ignore_index=True)
    PU.to_csv(ctx.RP("iti_population_push.csv"), index=False)
    # ---- pull mode (not paced): after each window ends, decode every probe's window from disk in turn, then pooled geometry
    guard.check("before pull bench")
    for P in PR:
        P["sup"] = ArtifactSuppressor(P["cfg"].sample_rate, nch)
        P["dec"] = OnlineDecoder(P["dec"].bank if hasattr(P["dec"], "bank") else TemplateBank.load(ctx.R_probe(rec, P["p"], "templates.npz")), P["cfg"],
                                 Whitening.load(ctx.R_probe(rec, P["p"], "whitening.npz")), DecoderParams(min_energy=50.0),
                                 units_mask=pd.read_csv(ctx.R_probe(rec, P["p"], "sorting_report.csv")).usable.to_numpy(bool), check=False, suppressor=P["sup"])
        decode_window(P["dec"], P["src"], P["s_begin"] - P["S"](3000), P["S"](1900))
    rows = []
    for k in demo[::PULL_EVERY]:
        for w in SHORT_WINDOWS + (WIN_S,):
            r = dict(trial=int(k), window_s=w); t0 = time.perf_counter(); xs = []
            for P in PR:
                tp = time.perf_counter(); off = int(P["off"][k])
                bb = decode_window(P["dec"], P["src"], off + P["S"](WIN_START_MS), P["S"](w * 1e3), condition=key_all[k]); r[f"decode_{P['p']}_ms"] = (time.perf_counter() - tp) * 1e3
                f, u = np.asarray(bb.frame), np.asarray(bb.unit); mu = np.isin(u, P["units"]) & (f >= off + P["S"](W0_MS))
                xs.append(bin_trial(f[mu], u[mu], P["units"], off + P["S"](W0_MS), min(nb[P["p"]], P["S"](w * 1e3) - P["S"](W0_MS - WIN_START_MS)), P["fs"], bin_ms=B_MS, smooth_ms=SM_MS) / (B_MS * 1e-3))
            tg = time.perf_counter(); red.transform(ctx.cat_units(xs, axis=1)); t1 = time.perf_counter()
            r.update(geometry_ms=(t1 - tg) * 1e3, total_ms=(t1 - t0) * 1e3); rows.append(r)
        if len(rows) % 30 == 0: guard.check("pull bench")
    PL = pd.DataFrame(rows); PL.to_csv(ctx.RP("iti_population_pull.csv"), index=False)
    print(PU.groupby("window_s").latency_ms.describe(percentiles=[.5, .99]).round(1).to_string(), flush=True)
    print(PL.groupby("window_s").total_ms.describe(percentiles=[.5, .99]).round(1).to_string(), flush=True)
    L.to_csv(ctx.RP("demo_population_log.csv"), index=False); BL.to_csv(ctx.RP("demo_population_buffers.csv"), index=False)
    summ = dict(rec=rec, probes=probes, runs=RUNS, n_trials=len(L), wall_s=wall, speed=SPEED, buffer=BUF,
                n_units_pooled=int(sum(len(P["units"]) for P in PR)),
                latency_ms_median=float(L.latency_ms.median()), latency_ms_p99=float(L.latency_ms.quantile(.99)), latency_ms_max=float(L.latency_ms.max()),
                geometry_ms_p99=float(L.geometry_ms.quantile(.99)), min_jaccard_vs_pull=float(L.min_jaccard_vs_pull.min()),
                rtf_total=float(sum(P["proc"] for P in PR) / wall), rtf_per_probe={P["p"]: float(P["proc"] / wall) for P in PR},
                behind_ms_max={P["p"]: float(pd.DataFrame(P["blog"]).behind_ms.max()) for P in PR},
                buffer_proc_ms_p99={P["p"]: float(pd.DataFrame(P["blog"]).proc_ms.quantile(.99)) for P in PR},
                push_latency_ms={str(w): dict(median=float(g.latency_ms.median()), p99=float(g.latency_ms.quantile(.99)), max=float(g.latency_ms.max()), n=int(len(g))) for w, g in PU.groupby("window_s")},
                pull_total_ms={str(w): dict(median=float(g.total_ms.median()), p99=float(g.total_ms.quantile(.99)), max=float(g.total_ms.max()), n=int(len(g))) for w, g in PL.groupby("window_s")})
    json.dump(summ, open(ctx.RP("demo_population_summary.json"), "w"), indent=2)
    print("[popdemo] DONE " + json.dumps(summ), flush=True); guard.check("done")

if __name__ == "__main__":
    main()
