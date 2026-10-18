"""Phase 2 replay harness: plugin code path vs Phase 1 pull-mode decoder on jlh32.

  1. equality   probe A, 60 s, GUI buffers of 1024 and 512 samples: spikes identical to the
                OnlineDecoder fed the same int16 data in large chunks (Phase 1 construction)
  2. hot-swap   templates.npz atomically replaced mid-run: one swap, spikes after it match
  3. end-to-end probes A,B,C + worker process, real-time pace: every pulse paired, per-trial
                pulse->loss and pulse->stim times, condition sequence, per-buffer real-time factor
Writes results/phase2/harness.json, results/phase2/configs/*.json and the two plugin figures.
"""
import dataclasses, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctx  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import OpenEphysSession, FileStreamSource  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, Whitening  # noqa: E402
from streamsort.online.decoder import SpikeBatch  # noqa: E402
from streamsort.preprocess.suppressor import ArtifactSuppressor  # noqa: E402
from streamsort.sorting.templates import TemplateBank  # noqa: E402
from streamsort.oe import replay  # noqa: E402

REC = "jlh32"; OUT = Path("results/phase2"); (OUT / "configs").mkdir(parents=True, exist_ok=True)
QUICK = bool(os.environ.get("HARNESS_QUICK"))
EQ_S, E2E_S = (20.0, 30.0) if QUICK else (60.0, 120.0)
root = ctx.RECS[REC]["root"]; NCH = ctx.RECS[REC]["nch"]; RD = root + "/Record Node 105/experiment1/recording1"
sess = OpenEphysSession(RD)
tr = pd.read_csv(ctx.R_probe(REC, "A", "trials.csv")); keys = ctx.cond_keys(tr)
keys = [tuple(int(v) if not isinstance(v, str) else v for v in k) for k in keys]
ttl = tr.ttl_onset.to_numpy()
t_seg = sess.stream("Neuropix-PXI-104.ProbeA-AP").sample_to_time(0) + 630.0
k0 = int(np.searchsorted(ttl, t_seg)); T0 = float(ttl[k0] - 1.0)


def probe_cfg(p, templates=None):
    R = lambda n: str(Path(ctx.R_probe(REC, p, n)).resolve())
    c = dict(probe=p, stream=f"Neuropix-PXI-104.Probe{p}-AP", templates=templates or R("templates.npz"),
             whitening=R("whitening.npz"), sorting_report=str((ctx.probe_out(REC, p) / "sorting_report.csv").resolve()),
             nch=NCH, bit_volts=0.195, input="uV", detect_threshold=4.0, min_energy=50.0, ttl_line=5,
             worker_address=["127.0.0.1", 6010], repo=str(Path.cwd().resolve()))
    (OUT / "configs" / f"streamsort_oe_{p}.json").write_text(json.dumps(c, indent=1))
    return c


def cond_points(st, t0, dur):
    """(sample relative to t0, key): condition of trial k is set 300 ms before its TTL."""
    s0 = int(st.time_to_sample(np.array([t0]))[0])
    on = st.time_to_sample(ttl) - s0; n = int(dur * st.effective_rate); lead = int(0.3 * st.effective_rate)
    return [(int(on[k] - lead), keys[k]) for k in range(len(ttl)) if lead <= on[k] < n]


def reference(p, t0, dur, pts):
    """Phase 1 construction, int16 data pushed in large chunks split only at condition points."""
    st = sess.stream(f"Neuropix-PXI-104.Probe{p}-AP")
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=NCH)
    src = FileStreamSource(st, NCH); bank = TemplateBank.load(ctx.R_probe(REC, p, "templates.npz"))
    usable = pd.read_csv(ctx.probe_out(REC, p) / "sorting_report.csv").usable.to_numpy(bool)
    dec = OnlineDecoder(bank, cfg, Whitening.load(ctx.R_probe(REC, p, "whitening.npz")), DecoderParams(min_energy=50.0),
                        units_mask=usable, check=False, suppressor=ArtifactSuppressor(cfg.sample_rate, NCH))
    s0 = int(st.time_to_sample(np.array([t0]))[0]); n = int(dur * st.effective_rate)
    dec.start_at(0); out = []; edges = [0] + [s for s, _ in pts] + [n]; kk = [None] + [k for _, k in pts]
    for a, b, key in zip(edges[:-1], edges[1:], kk):
        if key is not None:
            dec.suppressor.set_condition(key)
        for c in range(a, b, 30000):
            out.append(dec.push(np.asarray(src.read(s0 + c, min(30000, b - c)))))
    out.append(dec.flush()); bb = SpikeBatch.concat(out); m = bb.unit >= 0
    return bb.frame[m], bb.unit[m]


def scheduler(pts):
    i = {"i": 0}
    def f(core, a, b):
        while i["i"] < len(pts) and pts[i["i"]][0] < b:
            core.set_condition(pts[i["i"]][1]); i["i"] += 1
    return f


def compare(a, b, after=0):
    A = {(int(f), int(u)) for f, u in zip(*a) if f >= after}; B = {(int(f), int(u)) for f, u in zip(*b) if f >= after}
    return dict(n_plugin=len(A), n_ref=len(B), identical=A == B, jaccard=len(A & B) / max(len(A | B), 1))


def main():
    R = dict(t0=T0, first_trial=k0, transport=os.environ.get("HARNESS_TRANSPORT", "pipe"))
    stA = sess.stream("Neuropix-PXI-104.ProbeA-AP")
    E2E_ONLY = bool(os.environ.get("HARNESS_E2E_ONLY"))
    if E2E_ONLY and (OUT / "harness.json").exists():
        R = json.loads((OUT / "harness.json").read_text()) | R
    # 1. equality --------------------------------------------------------------------------
    pts = cond_points(stA, T0, EQ_S); ref = reference("A", T0, EQ_S, pts) if not E2E_ONLY else None
    for buf in (() if E2E_ONLY else (1024, 512)):
        r = replay.run(RD, {"A": probe_cfg("A")}, T0, EQ_S, buffer=buf, ttl_times=ttl, pace=0, schedule_fn=scheduler(pts))
        R[f"equality_buf{buf}"] = compare(r["spikes"]["A"], ref) | dict(n_pulses=len(r["cores"]["A"].pulses_out),
                                                                        n_ttl_in_window=len(r["feeders"]["A"].ttl))
        print("[harness] equality", buf, R[f"equality_buf{buf}"], flush=True)
    # 2. hot-swap --------------------------------------------------------------------------
    sw = OUT / "swap"; sw.mkdir(exist_ok=True); shutil.copy(ctx.R_probe(REC, "A", "templates.npz"), sw / "templates.npz")
    cA = probe_cfg("A", templates=str((sw / "templates.npz").resolve())); probe_cfg("A")
    SWAP_S = EQ_S / 2
    r = None if E2E_ONLY else replay.run(RD, {"A": cA}, T0, EQ_S, buffer=1024, ttl_times=ttl, pace=0, schedule_fn=scheduler(pts), swap_at_s=SWAP_S,
                   swap_fn=lambda: TemplateBank.load(sw / "templates.npz").save(sw / "templates.npz"))
    fs = stA.effective_rate; settle = int((SWAP_S + 0.6 + 0.5 + 0.05) * fs)   # swap waits <= swap_quiet + check period
    if not E2E_ONLY:
      R["hot_swap"] = dict(n_swaps=r["cores"]["A"].n_swaps, after_settle=compare(r["spikes"]["A"], ref, after=settle), all=compare(r["spikes"]["A"], ref))
    print("[harness] hot-swap", R.get("hot_swap"), flush=True)
    # 3. end-to-end ------------------------------------------------------------------------
    cfgs = {p: probe_cfg(p) for p in "ABC"}
    sched = [keys[k] for k in range(k0, len(keys))]
    wcfg = dict(probes=list("ABC"), address=["127.0.0.1", 6010], worker_address=["127.0.0.1", 6010], iti_s=2.0,
                analysis="procrustes", analysis_kw=dict(reducer="pca", k=3, n_fit=10), reprog_ms=100.0,
                schedule=sched, log=str((OUT / "worker_log.json").resolve()))
    (OUT / "configs" / "worker.json").write_text(json.dumps({k: v for k, v in wcfg.items() if k != "schedule"}, indent=1))
    t = time.perf_counter()
    r = replay.run(RD, cfgs, T0, E2E_S, buffer=1024, ttl_times=ttl, worker_cfg=wcfg, pace=1.0,
                   transport=os.environ.get("HARNESS_TRANSPORT", "pipe"))
    wall = time.perf_counter() - t
    log = r["worker"]; T = pd.DataFrame(log["trials"]); S = pd.DataFrame(log["stim"])
    n_ttl = len(r["feeders"]["A"].ttl)
    pts3 = {p: cond_points(sess.stream(f"Neuropix-PXI-104.Probe{p}-AP"), T0, E2E_S) for p in "ABC"}
    eq3 = {p: compare(r["spikes"][p], reference(p, T0, E2E_S, pts3[p])) for p in "ABC"}
    fsr = {p: np.asarray(v) / (1024 / fs) for p, v in r["buffer_s"].items()}
    tot = np.sum([fsr[p][: min(len(v) for v in fsr.values())] for p in fsr], axis=0)
    R["end_to_end"] = dict(seconds=E2E_S, wall_s=wall, startup_s=r["startup_s"], stream_wall_s=r["stream_s"], n_ttl=n_ttl, n_trials_analysed=len(T),
                           all_three_probes_paired=bool((T.n_probes_pulse == 3).all()) if len(T) else False,
                           ttl_index_sequential=bool((np.diff(T.k.astype(int)) == 1).all()) if len(T) > 1 else False,
                           cond_sequence_matches=bool([str(c) for c in T.cond] == [str(keys[k0 + int(k)]) for k in T.k]),
                           n_programmed=int((S.stim == "programmed").sum()), n_hold=int((S.stim == "hold").sum()),
                           pulse_to_loss_ms=dict(median=float(T.pulse_to_loss_ms.median()), max=float(T.pulse_to_loss_ms.max())),
                           pulse_to_stim_ms=dict(median=float(S.pulse_to_stim_ms.median()), max=float(S.pulse_to_stim_ms.max())),
                           geometry_ms_median=float(T.geometry_ms.median()), n_loss=int(T.loss.notna().sum()),
                           rt_factor_per_probe={p: dict(median=float(np.median(v)), p99=float(np.percentile(v, 99))) for p, v in fsr.items()},
                           rt_factor_3probe_serial=dict(median=float(np.median(tot)), p99=float(np.percentile(tot, 99)), max=float(tot.max())),
                           equality_vs_reference=eq3)
    print("[harness] end-to-end", json.dumps(R["end_to_end"], default=str), flush=True)
    (OUT / "harness.json").write_text(json.dumps(R, indent=1, default=str))
    np.savez(OUT / "rt_factor.npz", **{p: v for p, v in fsr.items()}, serial=tot)


if __name__ == "__main__":
    main()
