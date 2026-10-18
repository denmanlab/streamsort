"""Phase 2b validation: C++ decoder core vs the Python decoder on jlh32 (replayed as a stream).

  1. probe A, 60 s, buffers 1024 and 512, 1 thread: spikes and pulse onsets vs Python
  2. exported table folder (plugin format) -> C++ decoder: identical to in-memory tables
  3. probes A+B+C, 120 s, one Python thread per stream (GIL released in C++), 1 and 2 core
     threads per stream: per-probe agreement + per-block real-time factor (parallel = slowest stream)
Writes results/phase2b/phase2b_core_validation.json and phase2b_core_rt.npz.
The Phase 2 harness is imported read-only for its reference decoder and condition schedule.
"""
import dataclasses, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_harness as H, ctx  # noqa: E402
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import FileStreamSource  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, Whitening  # noqa: E402
from streamsort.online.decoder import SpikeBatch  # noqa: E402
from streamsort.preprocess.suppressor import ArtifactSuppressor  # noqa: E402
from streamsort.sorting.templates import TemplateBank  # noqa: E402
from streamsort.core_cpp import tables_from_decoder, export_tables, load_exported, CoreDecoder  # noqa: E402

OUT = Path("results/phase2b"); OUT.mkdir(parents=True, exist_ok=True)
QUICK = bool(os.environ.get("HARNESS_QUICK"))
EQ_S, E2E_S = (15.0, 20.0) if QUICK else (60.0, 120.0)


def py_decoder(p):
    st = H.sess.stream(f"Neuropix-PXI-104.Probe{p}-AP")
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=H.NCH)
    bank = TemplateBank.load(ctx.R_probe("jlh32", p, "templates.npz"))
    usable = pd.read_csv(ctx.probe_out("jlh32", p) / "sorting_report.csv").usable.to_numpy(bool)
    return st, OnlineDecoder(bank, cfg, Whitening.load(ctx.R_probe("jlh32", p, "whitening.npz")), DecoderParams(min_energy=50.0),
                             units_mask=usable, check=False, suppressor=ArtifactSuppressor(cfg.sample_rate, H.NCH))


def raw_block(st, t0, dur):
    src = FileStreamSource(st, H.NCH); s0 = int(st.time_to_sample(np.array([t0]))[0])
    return np.asarray(src.read(s0, int(dur * st.effective_rate))).astype(np.float32)


def run_core(cd, raw, pts, buf, times=None):
    n = len(raw); edges = [0] + [s for s, _ in pts] + [n]; kk = [None] + [k for _, k in pts]; out = []
    cd.start_at(0)
    for a, b, key in zip(edges[:-1], edges[1:], kk):
        if key is not None:
            cd.set_condition(key)
        for c in range(a, b, buf):
            t = time.perf_counter(); out.append(cd.push(raw[c:min(c + buf, b)]))
            if times is not None:
                times.append(time.perf_counter() - t)
    out.append(cd.flush()); bb = SpikeBatch.concat(out); m = bb.unit >= 0
    return (bb.frame[m], bb.unit[m])


def per_unit(a, b):
    ua, ca = np.unique(a[1], return_counts=True); ub, cb = np.unique(b[1], return_counts=True)
    u = np.union1d(ua, ub); da = dict(zip(ua, ca)); db = dict(zip(ub, cb))
    return np.array([da.get(x, 0) for x in u]), np.array([db.get(x, 0) for x in u])


def main():
    R = dict(t0=H.T0, first_trial=H.k0, quick=QUICK)
    ONLY3 = os.environ.get("HARNESS_PART") == "3"
    if ONLY3:                                     # timing-only rerun: keep parts 1-2 from the last full run
        R = json.loads((OUT / "phase2b_core_validation.json").read_text())
    else:
        part12(R)
    part3(R)


def part12(R):
    # 1 ------------------------------------------------------------------------------
    stA, decA = py_decoder("A"); tabA = tables_from_decoder(decA)
    pts = H.cond_points(stA, H.T0, EQ_S); ref = H.reference("A", H.T0, EQ_S, pts); rawA = raw_block(stA, H.T0, EQ_S)
    for buf in (1024, 512):
        cd = CoreDecoder(tabA, 1); got = run_core(cd, rawA, pts, buf)
        on_py = None
        R[f"probeA_buf{buf}"] = H.compare(got, ref) | dict(n_events=len(cd.events))
        print("[2b] probe A buf", buf, R[f"probeA_buf{buf}"], flush=True)
    # pulse onsets: Python decoder on the same schedule
    _, dref = py_decoder("A"); dref.start_at(0)
    n = len(rawA); edges = [0] + [s for s, _ in pts] + [n]; kk = [None] + [k for _, k in pts]
    for a, b, key in zip(edges[:-1], edges[1:], kk):
        if key is not None:
            dref.suppressor.set_condition(key)
        for c in range(a, b, 30000):
            dref.push(rawA[c:min(c + 30000, b)].astype(np.int16))
    dref.flush()
    on_py = [(e["onset"], e["span"], e["subtracted"]) for e in dref.suppressor.events]
    on_cpp = [(e["onset"], e["span"], e["subtracted"]) for e in cd.events]
    R["probeA_pulse_onsets_identical"] = on_py == on_cpp; R["probeA_n_pulses"] = len(on_cpp)
    print("[2b] onsets identical", on_py == on_cpp, len(on_cpp), flush=True)
    # 2 ------------------------------------------------------------------------------
    folder = OUT / "tables_jlh32_probeA"; export_tables(tabA, folder, extra=dict(sample_rate=stA.effective_rate))
    got2 = run_core(CoreDecoder(load_exported(folder), 1), rawA, pts, 1024)
    R["exported_tables_identical"] = H.compare(got2, ref)["identical"]
    print("[2b] exported tables identical", R["exported_tables_identical"], flush=True)


def part3(R):
    # 3 ------------------------------------------------------------------------------
    probes = "ABC"; S = {}; raws = {}; pts3 = {}; tabs = {}
    for p in probes:
        st, d = py_decoder(p); S[p] = st; tabs[p] = tables_from_decoder(d)
        pts3[p] = H.cond_points(st, H.T0, E2E_S)
        raws[p] = (FileStreamSource(st, H.NCH), int(st.time_to_sample(np.array([H.T0]))[0]), int(E2E_S * st.effective_rate))
    refs = {p: H.reference(p, H.T0, E2E_S, pts3[p]) for p in probes}
    fs = S["A"].effective_rate; BUF = 1024
    R["three_probe"] = {}; rt = {}
    for nt in (1, 2):
        cds = {p: CoreDecoder(tabs[p], nt) for p in probes}
        for p in probes:
            cds[p].start_at(0)
        nmin = min(raws[p][2] for p in probes); nblk = nmin // BUF + 1
        out = {p: [] for p in probes}; pi = {p: 0 for p in probes}; tpar = []; tser = {p: [] for p in probes}

        def step(p, b0):
            # conditions in schedule order; one GUI block per stream
            while pi[p] < len(pts3[p]) and pts3[p][pi[p]][0] < b0 + BUF:
                cds[p].set_condition(pts3[p][pi[p]][1]); pi[p] += 1
            src, s0, nn = raws[p]
            x = np.asarray(src.read(s0 + b0, min(BUF, nn - b0))).astype(np.float32)     # read outside the timed call
            t = time.perf_counter(); r = cds[p].push(x); dt = time.perf_counter() - t
            out[p].append(r); tser[p].append(dt)

        with ThreadPoolExecutor(len(probes)) as ex:
            for k in range(nblk):
                b0 = k * BUF
                if b0 >= nmin:
                    break
                list(ex.map(lambda p: step(p, b0), probes))
                tpar.append(max(tser[p][-1] for p in probes))   # parallel streams: block done when the slowest is
        agree = {}
        for p in probes:
            out[p].append(cds[p].flush()); bb = SpikeBatch.concat(out[p]); m = bb.unit >= 0
            agree[p] = H.compare((bb.frame[m], bb.unit[m]), refs[p])
            if nt == 1:
                R.setdefault("per_unit", {})[p] = [x.tolist() for x in per_unit((bb.frame[m], bb.unit[m]), refs[p])]
        bd = BUF / fs
        par = np.array(tpar) / bd; ser = np.sum([np.array(tser[p][:len(tpar)]) for p in probes], axis=0) / bd
        rt[f"par{nt}"] = par; rt[f"ser{nt}"] = ser
        for p in probes:
            rt[f"{p}{nt}"] = np.array(tser[p]) / bd
        R["three_probe"][f"threads_per_stream_{nt}"] = dict(
            agreement=agree, all_identical=all(v["identical"] for v in agree.values()),
            rt_parallel=dict(median=float(np.median(par)), p99=float(np.percentile(par, 99)), max=float(par.max())),
            rt_serial_sum=dict(median=float(np.median(ser)), p99=float(np.percentile(ser, 99)), max=float(ser.max())),
            rt_per_stream_median={p: float(np.median(rt[f"{p}{nt}"])) for p in probes})
        print("[2b] 3 probes, threads/stream", nt, json.dumps(R["three_probe"][f"threads_per_stream_{nt}"], default=float)[:900], flush=True)
    (OUT / "phase2b_core_validation.json").write_text(json.dumps(R, indent=1, default=float))
    np.savez(OUT / "phase2b_core_rt.npz", **rt)


if __name__ == "__main__":
    main()
