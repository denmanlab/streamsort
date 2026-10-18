"""Phase 2b end-to-end: `streamsort emulate` (C++ core, plugin message protocol) -> worker v2 over ZMQ
(inproc inside one process: this sandbox forbids binding sockets; tcp works the same on the rig).

  python scripts/phase2b_e2e.py headless [seconds]     checks + results/phase2b/phase2b_e2e.json
  python scripts/phase2b_e2e.py gui procrustes|image_decoder [seconds]   offscreen GUI + screenshot
"""
import dataclasses, json, os, sys, threading, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_harness as H, ctx  # noqa: E402  (read-only reuse: recording, trials, Python reference decoder)
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, Whitening  # noqa: E402
from streamsort.preprocess.suppressor import ArtifactSuppressor  # noqa: E402
from streamsort.sorting.templates import TemplateBank  # noqa: E402
from streamsort.core_cpp import tables_from_decoder, export_tables  # noqa: E402
from streamsort.worker2.emulator import Emulator  # noqa: E402
from streamsort.worker2.analyses import make_placeholder_images  # noqa: E402

OUT = Path("results/phase2b"); FIG = Path("figures/phase2b"); FIG.mkdir(parents=True, exist_ok=True)
PROBES = "ABC"; PUB, COND = "inproc://streamsort-pub", "inproc://streamsort-cond"


def bank_folders():
    f = {}
    for p in PROBES:
        d = OUT / f"tables_jlh32_probe{p}"
        if not (d / "decoder_tables.txt").exists():
            st = H.sess.stream(f"Neuropix-PXI-104.Probe{p}-AP")
            cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=H.NCH)
            bank = TemplateBank.load(ctx.R_probe("jlh32", p, "templates.npz"))
            usable = pd.read_csv(ctx.probe_out("jlh32", p) / "sorting_report.csv").usable.to_numpy(bool)
            dec = OnlineDecoder(bank, cfg, Whitening.load(ctx.R_probe("jlh32", p, "whitening.npz")), DecoderParams(min_energy=50.0),
                                units_mask=usable, check=False, suppressor=ArtifactSuppressor(cfg.sample_rate, H.NCH))
            export_tables(tables_from_decoder(dec), d, extra=dict(sample_rate=st.effective_rate, n_units=bank.n_units, n_usable=int(usable.sum())))
        f[f"Neuropix-PXI-104.Probe{p}-AP"] = str(d)
    return f


def conditions():
    keys = H.keys                                       # per-trial condition keys (amplitude, polarity, duration, c-, c+)
    uniq = sorted(set(keys), key=lambda k: (k[0], str(k[1]), k[2], k[3], k[4]))
    cid = {k: i for i, k in enumerate(uniq)}
    conds = [dict(amplitude=int(k[0]), polarity=str(k[1]), pulse_duration=int(k[2]), contact_negative=int(k[3]), contact_positive=int(k[4])) for k in uniq]
    return conds, [cid[k] for k in keys[H.k0:]], cid


def worker_cfg(analysis, conds, sched, images=None):
    akw = dict(reducer="pca", k=3, n_fit=10) if analysis == "procrustes" else dict(images=images, target_class=sched[12], n_fit=10)
    return dict(streams=list(bank_folders()), iti_s=2.0, analysis=analysis, analysis_kw=akw, stimulator="dummy",
                stimulator_kw=dict(reprog_ms=100.0), conditions=conds, schedule=sched, log=str(OUT / f"phase2b_worker_log_{analysis}.json"))


def build_placeholder_readout(seconds):
    """Pre-loaded placeholder decoder: one class (and PNG) per stimulation condition, nearest-centroid on
    0-500 ms mean rates of Phase 1 online-decoded spikes from every trial OUTSIDE the replay window, in the
    worker's pooled unit order (streams A,B,C; hello unit order). Saved in linear form (W, b, classes)."""
    from streamsort.core_cpp import load_exported
    conds, sched, cid = conditions(); fs = 29970.0757
    win = set(range(H.k0, H.k0 + int(np.ceil(seconds / 2.0)) + 1))
    cols, R = [], None
    for p, folder in zip(PROBES, bank_folders().values()):
        t = load_exported(folder); used = np.zeros(len(t["unit_ids"]), bool); used[np.asarray(t["cand_units"], int)] = True
        units = np.asarray(t["unit_ids"])[used]
        z = np.load(ctx.R_probe("jlh32", p, "online_spikes_all_trials_sup.npz")); off = z["meta_pulse_offset"]
        rel = z["frame"] - off[z["trial"]]; ok = (rel >= int(0.002 * fs)) & (rel < int(0.502 * fs))
        col = {int(u): i for i, u in enumerate(units)}; ci = np.array([col.get(int(u), -1) for u in z["unit"][ok]])
        M = np.zeros((len(off), len(units))); np.add.at(M, (z["trial"][ok][ci >= 0], ci[ci >= 0]), 1.0 / 0.5)
        cols.append(M)
    R = np.concatenate(cols, axis=1); y = np.array([cid[k] for k in H.keys])
    tr = np.array([i not in win for i in range(len(y))])
    classes = sorted(set(y[tr].tolist())); C = np.stack([R[tr & (y == c)].mean(0) for c in classes])
    s = float(np.median(np.linalg.norm(R[tr] - C[[classes.index(c) for c in y[tr]]], axis=1)))
    W = (C / s ** 2).T; b = -(C ** 2).sum(1) / (2 * s ** 2)
    path = OUT / "phase2b_placeholder_readout.npz"
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32), classes=np.asarray(classes, np.int64))
    acc = float(np.mean(np.argmax(R[~tr] @ W + b, 1) == np.array([classes.index(c) if c in classes else -1 for c in y[~tr]])))
    print(f"[readout] {len(classes)} classes, {int(tr.sum())} training trials, held-out window accuracy {acc:.2f}", flush=True)
    return path


def headless(seconds):
    import zmq
    from streamsort.worker2.engine import Engine, ZmqWorker
    zctx = zmq.Context(); conds, sched, cid = conditions()
    eng = Engine(worker_cfg("procrustes", conds, sched)); zw = ZmqWorker(eng, pub=PUB, cond=COND, context=zctx).start()
    em = Emulator(H.RD, bank_folders(), H.T0, seconds, pace=1.0, pub=PUB, cond=COND, context=zctx, threads=1, record=True)
    eng.arm(); t = time.perf_counter(); em.run(); time.sleep(1.0); wall = time.perf_counter() - t
    zw.stop(); eng.close(); em.close()
    T = pd.DataFrame(eng.rows); S = pd.DataFrame(eng.stim_rows)
    eq = {}
    for p in PROBES:
        name = f"Neuropix-PXI-104.Probe{p}-AP"; st = H.sess.stream(name)
        got = (np.concatenate([a for a, _ in em.recorded[name]]), np.concatenate([b for _, b in em.recorded[name]]))
        eq[p] = H.compare(got, H.reference(p, H.T0, seconds, H.cond_points(st, H.T0, seconds)))
    exp = [sched[int(k)] for k in T.k]
    R = dict(seconds=seconds, wall_s=wall, n_ttl=em.n_ttl, n_trials=len(T), messages=zw.n_msgs,
             all_streams_paired=bool((T.n_streams_pulse == 3).all()), ttl_index_sequential=bool((np.diff(T.k) == 1).all()),
             condition_sequence_matches=bool([int(c) for c in T.cond] == exp), n_programmed=int((S.stim == "programmed").sum()),
             n_hold=int((S.stim == "hold").sum()), n_loss=int(T.loss.notna().sum()),
             ttl_to_loss_ms=dict(median=float(T.ttl_to_loss_ms.median()), max=float(T.ttl_to_loss_ms.max())),
             ttl_to_stim_ms=dict(median=float(S.ttl_to_stim_ms.median()), max=float(S.ttl_to_stim_ms.max())),
             geometry_ms_median=float(T.geometry_ms.median()), emulator_block_rt=dict(
                 median=float(np.median(em.block_s) / (1024 / 29970.0757)), p99=float(np.percentile(em.block_s, 99) / (1024 / 29970.0757))),
             spikes_identical_to_python=eq, latency_reference=str(T.ref.iloc[0]) if len(T) else None)
    (OUT / "phase2b_e2e.json").write_text(json.dumps(R, indent=1, default=float)); print(json.dumps(R, default=float))


def gui(analysis, seconds):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import zmq
    from PySide6 import QtWidgets
    from streamsort.worker2.gui import WorkerWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    zctx = zmq.Context(); conds, sched, cid = conditions()
    make_placeholder_images(len(conds), OUT / "placeholder_images")
    readout = str(build_placeholder_readout(seconds)) if analysis == "image_decoder" else ""
    cfg = dict(pub=PUB, cond=COND, conditions=conds, schedule=sched, analysis=analysis, n_fit=10, readout=readout,
               images_dir=str(OUT / "placeholder_images"), target_condition=sched[12], streams=list(bank_folders()),
               log=str(OUT / f"phase2b_gui_log_{analysis}.json"))
    w = WorkerWindow(cfg, context=zctx); w.resize(1400, 820); w.show()
    shots = {"configure": FIG / f"phase2b_worker_gui_{analysis}_configure.png"}
    w.grab().save(str(shots["configure"]))
    w.arm()
    em = Emulator(H.RD, bank_folders(), H.T0, seconds, pace=1.0, pub=PUB, cond=COND, context=zctx, threads=1)
    th = threading.Thread(target=em.run, daemon=True); th.start()
    while th.is_alive():
        app.processEvents(); time.sleep(0.02)
    for _ in range(50):
        app.processEvents(); time.sleep(0.02)
    name = "procrustes" if analysis == "procrustes" else "decoder"
    w.grab().save(str(FIG / f"phase2b_worker_gui_{name}.png"))
    n = len(w.engine.rows); holds = w.holds; state = w.state
    w.stop(); em.close(); w.close()
    print(json.dumps(dict(analysis=analysis, trials=n, holds=holds, state=state, n_ttl=em.n_ttl,
                          lat_loss_median=float(np.median(w.lat_loss)) if w.lat_loss else None,
                          lat_stim_median=float(np.median(w.lat_stim)) if w.lat_stim else None)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "headless"
    if mode == "headless":
        headless(float(sys.argv[2]) if len(sys.argv) > 2 else 60.0)
    else:
        gui(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 60.0)
