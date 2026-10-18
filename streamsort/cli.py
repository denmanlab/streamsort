"""streamsort command line.

  streamsort fit     initial sort of a leading segment -> templates.npz, whitening.npz, plugin config
  streamsort watch   wait for the segment of a recording still being written, snapshot it, fit
  streamsort replay  replay a recorded session through the plugin code path (+ worker)
  streamsort worker  run the pooling / analysis / stimulator worker for live use

``fit``/``watch`` run the Phase 1 scripts (spikeinterface + Kilosort 4) from a repository
checkout; ``replay``/``worker`` and the Open Ephys processor never import spikeinterface.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _fit_env(a) -> dict:
    rec_dir = Path(a.rec_dir).resolve()
    out = Path(a.out).resolve()
    env = dict(os.environ, REC=a.name, PROBE=a.probe, REC_ROOT=str(rec_dir.parents[2]), REC_DIR=str(rec_dir),
               STREAM=a.stream, NCH=str(a.nch), OUT_DIR=str(out), CTX_FIG=str(out / "figures"),
               SCRATCH_DIR=str(Path(a.scratch).resolve() if a.scratch else out / "scratch"),
               TTL_STREAM=a.ttl_stream, TTL_LINE=str(a.ttl_line), REC_NWB=str(Path(a.nwb).resolve()) if a.nwb else "",
               SEG_TAG=f"seg{int(a.seconds)}", SORTER=a.sorter,
               PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return env


def fit_steps(a) -> list[list[str]]:
    py = sys.executable
    steps = [[py, str(SCRIPTS / "prep_session.py")]] if a.nwb else []
    steps += [[py, str(SCRIPTS / "fit_initial_sort.py"), "--seconds", str(a.seconds), "--sorter", a.sorter,
               "--n-jobs", str(a.n_jobs), "--tag", f"seg{int(a.seconds)}"],
              [py, str(SCRIPTS / "build_template_bank.py")],
              [py, str(SCRIPTS / "est_whitening.py")]]
    return steps


def plugin_config(a) -> dict:
    out = Path(a.out).resolve()
    return dict(probe=a.probe, stream=a.stream, templates=str(out / "templates.npz"), whitening=str(out / "whitening.npz"),
                sorting_report=str(out / "sorting_report.csv"), nch=a.nch, bit_volts=a.bit_volts, input="uV",
                detect_threshold=4.0, min_energy=50.0, ttl_line=a.plugin_ttl_line, worker_address=["127.0.0.1", 6010],
                repo=str(REPO))


def cmd_fit(a) -> int:
    env, steps = _fit_env(a), fit_steps(a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cfg_path = Path(a.out) / f"streamsort_oe_{a.probe}.json"
    if a.dry_run:
        print(json.dumps(dict(env={k: env[k] for k in ("REC", "PROBE", "REC_ROOT", "REC_DIR", "STREAM", "NCH",
                                                         "OUT_DIR", "SCRATCH_DIR", "TTL_STREAM", "TTL_LINE", "REC_NWB")},
                              steps=[" ".join(s[1:]) for s in steps], plugin_config=str(cfg_path)), indent=1))
        return 0
    for s in steps:
        print(f"[fit] {' '.join(Path(x).name if i == 1 else x for i, x in enumerate(s[1:], 1))}", flush=True)
        r = subprocess.run(s, env=env, cwd=str(REPO))
        if r.returncode:
            print(f"[fit] step failed ({r.returncode})", file=sys.stderr)
            return r.returncode
    cfg_path.write_text(json.dumps(plugin_config(a), indent=1))
    from .bank_export import export_from_bank_files
    ks_out = Path(env["SCRATCH_DIR"]) / f"sort_{a.sorter}_seg{int(a.seconds)}" / "sorter_output"
    folders = [Path(a.out) / "streamsort"] + ([ks_out / "streamsort"] if ks_out.exists() else [])
    export_from_bank_files(a.out, a.rec_dir, a.stream, a.nch, folders)
    print(f"[fit] done: {Path(a.out) / 'templates.npz'} | plugin config {cfg_path} | plugin bank {', '.join(map(str, folders))}")
    return 0


def cmd_watch(a) -> int:
    from . import watch
    snap_root = Path(a.snapshot_dir or Path(a.out) / "segment").resolve()
    watch.wait_for_segment(a.rec_dir, a.stream, a.seconds + 1.0, poll_s=a.poll_s, timeout_s=a.timeout_s)
    snap = watch.snapshot(a.rec_dir, a.stream, a.seconds + 1.0, snap_root, event_streams=[a.ttl_stream])
    print(f"[watch] snapshot {snap}", flush=True)
    a.rec_dir = str(snap)
    return cmd_fit(a)


def cmd_replay(a) -> int:
    import numpy as np
    from .io.openephys import OpenEphysSession
    from .oe import replay
    cfgs = {}
    for f in a.probe_config:
        c = replay.load_probe_cfg(f); cfgs[str(c["probe"])] = c
    sess = OpenEphysSession(a.session)
    ttl = sess.ttl_edges(a.ttl_stream, a.ttl_line, True) if a.ttl_stream else None
    wcfg = json.loads(Path(a.worker_config).read_text()) if a.worker_config else None
    if wcfg is not None:
        wcfg.setdefault("probes", list(cfgs))
        if a.log:
            wcfg["log"] = str(Path(a.log).resolve())
    r = replay.run(a.session, cfgs, a.t0, a.seconds, buffer=a.buffer, ttl_times=ttl, worker_cfg=wcfg,
                   pace=a.pace, transport=a.transport)
    fs = next(iter(r["feeders"].values())).src.stream.effective_rate
    summ = dict(probes={p: dict(n_spikes=int(len(r["spikes"][p][0])),
                                rt_factor_median=float(np.median(np.asarray(v) / (a.buffer / fs))),
                                rt_factor_p99=float(np.percentile(np.asarray(v) / (a.buffer / fs), 99)))
                        for p, v in r["buffer_s"].items()},
                n_trials=len(r["worker"]["trials"]) if r["worker"] else None, startup_s=r["startup_s"])
    print(json.dumps(summ, indent=1))
    return 0


def cmd_bank(a) -> int:
    from .bank_export import build_bank
    out = build_bank(a.ks_folder, a.rec_dir, a.stream, a.nch, out=a.out, ttl_stream=a.ttl_stream or None, ttl_line=a.ttl_line,
                     offset_samples=a.offset_samples, noise_s=(a.noise_start, a.noise_dur), whiten_s=(a.whiten_start, a.whiten_dur))
    print(f"[bank] plugin bank written to {out}")
    return 0


def cmd_emulate(a) -> int:
    from .worker2.emulator import Emulator
    banks = dict(x.split("=", 1) for x in a.bank)
    em = Emulator(a.session, banks, a.t0, a.seconds, buffer=a.buffer, pace=a.pace, ttl_stream=a.ttl_stream or None,
                  ttl_line=a.ttl_line, threads=a.threads, pub=a.pub, cond=a.cond)
    print(f"[emulate] {len(banks)} stream(s), {em.n_ttl if em.n_ttl else len(em.ttl)} stim TTLs, publishing on {a.pub}", flush=True)
    try:
        em.run()
    finally:
        em.close()
    return 0


def cmd_worker_zmq(a) -> int:
    import time
    from .worker2.engine import Engine, ZmqWorker
    cfg = json.loads(Path(a.config).read_text())
    eng = Engine(cfg); zw = ZmqWorker(eng, pub=cfg.get("pub", a.pub), cond=cfg.get("cond", a.cond)).start(); eng.arm()
    print(f"[worker] subscribed to {cfg.get('pub', a.pub)}, conditions on {cfg.get('cond', a.cond)}; Ctrl-C to stop", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        zw.stop(); eng.close()
    return 0


def cmd_worker_gui(a) -> int:
    from .worker2.gui import main as gui_main
    return gui_main(a.config)


def cmd_worker(a) -> int:
    from .oe.worker import Worker
    cfg = json.loads(Path(a.config).read_text())
    print(f"[worker] listening on {cfg.get('worker_address')} for probes {cfg['probes']}", flush=True)
    Worker(cfg).serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="streamsort")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def fit_args(p):
        p.add_argument("--rec-dir", required=True, help="Open Ephys recording dir (.../Record Node N/experimentE/recordingR)")
        p.add_argument("--stream", required=True, help="AP stream folder, e.g. Neuropix-PXI-104.ProbeA-AP")
        p.add_argument("--probe", default="A")
        p.add_argument("--name", default="custom", help="recording label")
        p.add_argument("--nch", type=int, default=384, help="channels used (first N)")
        p.add_argument("--out", required=True)
        p.add_argument("--scratch", default=None)
        p.add_argument("--seconds", type=float, default=600.0)
        p.add_argument("--sorter", default="kilosort4")
        p.add_argument("--n-jobs", type=int, default=6)
        p.add_argument("--ttl-stream", default="NI-DAQmx-107.PXI-6133", help="stream carrying the stim TTL (for blanking)")
        p.add_argument("--ttl-line", type=int, default=5)
        p.add_argument("--plugin-ttl-line", type=int, default=None, help="stim TTL line on the probe stream (live)")
        p.add_argument("--bit-volts", type=float, default=0.195)
        p.add_argument("--nwb", default=None, help="optional NWB with trials (validation only)")
        p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("fit", help="initial sort + template bank + whitening"); fit_args(p); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("watch", help="wait for the leading segment of a live recording, snapshot, fit"); fit_args(p)
    p.add_argument("--snapshot-dir", default=None); p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--timeout-s", type=float, default=None); p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("replay", help="replay a session through the Open Ephys processor path")
    p.add_argument("--session", required=True, help="recording dir")
    p.add_argument("--probe-config", action="append", required=True, help="plugin JSON per probe (repeat)")
    p.add_argument("--worker-config", default=None)
    p.add_argument("--t0", type=float, required=True, help="start time (s, session clock)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--buffer", type=int, default=1024)
    p.add_argument("--pace", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    p.add_argument("--ttl-stream", default="NI-DAQmx-107.PXI-6133", help="stim TTL injected on the probe streams")
    p.add_argument("--ttl-line", type=int, default=5)
    p.add_argument("--transport", default="listener", choices=["listener", "pipe"])
    p.add_argument("--log", default=None); p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bank", help="decoder bank (plugin tables) from any Kilosort output folder")
    p.add_argument("--ks-folder", required=True); p.add_argument("--rec-dir", required=True); p.add_argument("--stream", required=True)
    p.add_argument("--nch", type=int, default=384); p.add_argument("--out", default=None, help="default <ks-folder>/streamsort")
    p.add_argument("--ttl-stream", default="NI-DAQmx-107.PXI-6133", help="stim TTL stream ('' = none)"); p.add_argument("--ttl-line", type=int, default=5)
    p.add_argument("--offset-samples", type=int, default=0, help="KS spike_times offset into the stream (sorted segment start)")
    p.add_argument("--noise-start", type=float, default=10.0); p.add_argument("--noise-dur", type=float, default=30.0)
    p.add_argument("--whiten-start", type=float, default=60.0); p.add_argument("--whiten-dur", type=float, default=20.0)
    p.set_defaults(fn=cmd_bank)
    p = sub.add_parser("emulate", help="replay a session through the C++ core, publishing the plugin's ZMQ messages")
    p.add_argument("--session", required=True); p.add_argument("--bank", action="append", required=True, help="STREAM=bank_or_ks_folder (repeat)")
    p.add_argument("--t0", type=float, required=True); p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--buffer", type=int, default=1024); p.add_argument("--pace", type=float, default=1.0); p.add_argument("--threads", type=int, default=1)
    p.add_argument("--ttl-stream", default="NI-DAQmx-107.PXI-6133"); p.add_argument("--ttl-line", type=int, default=5)
    p.add_argument("--pub", default="tcp://*:5557"); p.add_argument("--cond", default="tcp://127.0.0.1:5558"); p.set_defaults(fn=cmd_emulate)
    p = sub.add_parser("worker-zmq", help="headless worker v2 (ZMQ client of the Open Ephys plugin)")
    p.add_argument("--config", required=True); p.add_argument("--pub", default="tcp://127.0.0.1:5557"); p.add_argument("--cond", default="tcp://*:5558")
    p.set_defaults(fn=cmd_worker_zmq)
    p = sub.add_parser("worker-gui", help="worker v2 with GUI (configure -> armed -> running)")
    p.add_argument("--config", default=None); p.set_defaults(fn=cmd_worker_gui)
    p = sub.add_parser("worker", help="live pooling / analysis / stimulator worker")
    p.add_argument("--config", required=True); p.set_defaults(fn=cmd_worker)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
