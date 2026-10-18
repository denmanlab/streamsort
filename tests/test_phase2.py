"""Phase 2: CLI, configs, live-file handling, plugin core, worker, portability, replay equality."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pytest
from conftest import DATA, needs_data, needs_results, ROOT_REPO

REPO = Path(ROOT_REPO)
RD = DATA + "/Record Node 105/experiment1/recording1"


# ---- CLI ---------------------------------------------------------------------------
def test_cli_parses_all_subcommands():
    from streamsort.cli import build_parser
    ap = build_parser()
    for argv in (["fit", "--rec-dir", "x/a/b/c", "--stream", "S", "--out", "o"],
                 ["watch", "--rec-dir", "x/a/b/c", "--stream", "S", "--out", "o", "--poll-s", "1"],
                 ["replay", "--session", "s", "--probe-config", "a.json", "--t0", "1"],
                 ["worker", "--config", "w.json"]):
        assert ap.parse_args(argv).cmd == argv[0]


def test_fit_dry_run_without_and_with_nwb(tmp_path, capsys):
    from streamsort.cli import main
    base = ["fit", "--rec-dir", RD, "--stream", "Neuropix-PXI-104.ProbeA-AP", "--nch", "200", "--out", str(tmp_path), "--dry-run"]
    assert main(base) == 0
    d = json.loads(capsys.readouterr().out)
    assert [s.split()[0].split("/")[-1] for s in d["steps"]] == ["fit_initial_sort.py", "build_template_bank.py", "est_whitening.py"]
    assert d["env"]["REC_NWB"] == "" and d["env"]["NCH"] == "200"
    assert main(base + ["--nwb", "x.nwb"]) == 0
    assert json.loads(capsys.readouterr().out)["steps"][0].endswith("prep_session.py")


def test_console_entry_point_declared():
    txt = (REPO / "pyproject.toml").read_text()
    assert 'streamsort = "streamsort.cli:main"' in txt
    dep = txt.split("dependencies = [")[1].split("]")[0]
    assert "spikeinterface" not in dep            # online install stays spikeinterface-free


def test_ctx_custom_recording_env():
    env = dict(os.environ, REC="custom", PROBE="B", REC_ROOT="/r", REC_DIR="/r/Record Node 7/experiment1/recording1",
               STREAM="S-AP", NCH="96", OUT_DIR="/o", PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'scripts'); import ctx; "
                          "print(ctx.SI_STREAM, ctx.NCH, ctx.OUT, ctx.NWB == '', ctx.LEGACY)"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.stdout.split() == ["Record", "Node", "7#S-AP", "96", "/o", "True", "False"], out.stderr


# ---- configs -------------------------------------------------------------------------
@needs_results
def test_plugin_config_roundtrip_and_fingerprint(tmp_path, cfg):
    from streamsort.cli import build_parser, plugin_config
    a = build_parser().parse_args(["fit", "--rec-dir", RD, "--stream", "Neuropix-PXI-104.ProbeA-AP", "--nch", "200",
                                   "--out", str(REPO / "results"), "--plugin-ttl-line", "5"])
    c = plugin_config(a); p = tmp_path / "streamsort_oe_A.json"; p.write_text(json.dumps(c))
    assert json.loads(p.read_text()) == c
    from streamsort.sorting.templates import TemplateBank
    from streamsort.oe.processor import _CFG_FIELDS
    from streamsort.config import StreamConfig
    import dataclasses
    bank = TemplateBank.load(REPO / "results" / "templates_jlh32_probeA.npz")
    s = StreamConfig(**{k: v for k, v in bank.config.items() if k in _CFG_FIELDS})
    s = dataclasses.replace(s, detect_threshold=4.0, n_channels_used=200)
    assert s.fingerprint() == dataclasses.replace(cfg, n_channels_used=200).fingerprint()


def test_gui_microvolt_roundtrip_exact():
    c = np.arange(-32768, 32768, dtype=np.int16)
    uV = c.astype(np.float32) * np.float32(0.195)          # what the GUI hands to process()
    assert np.array_equal(np.rint(uV / 0.195).astype(np.int16), c)


# ---- live files ------------------------------------------------------------------------
def test_read_growing_npy(tmp_path):
    from streamsort.watch import read_growing_npy
    p = tmp_path / "ts.npy"; np.save(p, np.zeros(0, np.float64))         # header says shape (0,)
    with open(p, "ab") as f:
        f.write(np.arange(10, dtype=np.float64).tobytes() + b"\x00\x01")  # 10 entries + partial write
    assert np.array_equal(read_growing_npy(p), np.arange(10.0))


def test_wait_for_segment_size_stable(tmp_path):
    from streamsort.watch import wait_for_segment
    rec = tmp_path / "Record Node 1" / "experiment1" / "recording1"; d = rec / "continuous" / "S"; d.mkdir(parents=True)
    (rec / "structure.oebin").write_text(json.dumps({"continuous": [{"folder_name": "S/", "num_channels": 2, "sample_rate": 100.0}]}))
    (d / "continuous.dat").write_bytes(b"\x00" * 2 * 2 * 50)
    with pytest.raises(TimeoutError):
        wait_for_segment(rec, "S", 1.0, poll_s=0.01, timeout_s=0.05, log=lambda *_: None)
    (d / "continuous.dat").write_bytes(b"\x00" * 2 * 2 * 100)
    assert wait_for_segment(rec, "S", 1.0, poll_s=0.01, log=lambda *_: None) == 400


@needs_data
def test_snapshot_is_self_consistent(tmp_path):
    from streamsort.watch import snapshot
    from streamsort.io import OpenEphysSession, FileStreamSource
    snap = snapshot(RD, "Neuropix-PXI-104.ProbeA-AP", 2.0, tmp_path, event_streams=["NI-DAQmx-107.PXI-6133"])
    st, st0 = OpenEphysSession(snap).stream("Neuropix-PXI-104.ProbeA-AP"), OpenEphysSession(RD).stream("Neuropix-PXI-104.ProbeA-AP")
    assert st.n_samples == 60000
    assert np.array_equal(np.asarray(FileStreamSource(st).read(59000, 1000)), np.asarray(FileStreamSource(st0).read(59000, 1000)))
    assert not list(tmp_path.rglob("*.part"))


@needs_results
def test_bank_save_is_atomic(tmp_path):
    from streamsort.sorting.templates import TemplateBank
    b = TemplateBank.load(REPO / "results" / "templates_jlh32_probeA.npz"); out = tmp_path / "templates.npz"
    b.save(out); m0 = out.stat().st_mtime_ns; time.sleep(0.01); b.save(out)
    assert out.stat().st_mtime_ns > m0 and [p.name for p in tmp_path.iterdir()] == ["templates.npz"]
    assert np.array_equal(TemplateBank.load(out).templates, b.templates)


# ---- plugin core: TTL / artifact pairing -------------------------------------------------
def _pairer(ttl_line):
    from streamsort.oe.processor import DecoderProcessor
    d = DecoderProcessor.__new__(DecoderProcessor)
    d.cfg, d.fs, d.ttl_line = {}, 30000.0, ttl_line
    class S: events = []
    d.sup, d._n_ev, d._cand, d._pending_ttl, d._ttl_k, d.head = S(), 0, [], [], -1, 0
    return d


def test_pairing_ttl_and_artifact():
    d = _pairer(5)
    d.head = 1000; d.ttl(5, True); d.ttl(3, True)                 # stim TTL, then another line (ignored)
    d.sup.events += [dict(onset=1500)]; d.head = 2024
    assert d._pair() == [(1500, 0)]
    d.sup.events += [dict(onset=60000)]; d.head = 61000          # clip without TTL ...
    assert d._pair() == []
    d.head = 61000 + 2000; assert d._pair() == [] and d._cand == []   # ... dropped after tol
    d.head = 90000; d.ttl(5, True); d.head = 90000 + 2000          # TTL without artifact -> block start
    assert d._pair() == [(90000, 1)]


def test_pairing_artifact_only():
    d = _pairer(None); d.sup.events += [dict(onset=10), dict(onset=70000)]
    assert d._pair() == [(10, None), (70000, None)]


# ---- worker ------------------------------------------------------------------------------
def _feed(w, probes, n_trials, fs=30000.0, iti=2.0):
    rng = np.random.default_rng(0); units = np.arange(5)
    for p in probes:
        w.handle(("hello", p, fs, units, 0))
    for k in range(n_trials):
        on = int((1 + k * iti) * fs)
        for p in probes:
            w.handle(("pulse", p, on + (3 if p == "B" else 0), k, time.perf_counter()))
        for p in probes:
            f = np.sort(rng.integers(on, on + int(0.6 * fs), 200)); w.handle(("buf", p, on + int(0.6 * fs), f, rng.integers(0, 5, 200)))
    return w


def test_worker_pools_probes_and_computes_loss():
    from streamsort.oe.worker import Worker
    w = Worker(dict(probes=["A", "B"], schedule=["c0", "c1"], analysis_kw=dict(n_fit=4), reprog_ms=0.0))
    _feed(w, ["A", "B"], 8)
    assert len(w.rows) == 8 and all(r["n_probes_pulse"] == 2 and r["n_units"] == 10 for r in w.rows)
    assert sum(r["loss"] is not None for r in w.rows) == 4


def test_worker_reducer_and_decoder_swappable(tmp_path):
    from streamsort.oe.worker import Worker
    for red in ("pca", "fa"):
        w = Worker(dict(probes=["A"], analysis_kw=dict(reducer=red, n_fit=3)))
        assert sum(r["loss"] is not None for r in _feed(w, ["A"], 5).rows) == 2
    m = tmp_path / "readout.npz"; np.savez(m, W=np.ones((5, 2)), b=np.zeros(2), classes=np.array(["x", "y"]))
    w = Worker(dict(probes=["A"], analysis="decoder", analysis_kw=dict(model=str(m), target_class="y")))
    assert all(abs(r["loss"] - np.log(2)) < 1e-9 for r in _feed(w, ["A"], 3).rows)


def test_worker_holds_when_deadline_missed():
    import threading
    from streamsort.oe.worker import Worker, DummyStimulator
    w = Worker(dict(probes=["A"], schedule=["c0", "c1"], iti_s=0.05), stimulator=DummyStimulator(100.0))
    threading.Thread(target=w._stim_loop, daemon=True).start()
    _feed(w, ["A"], 2, iti=2.0); time.sleep(0.3); w.stim_q.put(None); w.done.wait(2)
    assert [r["stim"] for r in w.stim_rows] == ["hold", "hold"]


# ---- portability ------------------------------------------------------------------------
def test_online_path_imports_no_spikeinterface():
    for mods, banned in (("streamsort.oe.processor, streamsort.oe.worker", ("spikeinterface", "pandas", "torch", "kilosort")),
                         ("streamsort.oe.replay, streamsort.watch, streamsort.cli", ("spikeinterface", "torch", "kilosort"))):
        code = (f"import sys; import {mods}; "
                f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({banned!r})))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)), capture_output=True, text=True)
        assert out.stdout.strip() == "[]", mods + out.stdout + out.stderr


def test_plugin_files_portable():
    files = list((REPO / "streamsort" / "oe").glob("*.py")) + [REPO / "oe" / "streamsort_oe_processor.py", REPO / "streamsort" / "watch.py"]
    for f in files:
        t = f.read_text()
        assert "/Users/" not in t and "C:\\\\" not in t, f
    plug = (REPO / "oe" / "streamsort_oe_processor.py").read_text()
    assert plug.index('os.environ[_v] = "1"') < plug.index("import numpy")
    for m in ("def process(self, data)", "def start_acquisition(self)", "def stop_acquisition(self)",
              "def handle_ttl_event(self, source_node, channel, sample_number, line, state)"):
        assert m in plug


# ---- replay equality (short) ----------------------------------------------------------------
@needs_data
@needs_results
def test_replay_equals_pull_decoder_short():
    sys.path.insert(0, str(REPO / "scripts"))
    import replay_harness as H
    from streamsort.oe import replay
    pts = H.cond_points(H.sess.stream("Neuropix-PXI-104.ProbeA-AP"), H.T0, 6.0); ref = H.reference("A", H.T0, 6.0, pts)
    r = replay.run(H.RD, {"A": H.probe_cfg("A")}, H.T0, 6.0, buffer=1024, ttl_times=H.ttl, pace=0, schedule_fn=H.scheduler(pts))
    assert H.compare(r["spikes"]["A"], ref)["identical"]
