"""Phase 2b: decoder bank from any Kilosort folder."""
import json
from pathlib import Path
import numpy as np, pytest
from conftest import ROOT_REPO

REPO = Path(ROOT_REPO)
BANK = REPO / "results" / "phase2b" / "bank_jlh32A_seg600"
needs_bank = pytest.mark.skipif(not (BANK / "decoder_tables.txt").exists(), reason="run `streamsort bank` on the seg600 KS folder first")


def test_read_ks_folder_synthetic(tmp_path):
    from streamsort.bank_export import read_ks_folder
    np.save(tmp_path / "spike_times.npy", np.array([[30], [10], [20], [40]], np.uint64))
    np.save(tmp_path / "spike_clusters.npy", np.array([2, 1, 2, 1], np.int32))
    np.save(tmp_path / "channel_map.npy", np.array([0, 1, 3], np.int32))
    np.save(tmp_path / "channel_positions.npy", np.array([[0, 0], [0, 20], [0, 60]], float))
    (tmp_path / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n2\tmua\n")
    fr, lab, cm, pos = read_ks_folder(tmp_path, offset_samples=100)
    assert fr[1].tolist() == [110, 140] and fr[2].tolist() == [120, 130]
    assert lab == {1: "good", 2: "mua"} and cm.tolist() == [0, 1, 3] and pos.shape == (3, 2)


@needs_bank
def test_bank_from_ks_folder_matches_phase1():
    from streamsort.sorting.templates import TemplateBank
    from streamsort.online.whitening import Whitening
    import pandas as pd
    a = TemplateBank.load(BANK / "templates.npz"); b = TemplateBank.load(REPO / "results" / "templates_jlh32_probeA.npz")
    assert np.array_equal(a.unit_ids, b.unit_ids) and np.array_equal(a.templates, b.templates)
    assert np.array_equal(Whitening.load(BANK / "whitening.npz").W, Whitening.load(REPO / "results" / "whitening_probeA.npz").W)
    assert np.array_equal(pd.read_csv(BANK / "sorting_report.csv").usable.to_numpy(bool),
                          pd.read_csv(REPO / "results" / "sorting_report.csv").usable.to_numpy(bool))
    info = json.loads((BANK / "bank_info.json").read_text())
    assert info["n_units"] == 327 and info["n_usable"] == 248


@needs_bank
def test_plugin_loader_reads_bank():
    from streamsort.core_cpp import _module
    C, n_units, n_usable, fs, fp = _module().load_tables_check(str(BANK))
    assert (C, n_units, n_usable) == (200, 327, 248) and abs(fs - 29970.0757) < 1e-3 and fp


def test_cli_bank_and_new_subcommands_parse():
    from streamsort.cli import build_parser
    ap = build_parser()
    assert ap.parse_args(["bank", "--ks-folder", "k", "--rec-dir", "r", "--stream", "s"]).cmd == "bank"
    assert ap.parse_args(["emulate", "--session", "s", "--bank", "A=x", "--t0", "1"]).cmd == "emulate"
    assert ap.parse_args(["worker-zmq", "--config", "w.json"]).cmd == "worker-zmq"
    assert ap.parse_args(["worker-gui"]).cmd == "worker-gui"
