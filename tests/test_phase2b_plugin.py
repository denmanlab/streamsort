"""Phase 2b: Open Ephys plugin build artefacts and the plugin's own decode/publish path."""
import json, subprocess, sys
from pathlib import Path
import pytest
from conftest import ROOT_REPO, needs_data

REPO = Path(ROOT_REPO)
BUNDLE = REPO / "plugin" / "streamsort-oe" / "build" / "streamsort-oe.bundle" / "Contents" / "MacOS" / "streamsort-oe"
DRIVER = REPO / "plugin" / "streamsort-oe" / "tests" / "build" / "plugin_path_driver"


@pytest.mark.skipif(not BUNDLE.exists(), reason="plugin not built (cmake -S plugin/streamsort-oe -B plugin/streamsort-oe/build)")
def test_plugin_bundle_exports_entry_points():
    out = subprocess.run(["nm", "-gU", str(BUNDLE)], capture_output=True, text=True).stdout
    assert "_getLibInfo" in out and "_getPluginInfo" in out


def test_plugin_sources_are_juce_free_where_required():
    for f in ("StreamDecoder.h", "StreamDecoder.cpp", "ZmqLink.h", "ZmqLink.cpp"):
        t = (REPO / "plugin" / "streamsort-oe" / "Source" / f).read_text()
        assert "Juce" not in t and "juce" not in t, f
    for f in (REPO / "cpp" / "core").rglob("*.*pp"):
        assert "juce" not in f.read_text().lower(), f


@needs_data
@pytest.mark.skipif(not DRIVER.exists(), reason="plugin path driver not built")
@pytest.mark.skipif(not (REPO / "results/phase2b/bank_jlh32A_seg600/decoder_tables.txt").exists(), reason="bank missing")
def test_plugin_path_identical_to_python(tmp_path):
    r = subprocess.run([sys.executable, "scripts/phase2b_plugin_path_check.py", "6", str(tmp_path)], cwd=REPO, capture_output=True, text=True)
    R = json.loads((tmp_path / "plugin_path_check.json").read_text())
    assert r.returncode == 0, r.stderr[-500:]
    assert R["spikes"]["identical"] and R["n_pulses"] == R["n_ttl"] and R["ttl_index_sequential"] and R["heads_monotonic"]
