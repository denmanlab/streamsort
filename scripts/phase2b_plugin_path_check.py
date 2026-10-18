"""Plugin-path check: the plugin's own StreamDecoder + ZmqLink (C++, built as tests/plugin_path_driver)
fed GUI-style blocks of jlh32 probe A; published messages vs the Python reference decoder."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_harness as H  # noqa: E402
from streamsort.io import FileStreamSource  # noqa: E402

D = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
W = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/phase2b/plugin_path"); W.mkdir(parents=True, exist_ok=True)
DRV = Path("plugin/streamsort-oe/tests/build/plugin_path_driver")
st = H.sess.stream("Neuropix-PXI-104.ProbeA-AP"); s0 = int(st.time_to_sample(np.array([H.T0]))[0]); n = int(D * st.effective_rate)
raw = W / "raw_probeA_200ch.bin"                     # 200 ch int16, workspace temp (removed below)
np.asarray(FileStreamSource(st, 200).read(s0, n)).astype(np.int16).tofile(raw)
ttl = st.time_to_sample(H.ttl) - s0; ttl = ttl[(ttl >= 0) & (ttl < n)]
(W / "ttl.txt").write_text("\n".join(map(str, ttl)))
pts = H.cond_points(st, H.T0, D); ids = {}
(W / "conds.txt").write_text("\n".join(f"{s} {ids.setdefault(k, len(ids))}" for s, k in pts))
r = subprocess.run([str(DRV), "results/phase2b/bank_jlh32A_seg600", str(raw), "200", "1024", str(W / "ttl.txt"), str(W / "conds.txt"),
                    str(W / "messages.jsonl")], capture_output=True, text=True)
print(r.stderr.strip())
msgs = [json.loads(l) for l in (W / "messages.jsonl").read_text().splitlines()]
kinds = {}
for m in msgs:
    kinds[m["env"]] = kinds.get(m["env"], 0) + 1
sp = [m for m in msgs if m["env"] == "SPIKES"]
frames = np.concatenate([np.asarray(m.get("frames", []), np.int64) for m in sp]); units = np.concatenate([np.asarray(m.get("units", []), np.int64) for m in sp])
ref = H.reference("A", H.T0, D, pts)
pl = [m["hdr"] for m in msgs if m["env"] == "PULSE"]
hello = [m["hdr"] for m in msgs if m["env"] == "STATUS"][0]
R = dict(seconds=D, message_counts=kinds, spikes=H.compare((frames, units), ref), n_ttl=int(len(ttl)), n_pulses=len(pl),
         pulses_with_artifact=sum(p["onset_sample"] >= 0 for p in pl), ttl_index_sequential=[p["ttl_index"] for p in pl] == list(range(len(pl))),
         onset_minus_ttl_ms=[round((p["onset_sample"] - int(t)) / st.effective_rate * 1e3, 3) for p, t in zip(pl, ttl)],
         hello_units=len(hello["unit_ids"]), heads_monotonic=bool(np.all(np.diff([m["hdr"]["head"] for m in sp]) >= 0)))
(W / "plugin_path_check.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R))
raw.unlink()
