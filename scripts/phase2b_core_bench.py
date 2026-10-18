"""Quick C++-core check on jlh32 probe A: identity vs the Python decoder and per-stage timing.
usage: python scripts/phase2b_core_bench.py [seconds] [threads,...]"""
import sys, time, dataclasses
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_harness as H, ctx  # noqa: E402  (Phase 2 harness: read-only reuse of its reference decoder)
from streamsort.config import StreamConfig  # noqa: E402
from streamsort.io import FileStreamSource  # noqa: E402
from streamsort.online import OnlineDecoder, DecoderParams, Whitening  # noqa: E402
from streamsort.online.decoder import SpikeBatch  # noqa: E402
from streamsort.preprocess.suppressor import ArtifactSuppressor  # noqa: E402
from streamsort.sorting.templates import TemplateBank  # noqa: E402
from streamsort.core_cpp import tables_from_decoder, CoreDecoder  # noqa: E402

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
THREADS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2]
st = H.sess.stream("Neuropix-PXI-104.ProbeA-AP"); pts = H.cond_points(st, H.T0, DUR); ref = H.reference("A", H.T0, DUR, pts)
cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=H.NCH)
bank = TemplateBank.load(ctx.R_probe("jlh32", "A", "templates.npz"))
usable = pd.read_csv(ctx.probe_out("jlh32", "A") / "sorting_report.csv").usable.to_numpy(bool)
dec = OnlineDecoder(bank, cfg, Whitening.load(ctx.R_probe("jlh32", "A", "whitening.npz")), DecoderParams(min_energy=50.0),
                    units_mask=usable, check=False, suppressor=ArtifactSuppressor(cfg.sample_rate, H.NCH))
tab = tables_from_decoder(dec)
src = FileStreamSource(st, H.NCH); s0 = int(st.time_to_sample(np.array([H.T0]))[0]); n = int(DUR * st.effective_rate)
raw = np.asarray(src.read(s0, n)).astype(np.float32)
for nt in THREADS:
    cd = CoreDecoder(tab, nt); cd.start_at(0); out, bt = [], []
    edges = [0] + [s for s, _ in pts] + [n]; kk = [None] + [k for _, k in pts]
    for a, b, key in zip(edges[:-1], edges[1:], kk):
        if key is not None:
            cd.set_condition(key)
        for c in range(a, b, 1024):
            t = time.perf_counter(); out.append(cd.push(raw[c:min(c + 1024, b)])); bt.append(time.perf_counter() - t)
    out.append(cd.flush()); bb = SpikeBatch.concat(out); m = bb.unit >= 0
    rt = np.array(bt) / (1024 / st.effective_rate)
    print(nt, "threads identical:", H.compare((bb.frame[m], bb.unit[m]), ref)["identical"],
          f"rt median {np.median(rt):.3f} p99 {np.percentile(rt, 99):.3f}", {k: round(v, 2) for k, v in cd.timing().items()}, flush=True)
