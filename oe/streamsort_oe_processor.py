"""Open Ephys Python Processor script: streamsort online decoder for ONE probe stream.

Setup (one Python Processor per probe AP stream):
  1. copy this file once per probe, e.g. streamsort_oe_A.py, streamsort_oe_B.py (module names must
     differ: the GUI shares one interpreter between all Python Processors);
  2. put a JSON config with the same stem next to it (streamsort_oe_A.json), see README;
  3. start the worker first:  streamsort worker --config worker.json
  4. route the stim TTL onto each probe stream (basestation sync/SMA input) and set "ttl_line".
The GUI buffer is passed through unchanged.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"          # the signal-chain thread must not fan out (set before numpy loads)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

try:                                # numpy may already be loaded by another processor: pin at runtime too
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

CONFIG = Path(__file__).with_suffix(".json")


class PyProcessor:
    def __init__(self, processor, num_channels, sample_rate):
        cfg = json.loads(CONFIG.read_text())
        if cfg.get("repo"):
            import sys
            sys.path.insert(0, str(Path(cfg["repo"])))
        from streamsort.oe.processor import DecoderProcessor
        self.cfg, self.processor = cfg, processor
        self.core = DecoderProcessor(cfg, int(num_channels), float(sample_rate))
        self.conn = None

    def process(self, data):
        if self.core is not None and getattr(self.core, "head", None) is not None:
            self.core.process(data)

    def start_acquisition(self):
        from streamsort.oe import protocol
        addr = protocol.address(self.cfg)
        if addr is not None:
            from multiprocessing.connection import Client
            try:
                self.conn = Client(addr, family=protocol.family(self.cfg), authkey=protocol.authkey(self.cfg))
            except OSError as e:
                print(f"[streamsort] worker not reachable at {addr}: {e}; decoding without worker")
                self.conn = None
        self.core.conn = self.conn
        self.core.start(0)

    def stop_acquisition(self):
        self.core.stop()
        if self.conn is not None:
            self.conn.close()
        self.core.head = None
        t = np.asarray(self.core.timing)
        if len(t):
            print(f"[streamsort {self.core.probe}] buffers {len(t)} | median {np.median(t)*1e3:.2f} ms | "
                  f"p99 {np.percentile(t, 99)*1e3:.2f} ms | swaps {self.core.n_swaps}")

    def handle_ttl_event(self, source_node, channel, sample_number, line, state):
        self.core.ttl(line, state)

    def handle_spike(self, source_node, electrode_name, num_channels, num_samples, sample_number,
                     sorted_id, spike_data):
        pass

    def start_recording(self, recording_dir):
        pass

    def stop_recording(self):
        pass
