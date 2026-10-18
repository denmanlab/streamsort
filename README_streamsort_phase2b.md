# streamsort — Python package (Phase 2b)

Online template-matching spike decoding and per-trial population geometry for closed-loop electrical
stimulation. Phase 2b adds:
- a C++ decoder core with a Python binding;
- a decoder bank built from **any Kilosort folder**;
- **worker v2**, a ZMQ client of the Open Ephys plugin, with a GUI;
- a session **emulator** that publishes exactly what the plugin publishes.

The Open Ephys plugin itself is documented separately in `plugin/streamsort-oe/README.md`.
The Phase 2 Python Processor path (`oe/streamsort_oe_processor.py`, `streamsort replay`, `streamsort worker`) is kept
as the **legacy fallback**.

## Install

```
pip install -e .              # online path: numpy, scipy, numba, scikit-learn
pip install -e ".[fit]"       # + spikeinterface, kilosort: `streamsort fit`, `watch`, `bank`
pip install ./cpp             # C++ core -> Python module `streamsort_core` (scikit-build-core + pybind11)
pip install pyzmq PySide6 pyqtgraph pyserial    # worker v2, GUI, serial stimulator
```
For development, `cmake -S cpp -B cpp/build && cmake --build cpp/build` also works. `streamsort.core_cpp` then
finds the module in `cpp/build` (or set `STREAMSORT_CORE_PATH`). `fit`, `watch` and `bank` run from this repository checkout.

## Command line

| command | does |
|---|---|
| `streamsort fit --rec-dir R --stream S --nch N --out O [--seconds 600]` | KS4 initial sort → template bank → whitening. Also writes the plugin bank to `O/streamsort/` and `<sort>/sorter_output/streamsort/`. |
| `streamsort watch …` | Live recording: wait for the leading segment, snapshot it, then `fit`. |
| `streamsort bank --ks-folder K --rec-dir R --stream S --nch N [--out D]` | Plugin bank from **any** Kilosort output folder → `K/streamsort/`. Re-extracts templates through the causal path, estimates noise and whitening from a stimulus-free stretch (`--whiten-start/--whiten-dur`, default 60 s / 20 s), and applies the Phase 1 usable rule. |
| `streamsort emulate --session R --bank STREAM=K … --t0 T [--seconds 60] [--pace 1]` | Replay a session through the C++ core, publishing the plugin's ZMQ messages and obeying its condition back-channel. |
| `streamsort worker-gui [--config cfg.json]` | Worker v2 with GUI. |
| `streamsort worker-zmq --config cfg.json` | Worker v2, headless (same engine). |
| `streamsort replay`, `streamsort worker` | Phase 2 legacy (Python Processor path). |

Rehearse a session without the GUI (two terminals):
```
streamsort worker-gui --config results/phase2b/configs/worker_gui_example.json      # press Arm
streamsort emulate --session "<jlh32>/Record Node 105/experiment1/recording1" \
  --bank Neuropix-PXI-104.ProbeA-AP=<KS folder A> --bank Neuropix-PXI-104.ProbeB-AP=<KS folder B> \
  --bank Neuropix-PXI-104.ProbeC-AP=<KS folder C> --t0 3108.697 --seconds 120
```

## C++ core (`cpp/`, module `streamsort_core`)

`streamsort.core_cpp.CoreDecoder(tables, n_threads)` has the `OnlineDecoder` API (`start_at`, `push`, `flush`,
`set_condition`, `events`, `timing()`).

`tables_from_decoder(OnlineDecoder)` derives every table from the validated Python decoder, so both paths share
one set of constants. These tables include SOS and priming state, whitening (applied as a band), thresholds, neighbour
tables, candidate templates and artifact-suppressor parameters. `export_tables()` and `load_exported()` write and read
the folder format the plugin loads.

The arithmetic follows the numba reference kernels operation for operation, with no FMA and no fast-math. On jlh32 the
spike sets are **identical** to the Python decoder at every buffer size and thread count tested.

## Worker v2 (`streamsort/worker2/`)

- **Input:** ZMQ SUB to the plugin, or the emulator (`protocol.py` documents every message).
- **Pooling:** spikes are pooled across streams per trial. Pulses are matched by TTL index, or by time (±5 ms) without
  a TTL. A stream with no detected artifact borrows another stream's onset.
- **Binning:** 10 ms bins (20 ms smoothing), 2–502 ms after the pulse.
- **Analyses** (`analyses.py`, swappable):
  - `procrustes`: reducer `pca` | `fa` | `isomap` with k dimensions, fitted causally on condition means of the first
    `n_fit` trials. The target is the mean trajectory of the target condition, or a loaded `.npy`. Loss = Procrustes
    distance.
  - `image_decoder` (placeholder): one class per condition, each with a PNG from a fixed set of presented images.
    The readout is a loaded linear readout (`W`, `b`, `classes` npz), or nearest-centroid fitted on the first `n_fit`
    trials. Output = the PNG of the decoded class; loss = −log p(target class).
- **Stimulator** (`stimulator.py`): `dummy` (sleeps `reprog_ms`) or `serial` (pyserial; a command template is formatted
  with the condition's fields, with optional ack).
  - Reprogramming starts when the stim TTL message arrives, on its own thread.
  - If it cannot finish before the next pulse (TTL + ITI − 50 ms), the current parameters are **held**.
  - After programming, the condition id is sent to the plugin so its artifact suppressor uses the right template.
- **Logs:** per-trial log with TTL → loss ready and TTL → stimulator programmed (from `t_wall_us`). A warning is printed
  when TTL → loss exceeds the ITI.

**Conditions file:** `{"conditions": [{amplitude, polarity, pulse_duration, contact_negative, contact_positive}, …],
"schedule": [ids…]}`. A condition's id is its index; see `results/phase2b/configs/conditions_jlh32.json`.

### Worker GUI states

1. **Configure:** set addresses, ITI, window and bin; loss (Procrustes | image decoder); reducer and latent dimensions;
   fit trials; target condition or target file; image set and readout; stimulator (type, port, command, reprogramming
   time); conditions file. Load and Save as JSON.
2. **Armed** (after *Arm*): connected; the first condition is programmed and sent to the plugin. The target appears once
   it exists: the latent trajectory for Procrustes, or the target PNG for the decoder.
3. **Running:** each trial updates these displays:
   - the trial's aligned latents next to the target (PC1–3 vs time and the PC1 × PC2 plane), or the decoded image next
     to the target image with class probabilities;
   - loss per trial;
   - latency after the stim TTL (loss ready, stimulator programmed) against the ITI.

## Verification (jlh32, this machine: Apple M1, 8 GB)

| check | result |
|---|---|
| pytest (Phase 1 + 2 + 2b) | 56 passed |
| C++ core vs Python, A 60 s (1024/512) / A,B,C 120 s (1 and 2 threads) | identical (62727; 129324 / 112051 / 64641) |
| Kilosort-folder bank (Phase 1 seg600 KS4 folder) vs Phase 1 bank | 327 units, templates and whitening Δ = 0, usable 248 = 248, decode identical |
| Phase 2 120 s KS4 folder → bank | 187 units, 151 usable, loads in the C++ loader |
| emulate → worker v2 over ZMQ (inproc), 3 probes, 120 s, real-time pace | 60/60 trials, pulses paired on all 3 streams, condition order correct, 0 holds, spikes identical to Python on all probes |
| TTL → loss ready / TTL → stimulator programmed (100 ms) | median 534 / 109 ms, max 562 / 110 ms (ITI 2000 ms) |
| worker GUI offscreen vs emulator, 60 s each | Procrustes and image decoder: RUNNING, 30 trials, 0 holds |
| placeholder image readout (26 classes, trained outside the replay window) | held-out window accuracy 0.68 |

**Not tested here:** TCP ZMQ between processes (this sandbox blocks socket binding, so all ZMQ runs used `inproc`,
which uses the same sockets API); a serial stimulator; Windows.
