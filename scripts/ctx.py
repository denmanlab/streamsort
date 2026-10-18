"""Per-recording / per-probe context shared by every pipeline script.

Select with environment variables:  REC=jlh32|jlh34  PROBE=A|B|C
Outputs go to results/<REC>_probe<P>/ and figures/<REC>_probe<P>/, except the
original jlh32 probe A run, which keeps its Phase 1 file names in results/.
"""
import os
from pathlib import Path

RECS = {
    "jlh32": dict(root="/Users/danieldenman/ephys_data_stim/jlh32_2023-02-18_16-07-58", nwb="jlh32.nwb", nch=200),
    "jlh34": dict(root="/Users/danieldenman/ephys_data_stim/jlh34_2023-05-15_16-03-21", nwb="jlh34.nwb", nch=225),
}
REC = os.environ.get("REC", "jlh32"); PROBE = os.environ.get("PROBE", "A")
if os.environ.get("REC_ROOT"):            # any Open Ephys recording (streamsort fit / watch); NWB optional
    RECS[REC] = dict(root=os.environ["REC_ROOT"], nwb=os.environ.get("REC_NWB", ""), nch=int(os.environ.get("NCH", "384")))
_r = RECS[REC]
ROOT = _r["root"]; NWB = (f"{ROOT}/{_r['nwb']}" if _r["nwb"] and not os.path.isabs(_r["nwb"]) else _r["nwb"]); NCH = int(os.environ.get("NCH", _r["nch"]))
REC_DIR = os.environ.get("REC_DIR", f"{ROOT}/Record Node 105/experiment1/recording1")
STREAM = os.environ.get("STREAM", f"Neuropix-PXI-104.Probe{PROBE}-AP")
SI_STREAM = f"{Path(REC_DIR).parents[1].name}#{STREAM}"   # neo stream id: "<Record Node>#<stream>"
TTL_STREAM, TTL_LINE = os.environ.get("TTL_STREAM", "NI-DAQmx-107.PXI-6133"), int(os.environ.get("TTL_LINE", "5"))
TAG = f"{REC}_probe{PROBE}"
SCRATCH = Path(os.environ["SCRATCH_DIR"]) if os.environ.get("SCRATCH_DIR") else Path(f"/Users/danieldenman/stim_scratch/{TAG}")
LEGACY = (REC == "jlh32" and PROBE == "A" and not os.environ.get("OUT_DIR"))
OUT = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else (Path("results") if LEGACY else Path("results") / TAG)
FIG = Path(os.environ["CTX_FIG"]) if os.environ.get("CTX_FIG") else (Path("figures") if LEGACY else Path("figures") / TAG)
_LEGACY_NAMES = {"templates.npz": "templates_jlh32_probeA.npz", "whitening.npz": "whitening_probeA.npz",
                 "trials.csv": "jlh32_probeA_trials.csv", "session.json": "jlh32_probeA_session.json",
                 "validation_units_sup.csv": "validation_jlh32_probeA_sup.csv", "validation_units.csv": "validation_jlh32_probeA.csv"}

def R(name: str) -> Path:
    """Path of a per-probe result file."""
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / (_LEGACY_NAMES.get(name, name) if LEGACY else name)

def F(name: str) -> Path:
    """Figure path; new probes get the tag as a prefix so artifact names are unique."""
    FIG.mkdir(parents=True, exist_ok=True)
    return FIG / (name if LEGACY else f"{TAG}_{name}")

def describe() -> str:
    return f"{TAG} (root={ROOT}, stream={STREAM}, channels 0-{NCH-1}, out={OUT})"

# ---- stimulus conditions -----------------------------------------------------
COND_COLS = ("amplitude", "polarity", "pulse_duration", "contact_negative", "contact_positive")

def cond_labels(tr):
    """Condition label per trial: amplitude/polarity/duration/c-<neg>+<pos>."""
    return (tr.amplitude.astype(int).astype(str) + "/" + tr.polarity.astype(str) + "/" + tr.pulse_duration.astype(int).astype(str)
            + "/c" + tr.contact_negative.astype(int).astype(str) + "-" + tr.contact_positive.astype(int).astype(str)).to_numpy()

def cond_keys(tr):
    """Hashable key per trial for the ArtifactSuppressor (the stim controller's view)."""
    return list(zip(tr.amplitude.astype(int), tr.polarity.astype(str), tr.pulse_duration.astype(int),
                    tr.contact_negative.astype(int), tr.contact_positive.astype(int)))

# ---- multi-probe (recording-level) helpers -----------------------------------
def probe_out(rec: str, probe: str) -> Path:
    return Path("results") if (rec == "jlh32" and probe == "A") else Path("results") / f"{rec}_probe{probe}"

def R_probe(rec: str, probe: str, name: str) -> Path:
    legacy = (rec == "jlh32" and probe == "A")
    return probe_out(rec, probe) / (_LEGACY_NAMES.get(name, name) if legacy else name)

def available_probes(rec: str):
    """Probes of a recording whose per-probe pipeline has finished (validation + geometry present)."""
    return [p for p in "ABC" if R_probe(rec, p, "online_spikes_all_trials_sup.npz").exists()
            and R_probe(rec, p, "geometry_comparison_summary.json").exists()]

POP_TAG = f"{REC}_population"
POP_OUT = Path("results") / POP_TAG
POP_FIG = Path("figures") / POP_TAG

def RP(name: str) -> Path:
    POP_OUT.mkdir(parents=True, exist_ok=True); return POP_OUT / name

def FP(name: str) -> Path:
    POP_FIG.mkdir(parents=True, exist_ok=True); return POP_FIG / f"{POP_TAG}_{name}"


def cat_units(parts, axis):
    """Concatenate per-probe binned arrays along the unit axis, trimming the bin axis to the shortest probe.
    Probe sample clocks differ slightly, so the same window in ms can give n or n-1 bins."""
    import numpy as _np
    tax = axis - 1; n = min(x.shape[tax] for x in parts)
    return _np.concatenate([_np.take(x, range(n), axis=tax) for x in parts], axis=axis)


def load_trials(sess):
    """Trial table with exact TTL onset/offset times: NWB trials reconciled with TTL edges when an NWB file
    is available (validation), otherwise TTL edges alone (live / fresh recordings)."""
    from streamsort.io import load_nwb_trials, reconcile_with_ttl, ttl_trials
    up, dn = sess.ttl_edges(TTL_STREAM, TTL_LINE, True), sess.ttl_edges(TTL_STREAM, TTL_LINE, False)
    if NWB and os.path.exists(NWB):
        return reconcile_with_ttl(load_nwb_trials(NWB), up, dn)["trials_out"]
    return ttl_trials(up, dn)
