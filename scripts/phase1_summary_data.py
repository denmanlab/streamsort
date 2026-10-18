"""Data for the Phase 1 summary figure: a filtered, artifact-suppressed raw snippet after a 50 uA pulse
(jlh32 probe A) with the online template matches the decoder made on it.

The snippet is captured from inside the real OnlineDecoder (after suppression, causal highpass and
median reference, before whitening), so the traces are exactly what the matcher saw. The suppressor's
per-condition artifact template is built causally from the preceding same-condition trials first.
  REC=jlh32 PROBE=A python scripts/phase1_summary_data.py      -> results/phase1_summary/snippet.npz
"""
import os, sys, dataclasses, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx
from streamsort.config import StreamConfig
from streamsort.io import OpenEphysSession, FileStreamSource
from streamsort.sorting.templates import TemplateBank
from streamsort.online import OnlineDecoder, DecoderParams, Whitening, decode_window
from streamsort.online import decoder as decmod
from streamsort.preprocess import ArtifactSuppressor

COND = os.environ.get("SNIP_COND", "-50/bipolar/100/c3-14"); PRE_MS, POST_MS = 3.0, 45.0; N_WARM = 20

st = OpenEphysSession(ctx.REC_DIR).stream(ctx.STREAM)
cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts), detect_threshold=4.0, n_channels_used=ctx.NCH)
src = FileStreamSource(st, ctx.NCH); bank = TemplateBank.load(ctx.R("templates.npz"))
usable = pd.read_csv(ctx.R("sorting_report.csv")).usable.to_numpy(bool); wh = Whitening.load(ctx.R("whitening.npz"))
sup = ArtifactSuppressor(cfg.sample_rate, ctx.NCH)
dec = OnlineDecoder(bank, cfg, wh, DecoderParams(min_energy=50.0), units_mask=usable, check=False, suppressor=sup)
tr = pd.read_csv(ctx.R("trials.csv")); lab = ctx.cond_labels(tr); keys = ctx.cond_keys(tr)
on = st.time_to_sample(tr.ttl_onset.to_numpy()); off = st.time_to_sample(tr.ttl_offset.to_numpy())
S = cfg.samples
idx = np.flatnonzero(lab == COND); assert len(idx) > N_WARM + 5, (COND, len(idx))
# representative trial: median online spike count 2-30 ms after offset, among trials after the warm-up
V = np.load(ctx.R("online_spikes_all_trials_sup.npz")); vt, vf = V["trial"], V["frame"]
cand = idx[N_WARM:]
cnt = np.array([np.sum((vt == k) & (vf >= off[k] + S(2)) & (vf < off[k] + S(30))) for k in cand])
k = int(cand[np.argsort(cnt)[len(cnt) // 2]])
# causal warm-up of the per-condition artifact template on the preceding same-condition trials
prev = idx[idx < k][-N_WARM:]
for j in prev:
    decode_window(dec, src, int(on[j]) - S(PRE_MS), S(PRE_MS + POST_MS), condition=keys[j])
# capture the decoder's filtered signal (pre-whitening) for the target trial
cap = []
orig_mr = decmod.kernels.median_reference
def mr(y):
    out = orig_mr(y); cap.append(np.array(out, copy=True)); return out
decmod.kernels.median_reference = mr
orig_proc = dec.pre.process
dec.pre.process = lambda x, scale=True: (lambda y: (cap.append(y.copy()), y)[1])(orig_proc(x, scale=scale))
start = int(on[k]) - S(PRE_MS); n = S(PRE_MS + POST_MS)
b = decode_window(dec, src, start, n, condition=keys[k])
decmod.kernels.median_reference = orig_mr; dec.pre.process = orig_proc
Y = np.concatenate(cap); W = cfg.samples(15.0); s0 = max(start - W, 0)
Y = Y[start - s0: start - s0 + n]                   # (n, C) uV, sample 0 = start
ev = [e for e in sup.events if start - S(1) <= e.get("onset", -1) <= start + S(PRE_MS + 2)]
fr, un = np.asarray(b.frame), np.asarray(b.unit); m = un >= 0
# agreement with the stored validation decode of this trial (pull mode, window from +1 ms)
sel_v = (vt == k) & (vf < start + n - S(5))
stored = set(zip(vf[sel_v].tolist(), V["unit"][sel_v].tolist()))
here = {(f, u) for f, u in zip(fr[m].tolist(), un[m].tolist()) if off[k] + S(1) <= f < start + n - S(5)}
both = here & stored
np.savez_compressed("results/phase1_summary/snippet.npz", Y=Y.astype(np.float32), start=start, fs=cfg.sample_rate, onset=int(on[k]), offset=int(off[k]),
                    frame=fr[m], unit=un[m], trial=k, cond=COND, n_warm=len(prev), sup_events=json.dumps(ev, default=str),
                    jaccard_vs_validation=len(both) / max(len(here | stored), 1))
print(f"trial {k} ({COND}, run {int(tr.run[k])}), spikes in window {int(m.sum())}, warm-up {len(prev)} trials; "
      f"suppressor events {ev}; Jaccard vs stored validation decode (>= +1 ms) {len(both) / max(len(here | stored), 1):.3f}")
