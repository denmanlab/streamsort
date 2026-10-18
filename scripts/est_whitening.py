"""Stage: local whitening (k=32 neighbours) from a stimulus-free stretch (60-80 s after start)."""
import os, sys, json, dataclasses, time, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx
from streamsort.config import StreamConfig
from streamsort.io import OpenEphysSession, FileStreamSource
from streamsort.sorting.templates import TemplateBank
from streamsort.online import estimate_whitening

def main():
    st = OpenEphysSession(ctx.REC_DIR).stream(ctx.STREAM)
    cfg = dataclasses.replace(StreamConfig(sample_rate=st.effective_rate, bit_volts=st.bit_volts),
                              n_channels_used=ctx.NCH, detect_threshold=4.0)
    src = FileStreamSource(st, ctx.NCH)
    bank = TemplateBank.load(ctx.R("templates.npz"))
    pos = bank.channel_locations.astype(float)
    if ctx.R("session.json").exists():
        first = json.load(open(ctx.R("session.json")))["first_pulse_s"]
    else:                                  # no prep stage (fresh recording): first stim TTL relative to stream start
        sess = OpenEphysSession(ctx.REC_DIR); first = float(ctx.load_trials(sess).ttl_onset.min() - st.t_start)
        first = first if first == first else float("inf")    # no stimulation in the segment
    s0, dur = 60.0, 20.0
    assert s0 + dur < first - 1, f"whitening window overlaps stimulation (first pulse {first:.1f} s)"
    t = time.perf_counter()
    wh = estimate_whitening(src, cfg, pos, start=cfg.samples(s0 * 1e3), n=cfg.samples(dur * 1e3), k=32)
    wh.save(ctx.R("whitening.npz"))
    print(f"[whiten] {ctx.TAG}: {time.perf_counter()-t:.1f} s, noise sd median {np.median(wh.noise_sd):.3f}", flush=True)

if __name__ == "__main__":
    main()
