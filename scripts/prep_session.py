"""Stage 0: reconcile NWB trials with TTL edges; measure the artifact on this probe.
Outputs: trials.csv, session.json (in ctx.OUT)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx
from streamsort.io import OpenEphysSession, FileStreamSource, load_nwb_trials, load_nwb_units, reconcile_with_ttl
from streamsort.preprocess.artifact import detect_artifact_onset, clipped_span

def main():
    sess = OpenEphysSession(ctx.REC_DIR); st = sess.stream(ctx.STREAM); src = FileStreamSource(st, ctx.NCH)
    rec = reconcile_with_ttl(load_nwb_trials(ctx.NWB), sess.ttl_edges(ctx.TTL_STREAM, ctx.TTL_LINE, True),
                             sess.ttl_edges(ctx.TTL_STREAM, ctx.TTL_LINE, False))
    tr = rec["trials_out"].reset_index(drop=True)
    keep = ["run", "ttl_onset", "ttl_offset", "ttl_width_us", "amplitude", "polarity", "pulse_duration",
            "contact_negative", "contact_positive", "start_time", "stop_time"]
    tr = tr[[c for c in keep if c in tr.columns]].copy(); tr.insert(0, "trial", np.arange(len(tr)))
    # In jlh34 the TTL is a ~1 s gate, not the stimulus pulse.  Define the pulse offset as
    # onset + pulse duration (the jlh32 convention, where the TTL width equals the duration).
    wide = tr.ttl_width_us > 10_000
    ttl_offset_note = "ttl_offset = TTL falling edge"
    if wide.any():
        tr.loc[wide, "ttl_offset"] = tr.ttl_onset[wide] + tr.pulse_duration[wide] * 1e-6
        ttl_offset_note = f"{int(wide.sum())} trials had TTL width > 10 ms (median {tr.ttl_width_us[wide].median():.0f} us); ttl_offset reset to ttl_onset + pulse_duration"
    tr["cond"] = ctx.cond_labels(tr)
    tr.to_csv(ctx.R("trials.csv"), index=False)
    fs = st.effective_rate; S = lambda ms: int(round(ms * 1e-3 * fs))
    # artifact on this probe: clipped channels and span for a sample of trials
    rows = []
    for k in np.linspace(0, len(tr) - 1, 120).astype(int):
        on = int(st.time_to_sample(np.array([tr.ttl_onset[k]]))[0])
        raw = np.asarray(src.read(on - S(5), S(20)))
        o = detect_artifact_onset(raw, S(5), S(2.0))
        nclip = int((np.abs(raw) >= 6190).any(0).sum())
        rows.append(dict(trial=int(k), amp=int(tr.amplitude[k]), onset_found=o is not None, n_clipped_ch=nclip,
                         span=(clipped_span(raw, o, S(5)) if o is not None else 0),
                         ptp_uV_max=float(np.ptp(raw[S(5):S(8)].astype(float), 0).max() * st.bit_volts)))
    A = pd.DataFrame(rows)
    U = load_nwb_units(ctx.NWB, probe=ctx.PROBE)
    info = dict(tag=ctx.TAG, root=ctx.ROOT, stream=ctx.STREAM, n_channels_used=ctx.NCH,
                effective_rate=float(fs), t_start=float(st.sample_to_time(0)), n_samples=int(st.n_samples),
                duration_s=float(st.n_samples / fs), n_trials=int(len(tr)), n_runs=int(tr.run.nunique()),
                n_conditions=int(tr.groupby(["amplitude", "polarity", "pulse_duration"]).ngroups),
                ttl_offset_note=ttl_offset_note, n_conditions_with_contacts=int(tr.cond.nunique()),
                first_pulse_s=float(tr.ttl_onset.min() - st.sample_to_time(0)),
                reconcile={k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in rec.items() if k != "trials_out"},
                artifact=dict(frac_onset_detected=float(A.onset_found.mean()), median_clipped_ch=float(A.n_clipped_ch.median()),
                              max_clipped_ch=int(A.n_clipped_ch.max()), median_span=float(A.span.median()),
                              median_ptp_uV_max=float(A.ptp_uV_max.median())),
                ref_units=dict(n=int(len(U)), good=int((U.group == "good").sum()), mua=int((U.group == "mua").sum()),
                               max_ch=int(U.ch.max()), n_ch_ge_nch=int((U.ch >= ctx.NCH).sum())))
    json.dump(info, open(ctx.R("session.json"), "w"), indent=2, default=str)
    print(json.dumps({k: info[k] for k in ["tag", "n_trials", "n_runs", "n_conditions", "first_pulse_s", "artifact", "ref_units"]}, default=str))

if __name__ == "__main__":
    main()
