"""Unit and spike capture of the online decoder vs the offline KS2.5 sort, pooled over a recording's probes.
Units: active good offline units matched by the initial sort (gate, agreement > 0.5).
Spikes (held-out = trial windows after the 600 s sort segment, 1-1901 ms after pulse offset):
  per matched unit recall / precision (validation tables), and coverage = online-recovered spikes of good matched units
  / all spikes of ALL good offline units in the same windows (unmatched units count as missed).
  REC=jlh32 python scripts/phase1_summary_coverage.py   -> results/phase1_summary/coverage_<REC>.json
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import ctx
from streamsort.io import OpenEphysSession, load_nwb_units
rec = ctx.REC; out = dict(rec=rec, probes={})
for p in "ABC":
    R = lambda n: ctx.R_probe(rec, p, n)
    g = json.load(open(R("gate_summary.json"))); V = pd.read_csv(R("validation_units_sup.csv"))
    on = np.load(R("online_spikes_all_trials_sup.npz")); w0 = on["meta_window_start"].astype(np.int64); nW = int(on["meta_window_len"])
    st = OpenEphysSession(ctx.RECS[rec]["root"] + "/Record Node 105/experiment1/recording1").stream(f"Neuropix-PXI-104.Probe{p}-AP")
    seg_end = int(st.time_to_sample(np.array([st.t_start + 600.0]))[0]); ho = w0 >= seg_end
    U = load_nwb_units(ctx.RECS[rec]["root"] + "/" + ctx.RECS[rec]["nwb"], probe=p)
    idcol = "unit_id"
    ids = U[idcol].astype(int).to_numpy(); good = (U.group == "good").to_numpy()
    n_good_ho = 0; per = {}
    for uid, gd, stimes in zip(ids, good, U.spike_times):
        if not gd: continue
        s = np.sort(st.time_to_sample(np.asarray(stimes))); a = np.searchsorted(s, w0[ho]); b = np.searchsorted(s, w0[ho] + nW)
        per[int(uid)] = int((b - a).sum()); n_good_ho += per[int(uid)]
    goodV = V[V.nwb_unit.isin(list(per))]
    tp_good = int(goodV.tp_heldout.sum()); nref_good_matched = int(goodV.n_ref_heldout.sum())
    out["probes"][p] = dict(units_matched=g["good_active_matched_gt0.5"], units_active_good=g["n_ref_good_active"], n_good=int(good.sum()),
                            n_matched_validated=len(V), n_matched_good=len(goodV), spikes_good_heldout=n_good_ho,
                            spikes_good_matched_heldout=nref_good_matched, tp_good_heldout=tp_good,
                            recall=V.recall_heldout.tolist(), precision=V.precision_heldout.tolist(), n_ref_heldout=V.n_ref_heldout.tolist(),
                            check_nref_vs_nwb=float(nref_good_matched / max(sum(per[int(u)] for u in goodV.nwb_unit), 1)))
    q = out["probes"][p]; print(p, {k: v for k, v in q.items() if not isinstance(v, list)}, flush=True)
P = out["probes"].values()
out["pooled"] = dict(units_matched=sum(q["units_matched"] for q in P), units_active_good=sum(q["units_active_good"] for q in P),
                     coverage_all_good=sum(q["tp_good_heldout"] for q in P) / sum(q["spikes_good_heldout"] for q in P),
                     recall_matched_pooled=sum(q["tp_good_heldout"] for q in P) / sum(q["spikes_good_matched_heldout"] for q in P),
                     share_spikes_in_matched_units=sum(q["spikes_good_matched_heldout"] for q in P) / sum(q["spikes_good_heldout"] for q in P))
print(out["pooled"])
os.makedirs("results/phase1_summary", exist_ok=True); json.dump(out, open(f"results/phase1_summary/coverage_{rec}.json", "w"))
