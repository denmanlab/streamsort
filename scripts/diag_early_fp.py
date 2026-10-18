import numpy as np, pandas as pd
on = dict(np.load("results/online_spikes_all_trials.npz")); rf = dict(np.load("results/reference_spikes_all_trials.npz"))
fs = float(on["meta_fs"]); off = on["meta_pulse_offset"]
tr = pd.read_csv("results/jlh32_probeA_trials.csv")
units = np.unique(rf["ks4_to_nwb"][:, 0])
m = np.isin(on["unit"], units)
of, ou, ot, osc = on["frame"][m], on["unit"][m], on["trial"][m], on["score"][m]
TOL = 12
# TP flag per online spike
tp = np.zeros(len(of), bool)
for u in units:
    a = np.flatnonzero(ou == u); r = np.sort(rf["frame"][rf["unit"] == u])
    if len(r) == 0 or len(a) == 0: continue
    i = np.clip(np.searchsorted(r, of[a]), 1, len(r) - 1)
    d = np.minimum(np.abs(of[a] - r[i - 1]), np.abs(of[a] - r[i]))
    tp[a] = d <= TOL
lat = (of - off[ot]) / fs * 1e3; rlat = (rf["frame"] - off[rf["trial"]]) / fs * 1e3
cond = tr.amplitude.astype(str) + "/" + tr.polarity.str[:3] + "/" + tr.pulse_duration.astype(str)
c_on = cond.to_numpy()[ot]; c_rf = cond.to_numpy()[rf["trial"]]
ntr = cond.value_counts()
rows = []
for c in ntr.index:
    r = dict(cond=c, n=ntr[c])
    for lo, hi in [(1, 3), (3, 5), (5, 10), (50, 300)]:
        mo = (c_on == c) & (lat >= lo) & (lat < hi); mr = (c_rf == c) & (rlat >= lo) & (rlat < hi)
        w = (hi - lo) / 1e3 * ntr[c] * len(units)
        r[f"on{lo}"] = mo.sum() / w; r[f"rf{lo}"] = mr.sum() / w; r[f"P{lo}"] = tp[mo].mean() if mo.any() else np.nan
    rows.append(r)
df = pd.DataFrame(rows).sort_values("cond"); pd.set_option("display.width", 250)
print("rates in Hz per unit;  P = precision")
print(df.round(2).to_string(index=False))
# concentration of 1-3 ms FPs across units and their score vs TPs
e = (lat >= 1) & (lat < 3); fp = e & ~tp
cu = pd.Series(ou[fp]).value_counts()
print("\n1-3 ms FP:", fp.sum(), "TP:", (e & tp).sum(), "| units with FPs:", len(cu), "| top-5 units share:", round(cu.iloc[:5].sum() / fp.sum(), 2))
print("top units:", cu.iloc[:8].to_dict())
print("score median FP/TP early:", np.median(osc[fp]), np.median(osc[e & tp]), "| late TP:", np.median(osc[(lat > 50) & tp]))
# latency histogram fine (0.1 ms) of FPs in 1-5 ms
h, b = np.histogram(lat[(lat >= 1) & (lat < 5) & ~tp], np.arange(1, 5.01, 0.25)); print("FP lat hist 0.25ms:", h.tolist())
h, b = np.histogram(rlat[(rlat >= 1) & (rlat < 5)], np.arange(1, 5.01, 0.25)); print("REF lat hist     :", h.tolist())
