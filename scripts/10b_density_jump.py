"""E7b. Discrete McCrary-type density jump at the limit, model-free.

For each report on its own reporting grid (unit u = last reported decimal), k = (v - c) / u. Counts N(k) for
k = -K..-1 and 1..K are taken over values that are NOT multiples of 5u (removes round-number heaping). A Poisson
log-linear trend is fitted separately on each side and extrapolated to k = 0; theta = exp(a_left - a_right) is the
density jump at c. Under any smooth latent density (process control included) theta = 1. Centres: L and placebo
0.6 L, 0.8 L, 1.25 L (rounded to the report grid). Permit-cluster bootstrap CI.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E7_adjacent_value"
K = 10


def unit(vstr, v):
    s = pd.Series(vstr).astype(str)
    dec = s.str.extract(r"\.(\d+)$")[0].str.len().fillna(0).astype(float).to_numpy()
    vraw = pd.to_numeric(pd.Series(vstr), errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where((vraw > 0) & np.isfinite(vraw), v / vraw, 1.0)
    return 10.0 ** (-dec) * scale


def theta_from_counts(cnt):
    """cnt: array of length 2K (k=-K..-1, 1..K) of counts; returns log jump."""
    ks = np.r_[np.arange(-K, 0), np.arange(1, K + 1)]
    ok = cnt > 0
    left = (ks < 0) & ok
    right = (ks > 0) & ok
    if left.sum() < 3 or right.sum() < 3:
        return np.nan
    def fit(mask):
        X = sm.add_constant(ks[mask].astype(float))
        return sm.GLM(cnt[mask], X, family=sm.families.Poisson()).fit().params[0]
    return fit(left) - fit(right)


rows = []
rng = np.random.default_rng(0)
for p in ["BOD5", "CBOD5", "TSS", "NH3N", "TP", "FecalColi", "Ecoli", "TRC"]:
    d = pd.read_parquet(PROJ / "data" / "analysis_sample.parquet", columns=["permit", "vtype", "v", "L", "nd", "v_raw_str"],
                        filters=[("param", "==", p)])
    d = d[(d.nd == 0) & (d.v > 0)]
    v, L = d.v.to_numpy(float), d.L.to_numpy(float)
    u = unit(d.v_raw_str.values, v)
    heap = np.abs(v / (5 * u) - np.round(v / (5 * u))) < 1e-4
    perm = d.permit.astype(str).to_numpy()
    vt = d.vtype.astype(str).to_numpy()
    for cname, c in [("limit", L), ("placebo0.6", np.round(0.6 * L / u) * u), ("placebo0.8", np.round(0.8 * L / u) * u),
                     ("placebo1.25", np.round(1.25 * L / u) * u)]:
        k = (v - c) / u
        kr = np.round(k).astype(np.int64)
        m = (np.abs(k - kr) < 1e-4) & (np.abs(kr) <= K) & (kr != 0) & ~heap
        for vtype in ["C2", "C3"]:
            mm = m & (vt == vtype)
            if mm.sum() < 200:
                continue
            sub = pd.DataFrame({"permit": perm[mm], "k": kr[mm]})
            tab = sub.groupby(["permit", "k"]).size().unstack(fill_value=0)
            cols = [kk for kk in list(range(-K, 0)) + list(range(1, K + 1))]
            tab = tab.reindex(columns=cols, fill_value=0)
            arr = tab.to_numpy()
            lt = theta_from_counts(arr.sum(0))
            bs = []
            for _ in range(200):
                idx = rng.integers(0, len(arr), len(arr))
                bs.append(theta_from_counts(arr[idx].sum(0)))
            bs = np.array(bs)
            rows.append(dict(param=p, vtype=vtype, centre=cname, n=int(mm.sum()), theta=float(np.exp(lt)),
                             lo=float(np.exp(np.nanpercentile(bs, 2.5))), hi=float(np.exp(np.nanpercentile(bs, 97.5))),
                             n_minus1=int(arr[:, K - 1].sum()), n_plus1=int(arr[:, K].sum())))
    print(p, flush=True)
t = pd.DataFrame(rows)
t.to_csv(RES / "density_jump_theta.csv", index=False)
print(t.pivot_table(index=["param", "vtype"], columns="centre", values="theta").round(2).to_string())
print(t[t.centre == "limit"][["param", "vtype", "n", "theta", "lo", "hi", "n_minus1", "n_plus1"]].round(3).to_string())
