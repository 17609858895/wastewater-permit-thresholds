"""E6u-TP2. Phosphorus limit levels over time, near-limit report distributions and exact-limit reporting.

Same TP C2/C3 modelling populations and limit groups (< 0.5, 0.5-1, 1-2, >= 2 mg/L) as 21u_tp_limit_levels.py:
  by_fy     share of reports in each limit group per fiscal year (October-September) and the median limit
  hist      detected reports with 0.5 < v/L <= 1.5 in bins of 0.05 (right-closed), as shares of that range,
            with values exactly at the limit counted separately
  exact     share of detected reports with 0.9 < v/L <= 1 that equal the limit (300 permit-bootstrap resamples), and the
            just-below/just-above ratio (1 < v/L <= 1.1) with and without values exactly at the limit
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
U = PROJ / "results" / "E2u_unified"
OUT = PROJ / "results" / "E6u_aggregate"
BINS, LABS = [0, .5, 1.0, 2.0, np.inf], ["<0.5", "0.5-1", "1-2", ">=2"]
EDGES = np.round(np.arange(.5, 1.5 + 1e-9, .05), 2)
rng = np.random.default_rng(0)
fy_rows, hist_rows, ex_rows = [], [], []
for st in ["TP_C2", "TP_C3"]:
    d = pd.read_parquet(U / f"{st}_k0.8_lam2.0_pred.parquet", columns=["permit", "period_end", "v", "L", "r", "nd"])
    d["lbin"] = pd.cut(d.L, BINS, right=False, labels=LABS)
    d["fy"] = d.period_end.dt.year + (d.period_end.dt.month >= 10)
    c = d.groupby(["fy", "lbin"], observed=True).size().unstack(fill_value=0)
    for fy, row in c.iterrows():
        for lb in LABS:
            fy_rows.append(dict(stratum=st, fy=fy, limit_bin=lb, n=int(row.get(lb, 0)), share=row.get(lb, 0) / row.sum(),
                                median_limit=d.L[d.fy == fy].median()))
    det = d[d.nd == 0]
    for lb, g in det.groupby("lbin", observed=True):
        x = g[(g.r > .5) & (g.r <= 1.5)]
        h = pd.cut(x.r, EDGES, right=True)
        n = x.groupby(h, observed=False).size().values
        ne = x[x.r == 1].groupby(h[x.r == 1], observed=False).size().values
        for lo, hi, k, ke in zip(EDGES[:-1], EDGES[1:], n, ne):
            hist_rows.append(dict(stratum=st, limit_bin=lb, bin_lo=lo, bin_hi=hi, n=int(k), n_exact=int(ke),
                                  share=k / len(x), share_exact=ke / len(x), n_range=len(x)))
        b = g[(g.r > .9) & (g.r <= 1)].assign(exact=lambda z: (z.r == 1).astype(float))
        t = b.groupby("permit").exact.agg(["sum", "size"])
        idx = rng.integers(0, len(t), (300, len(t)))
        bs = t["sum"].values[idx].sum(1) / t["size"].values[idx].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        na = int(((g.r > 1) & (g.r <= 1.1)).sum())
        ex_rows.append(dict(stratum=st, limit_bin=lb, n_below=len(b), n_exact=int(b.exact.sum()), n_above=na, permits=len(t),
                            exact_share=b.exact.mean(), ci_lo=lo, ci_hi=hi, ratio_all=len(b) / max(na, 1),
                            ratio_strict=(len(b) - b.exact.sum()) / max(na, 1)))
pd.DataFrame(fy_rows).to_csv(OUT / "tp_limit_level_by_fy.csv", index=False)
pd.DataFrame(hist_rows).to_csv(OUT / "tp_near_limit_hist.csv", index=False)
E = pd.DataFrame(ex_rows)
E.to_csv(OUT / "tp_exact_limit_share.csv", index=False)
F = pd.DataFrame(fy_rows)
print(F[(F.stratum == "TP_C2")].pivot(index="fy", columns="limit_bin", values="share").round(3).to_string())
print(F[F.stratum == "TP_C2"].groupby("fy").median_limit.first().to_string())
print(E.round(3).to_string())
