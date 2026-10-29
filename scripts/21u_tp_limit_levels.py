"""E6u-TP. Phosphorus near-limit accumulation and reconstructed shortfall by permit-limit level.

For TP C2 and C3, reports are grouped by limit (< 0.5, 0.5-1, 1-2, >= 2 mg/L). For each group: the model-free
just-below/just-above ratio of detected reports (0.9 < v/L <= 1 versus 1 < v/L <= 1.1) and the reconstructed shortfall
of the national stack (same estimand as 05u: (E_window - O_window) / (E_window + O_above)), with 300 permit-bootstrap
resamples and fitted distributions fixed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, expit

PROJ = Path(__file__).resolve().parents[1]
U = PROJ / "results" / "E2u_unified"
OUT = PROJ / "results" / "E6u_aggregate" / "tp_by_limit_level.csv"
BINS, LABS = [0, .5, 1.0, 2.0, np.inf], ["<0.5", "0.5-1", "1-2", ">=2"]
rows = []
rng = np.random.default_rng(0)
for st in ["TP_C2", "TP_C3"]:
    d = pd.read_parquet(U / f"{st}_k0.8_lam2.0_pred.parquet", columns=["permit", "v", "L", "nd", "mu_n", "s_n", "mu_l", "s_l", "T", "O"])
    w = json.loads((U / f"{st}_k0.8_lam2.0.json").read_text())["w_normal"]
    F = lambda x: w * ndtr((x - d.mu_n) / np.exp(d.s_n)) + (1 - w) * expit((x - d.mu_l) / np.exp(d.s_l))
    keep = d.nd == 0
    r = d.v / d.L
    d = d.assign(E=F(np.log(2 * d.L)) - F(np.log(d["T"])), Ow=((d.O == 1) & (d.v < 2 * d.L)).astype(float),
                 Oa=(keep & (d.v >= 2 * d.L)).astype(float), below=(keep & (r > .9) & (r <= 1)).astype(float),
                 above=(keep & (r > 1) & (r <= 1.1)).astype(float), lbin=pd.cut(d.L, BINS, right=False, labels=LABS))
    for lb, g in d.groupby("lbin", observed=True):
        t = g.groupby("permit")[["E", "Ow", "Oa"]].sum()
        idx = rng.integers(0, len(t), (300, len(t)))
        Eb, Ob, Ab = t.E.values[idx].sum(1), t.Ow.values[idx].sum(1), t.Oa.values[idx].sum(1)
        share = (g.E.sum() - g.Ow.sum()) / (g.E.sum() + g.Oa.sum())
        lo, hi = np.percentile((Eb - Ob) / (Eb + Ab), [2.5, 97.5])
        rows.append(dict(stratum=st, limit_bin=lb, n=len(g), permits=g.permit.nunique(), median_limit=g.L.median(),
                         below_above_ratio=g.below.sum() / max(g.above.sum(), 1), missing_share=share, ci_lo=lo, ci_hi=hi,
                         reported_exc_rate=(g.Ow.sum() + g.Oa.sum()) / len(g), counterfactual_exc_rate=(g.E.sum() + g.Oa.sum()) / len(g)))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.round(3).to_string())
