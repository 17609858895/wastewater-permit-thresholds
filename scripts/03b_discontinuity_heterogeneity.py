"""E1b. Heterogeneity of the near-limit discontinuity (descriptive): strict just-below / just-above ratio
(reports exactly at the limit excluded) relative to the same ratio at the 0.8 L placebo cut, by plant size,
major/minor, sampling frequency, sample type and fiscal year, for the seven conventional parameters (no TRC)."""
from pathlib import Path
import numpy as np
import pandas as pd
import re
PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E1_discontinuity"
pa = pd.read_parquet(PROJ / "data" / "permit_attributes.parquet", columns=["permit", "major", "design_flow"])
def fpm(c):
    m = re.match(r"^(\d+)/(\d+)$", str(c))
    return int(m.group(1)) * 30.0 / int(m.group(2)) if m and int(m.group(2)) > 0 else np.nan
parts = []
for p in ["BOD5", "CBOD5", "TSS", "NH3N", "TP", "FecalColi", "Ecoli"]:
    d = pd.read_parquet(PROJ / "data" / "analysis_sample.parquet", columns=["permit", "param", "vtype", "r", "nd", "fy", "lim_freq", "lim_sample"],
                        filters=[("param", "==", p)])
    d = d[d.nd == 0]
    r = d.r.values
    at = np.isclose(r, 1.0, rtol=1e-6)
    d = pd.DataFrame({"permit": d.permit.values, "param": p, "vtype": d.vtype.astype(str).values, "fy": d.fy.values,
                      "freq": d.lim_freq.astype(str).map(fpm).values, "stype": d.lim_sample.astype(str).str[:2].values,
                      "below": ((r > 0.9) & (r < 1.0) & ~at), "above": ((r > 1.0) & (r <= 1.1) & ~at),
                      "pb": ((r > 0.72) & (r <= 0.8)), "pa": ((r > 0.8) & (r <= 0.88)), "at": at})
    parts.append(d)
d = pd.concat(parts, ignore_index=True)
d = d.merge(pa, on="permit", how="left")
d["size"] = pd.cut(d.design_flow, [-0.1, 0.1, 1, 10, 1e6], labels=["<0.1 MGD", "0.1-1", "1-10", ">10"]).astype(str)
d["freq_cls"] = pd.cut(d.freq, [0, 1.01, 4.5, 12.5, 100], labels=["<=1/mo", "2-4/mo", "5-12/mo", ">12/mo"]).astype(str)
d["era"] = np.where(d.fy <= 2016, "FY2009-16", "FY2017-26")
def summ(g):
    b, a, pb, pa_, at = g.below.sum(), g.above.sum(), g.pb.sum(), g.pa.sum(), g["at"].sum()
    return pd.Series(dict(n=len(g), below=b, above=a, JR_strict=b / max(a, 1), JR_placebo=pb / max(pa_, 1),
                          excess_ratio=(b / max(a, 1)) / (pb / max(pa_, 1)), at_limit_share=at / len(g)))
out = {}
for by in ["size", "major", "freq_cls", "stype", "era", ["param", "era"], ["param", "size"], "fy"]:
    name = by if isinstance(by, str) else "_".join(by)
    t = d.groupby(by, observed=True)[["below", "above", "pb", "pa", "at"]].sum()
    n = d.groupby(by, observed=True).size().rename("n")
    t = t.join(n)
    t["JR_strict"] = t.below / t.above.clip(lower=1)
    t["JR_placebo"] = t.pb / t.pa.clip(lower=1)
    t["excess_ratio"] = t.JR_strict / t.JR_placebo
    t["at_limit_share"] = t["at"] / t.n
    t.to_csv(RES / f"heterogeneity_by_{name}.csv")
    print("\n==", name); print(t[["n", "JR_strict", "JR_placebo", "excess_ratio", "at_limit_share"]].round(3).to_string())
