"""E1. Near-limit discontinuity in self-reported effluent values (descriptive, no model).

For reports with an enforceable numeric upper limit L, r = v / L. The just-below / just-above
ratio JR(w) = N(r in (1-w, 1]) / N(r in (1, 1+w]) is compared with the same ratio at placebo
cut-points c (0.6, 0.7, 0.8 and 1.2, 1.3), where no regulatory incentive exists.
Windows are half-open on the same side so integer-valued reports contribute equally.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E1_discontinuity"
RES.mkdir(parents=True, exist_ok=True)


def analysis_sample(eff):
    m = (eff.vtype.isin(["C2", "C3"]) & eff.L_qual.isin(["<=", "<"]) & (eff.limit_type == "ENF")
         & (eff.stay.fillna("").str.strip() == "") & (eff.L > 0) & eff.v.notna() & (eff.v >= 0)
         & eff.qual.isin(["=", "<", "<="]) & (eff.nodi.fillna("").str.strip() == ""))
    d = eff.loc[m.fillna(False).astype(bool).values].copy()
    d["r"] = d.v / d.L
    d["nd"] = d.qual.isin(["<", "<="]).fillna(False).astype(np.int8)
    d["fy"] = (d.period_end.dt.year + (d.period_end.dt.month >= 10)).astype(int)
    return d


def jr(r, c, w):
    below = ((r > c * (1 - w)) & (r <= c)).sum()
    above = ((r > c) & (r <= c * (1 + w))).sum()
    return below, above, below / max(above, 1)


if __name__ == "__main__":
    import sys
    if not (PROJ / "data" / "analysis_sample.parquet").exists() or "--rebuild" in sys.argv:
        eff = pd.read_parquet(PROJ / "data" / "effluent_panel.parquet")
        d = analysis_sample(eff)
        d.to_parquet(PROJ / "data" / "analysis_sample.parquet", index=False)
        del eff
    d = pd.read_parquet(PROJ / "data" / "analysis_sample.parquet", columns=["permit", "param", "vtype", "r", "nd", "fy", "v", "L"])
    for c in ["param", "vtype"]:
        d[c] = d[c].astype("category")
    d["state"] = d.permit.str[:2].astype("category")
    d["permit"] = d.permit.astype("category")
    print("analysis rows", len(d), "permits", d.permit.nunique())
    rows = []
    det = d[d.nd == 0]
    del d
    for (p, vt), g in det.groupby(["param", "vtype"], observed=True):
        for c in [0.6, 0.7, 0.8, 1.0, 1.2, 1.3]:
            for w in [0.05, 0.10]:
                b, a, ratio = jr(g.r.values, c, w)
                rows.append(dict(param=p, vtype=vt, cut=c, w=w, below=b, above=a, JR=ratio, n=len(g),
                                 permits=g.permit.nunique(), share_over=(g.r > 1).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "jump_ratio_by_param.csv", index=False)
    ex = det.assign(at=(np.isclose(det.r, 1.0, rtol=1e-6)), strict=((det.r > 0.9) & (det.r < 1.0) & ~np.isclose(det.r, 1.0, rtol=1e-6)),
                    above=((det.r > 1.0) & (det.r <= 1.1) & ~np.isclose(det.r, 1.0, rtol=1e-6)))
    ex = ex.groupby(["param", "vtype"], observed=True)[["at", "strict", "above"]].sum()
    ex["JR_strict"] = ex.strict / ex.above
    ex["share_at_limit_of_below"] = ex["at"] / (ex["at"] + ex.strict)
    ex.to_csv(RES / "exact_at_limit_decomposition.csv")
    print(ex.round(3).to_string())
    print(t[(t.w == 0.10)].pivot_table(index=["param", "vtype"], columns="cut", values="JR").round(2).to_string())
    # by fiscal year (pooled conventional parameters, avg and max separately)
    rows = []
    for (fy, vt), g in det.groupby(["fy", "vtype"], observed=True):
        b, a, ratio = jr(g.r.values, 1.0, 0.10)
        b8, a8, r8 = jr(g.r.values, 0.8, 0.10)
        rows.append(dict(fy=fy, vtype=vt, below=b, above=a, JR=ratio, JR_placebo08=r8, n=len(g), share_over=(g.r > 1).mean()))
    pd.DataFrame(rows).to_csv(RES / "jump_ratio_by_year.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index="fy", columns="vtype", values=["JR", "JR_placebo08"]).round(2).to_string())
    # fine histogram for plotting
    bins = np.round(np.arange(0.50, 1.5001, 0.01), 4)
    hist = []
    for (p, vt), g in det.groupby(["param", "vtype"], observed=True):
        h, _ = np.histogram(g.r.values, bins=bins)
        hist.append(pd.DataFrame(dict(param=p, vtype=vt, lo=bins[:-1], hi=bins[1:], n=h)))
    pd.concat(hist).to_csv(RES / "ratio_histogram_0p01.csv", index=False)
    # by state
    rows = []
    for (st, vt), g in det.groupby(["state", "vtype"], observed=True):
        b, a, ratio = jr(g.r.values, 1.0, 0.10)
        rows.append(dict(state=st, vtype=vt, below=b, above=a, JR=ratio, n=len(g), permits=g.permit.nunique()))
    pd.DataFrame(rows).to_csv(RES / "jump_ratio_by_state.csv", index=False)
