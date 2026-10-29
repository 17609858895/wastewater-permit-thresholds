"""E4. Limit-change natural experiment: does the near-limit cliff move with the limit?

Within a series (permit, outfall, parameter, statistic), find changes of the enforceable limit by >= 20%
with >= 12 detected reports on each side. For every report we measure the share falling just below and
just above both the old and the new limit (w = 10%), by event time. If the discontinuity is limit-induced it
must appear at whichever limit is active and vanish at the inactive one.
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E4_limit_change"
RES.mkdir(parents=True, exist_ok=True)
KEY = ["permit", "outfall", "param", "statbase", "vtype"]

if __name__ == "__main__":
    import sys
    NOTRC = "notrc" in sys.argv
    SUF = "_noTRC" if NOTRC else ""
    d = pd.read_parquet(PROJ / "data" / "analysis_sample.parquet",
                        columns=KEY + ["period_end", "v", "L", "nd"],
                        filters=[("param", "!=", "TRC")] if NOTRC else None)
    for c in ["param", "statbase", "vtype", "outfall"]:
        d[c] = d[c].astype("category")
    d = d[d.nd == 0].sort_values(KEY + ["period_end"]).reset_index(drop=True)
    d["ser"] = d.groupby(KEY, sort=False, observed=True).ngroup()
    d["Lprev"] = d.groupby("ser").L.shift(1)
    chg = d[(d.Lprev.notna()) & (np.abs(np.log(d.L / d.Lprev)) >= np.log(1.2))]
    # keep the first change per series and require support on both sides
    chg = chg.groupby("ser").head(1)
    d = d.merge(chg[["ser", "period_end", "L", "Lprev"]].rename(columns={"period_end": "t0", "L": "L_new", "Lprev": "L_old"}),
                on="ser", how="inner")
    d["et"] = ((d.period_end - d.t0).dt.days / 30.44).round().astype(int)
    d = d[(d.et >= -36) & (d.et <= 36)]
    sup = d.groupby("ser").agg(pre=("et", lambda x: (x < 0).sum()), post=("et", lambda x: (x >= 0).sum()))
    ok = sup[(sup.pre >= 12) & (sup.post >= 12)].index
    d = d[d.ser.isin(ok)].copy()
    d["tight"] = d.L_new < d.L_old
    # is the report's limit the one that applies at that time?
    for lim in ["L_old", "L_new"]:
        r = d.v / d[lim]
        d[f"below_{lim}"] = ((r > 0.9) & (r <= 1.0)).astype(float)
        d[f"above_{lim}"] = ((r > 1.0) & (r <= 1.1)).astype(float)
    d["post"] = (d.et >= 0).astype(int)
    summ = d.groupby(["tight", "post"]).agg(
        n=("v", "size"), series=("ser", "nunique"),
        below_old=("below_L_old", "mean"), above_old=("above_L_old", "mean"),
        below_new=("below_L_new", "mean"), above_new=("above_L_new", "mean"))
    summ["JR_old"] = summ.below_old / summ.above_old
    summ["JR_new"] = summ.below_new / summ.above_new
    summ.to_csv(RES / f"limit_change_summary{SUF}.csv")
    print(summ.round(4).to_string())
    prof = d.groupby(["tight", "et"]).agg(n=("v", "size"), below_old=("below_L_old", "mean"), above_old=("above_L_old", "mean"),
                                           below_new=("below_L_new", "mean"), above_new=("above_L_new", "mean")).reset_index()
    prof.to_csv(RES / f"limit_change_event_profile{SUF}.csv", index=False)
    # series-level difference-in-differences of the cliff (below - above) at new vs old limit
    s = d.groupby(["ser", "tight", "post"]).agg(cn=("below_L_new", "mean"), an=("above_L_new", "mean"),
                                                  co=("below_L_old", "mean"), ao=("above_L_old", "mean")).reset_index()
    s["cliff_new"] = s.cn - s.an
    s["cliff_old"] = s.co - s.ao
    w = s.pivot_table(index=["ser", "tight"], columns="post", values=["cliff_new", "cliff_old"]).dropna()
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["did"] = (w.cliff_new_1 - w.cliff_new_0) - (w.cliff_old_1 - w.cliff_old_0)
    rng = np.random.default_rng(0)
    rows = []
    for t, g in w.groupby("tight"):
        bs = [g.did.values[rng.integers(0, len(g), len(g))].mean() for _ in range(1000)]
        rows.append(dict(tight=t, series=len(g), did=g.did.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                         cliff_new_pre=g.cliff_new_0.mean(), cliff_new_post=g.cliff_new_1.mean(),
                         cliff_old_pre=g.cliff_old_0.mean(), cliff_old_post=g.cliff_old_1.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"limit_change_did{SUF}.csv", index=False)
    print(out.round(4).to_string())

    # --- real performance vs reshaping: lower quantiles of log(v / L_old) by event time
    d["lr"] = np.log(np.maximum(d.v, 1e-6) / d.L_old)
    q = d.groupby(["ser", "tight", "post"]).lr.agg(q25=lambda x: x.quantile(0.25), q50="median",
                                                    q75=lambda x: x.quantile(0.75)).reset_index()
    w2 = q.pivot_table(index=["ser", "tight"], columns="post", values=["q25", "q50", "q75"]).dropna()
    w2.columns = [f"{a}_{b}" for a, b in w2.columns]
    w2 = w2.reset_index()
    rows = []
    for t, g in w2.groupby("tight"):
        rec = dict(tight=t, series=len(g), ratio_new_old=float(np.exp(np.log(d[d.tight == t].L_new / d[d.tight == t].L_old).median())))
        for qq in ["q25", "q50", "q75"]:
            delta = g[f"{qq}_1"] - g[f"{qq}_0"]
            bs = [delta.values[rng.integers(0, len(g), len(g))].mean() for _ in range(1000)]
            rec[f"d{qq}_pct"] = float(100 * (np.exp(delta.mean()) - 1))
            rec[f"d{qq}_lo"] = float(100 * (np.exp(np.percentile(bs, 2.5)) - 1))
            rec[f"d{qq}_hi"] = float(100 * (np.exp(np.percentile(bs, 97.5)) - 1))
        rows.append(rec)
    out2 = pd.DataFrame(rows)
    out2.to_csv(RES / f"limit_change_quantile_shift{SUF}.csv", index=False)
    print(out2.round(3).to_string())
    prof2 = d.groupby(["tight", "et"]).lr.agg(q25=lambda x: x.quantile(0.25), q50="median", q75=lambda x: x.quantile(0.75)).reset_index()
    prof2.to_csv(RES / f"limit_change_quantile_profile{SUF}.csv", index=False)
    # by parameter: DiD of the cliff
    par = d.groupby("ser").param.first().astype(str)
    w["param"] = w.ser.map(par)
    rows = []
    for (t, pname), g in w.groupby(["tight", "param"]):
        if len(g) < 30:
            continue
        bs = [g.did.values[rng.integers(0, len(g), len(g))].mean() for _ in range(1000)]
        rows.append(dict(tight=t, param=pname, series=len(g), did=g.did.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    out3 = pd.DataFrame(rows)
    out3.to_csv(RES / f"limit_change_did_by_param{SUF}.csv", index=False)
    print(out3.round(4).to_string())
