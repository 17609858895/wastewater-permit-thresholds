"""Within-series profiles around old and new permit limits.

Series here include limit-set identifiers, unlike the model-free event analysis.
Observed counts are spread over rounding intervals and compared with the fitted
mixture. Treatment changes, reporting and conditional fit can affect these
profiles. This is a retrospective threshold diagnostic without unchanged-limit
controls, not a causal effect estimate. Intervals use 1,000 series resamples.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliffmix as cm  # noqa: E402

OUT = cm.PROJ / "results" / "E4u_cliff_event"; OUT.mkdir(parents=True, exist_ok=True)
EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0]
COLS = ["sid", "param", "vtype", "period_end", "v", "L", "T", "nd", "mu_n", "s_n", "mu_l", "s_l"]
B = 1000


def events(p):
    d = p[p.nd == 0].sort_values(["sid", "period_end"])
    prev = d.groupby("sid").L.shift(1)
    chg = d[prev.notna() & (np.abs(np.log(d.L / prev)) >= np.log(1.2))].assign(L_old=prev)
    chg = chg.groupby("sid").head(1)[["sid", "period_end", "L", "L_old"]].rename(columns={"period_end": "t0", "L": "L_new"})
    e = p.merge(chg, on="sid", how="inner")
    e["et"] = ((e.period_end - e.t0).dt.days / 30.44).round().astype(int)
    e = e[(e.et >= -36) & (e.et <= 36)]
    det = e[e.nd == 0]
    sup = det.groupby("sid").et.agg(pre=lambda x: (x < 0).sum(), post=lambda x: (x >= 0).sum())
    e = e[e.sid.isin(sup[(sup.pre >= 12) & (sup.post >= 12)].index)].copy()
    e["post"] = (e.et >= 0).astype(int)
    e["tight"] = e.L_new < e.L_old
    return e


if __name__ == "__main__":
    recs = []
    for st in cm.CONV:
        p, res = cm.load(st, columns=COLS)
        e = events(cm.with_h(p, st))
        if e.empty:
            continue
        keep = e.nd.values == 0
        for lim in ["L_old", "L_new"]:
            Lx = e[lim].values
            for a, b in zip(EDGES[:-1], EDGES[1:]):
                E = cm.F(e, res, np.log(b * Lx)) - cm.F(e, res, np.log(a * Lx))
                O = cm.frac(e.v.values, e.h.values, a * Lx, b * Lx)   # latent-scale counts (see 19u)
                g = pd.DataFrame({"sid": e.sid.values, "tight": e.tight.values, "post": e.post.values,
                                  "E": np.where(keep, E, 0.0), "O": np.where(keep, O, 0.0)})
                s = g.groupby(["sid", "tight", "post"]).agg(E=("E", "sum"), O=("O", "sum")).reset_index()
                s["stratum"], s["limit"], s["lo"], s["hi"] = st, lim, a, b
                recs.append(s)
        ratio = (e.groupby("sid").L_old.first() / e.groupby("sid").L_new.first())
        print(st, "series", e.sid.nunique(), "tight", int(e.groupby("sid").tight.first().sum()), flush=True)
        recs.append(pd.DataFrame({"sid": ratio.index, "ratio": ratio.values, "stratum": st, "limit": "ratio"}))
    R = pd.concat(recs, ignore_index=True)
    rat = R[R.limit == "ratio"][["stratum", "sid", "ratio"]]
    S = R[R.limit != "ratio"].drop(columns="ratio").merge(rat, on=["stratum", "sid"])
    S["param"] = S.stratum.str.split("_").str[0]
    S["sep"] = (S.ratio >= 1.5) | (S.ratio <= 1 / 1.5)
    S["group"] = np.where(S.param.isin(["TP", "NH3N", "Ecoli"]), "TP, NH3-N, E. coli", "BOD5, CBOD5, TSS, F. coli")
    S.to_parquet(OUT / "future_limit_series_bins.parquet", index=False)
    rng = np.random.default_rng(0)
    rows = []
    for sel, sub in [("separated", S[S.sep]), ("all", S)]:
        for keys, g in sub.groupby(["tight", "post", "limit", "lo", "hi"]):
            ser = g.groupby(["stratum", "sid"])[["O", "E"]].sum()
            Ov, Ev = ser.O.values, ser.E.values
            idx = rng.integers(0, len(ser), (B, len(ser)))
            bs = Ov[idx].sum(1) / Ev[idx].sum(1)
            rows.append(dict(sel=sel, tight=keys[0], post=keys[1], limit=keys[2], lo=keys[3], hi=keys[4], series=len(ser),
                             O=Ov.sum(), E=Ev.sum(), OE=Ov.sum() / Ev.sum(), ci_lo=np.percentile(bs, 2.5),
                             ci_hi=np.percentile(bs, 97.5)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "future_limit_profile.csv", index=False)
    # by parameter group, separated tightenings, the two bins next to each limit
    rows = []
    for (grp, post, lim, lo), g in S[S.sep & S.tight & S.lo.isin([0.9, 0.95, 1.0, 1.05])].groupby(["group", "post", "limit", "lo"]):
        ser = g.groupby(["stratum", "sid"])[["O", "E"]].sum()
        idx = rng.integers(0, len(ser), (B, len(ser)))
        bs = ser.O.values[idx].sum(1) / ser.E.values[idx].sum(1)
        rows.append(dict(group=grp, post=post, limit=lim, lo=lo, series=len(ser), OE=ser.O.sum() / ser.E.sum(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(rows).to_csv(OUT / "future_limit_groups.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[(T.sel == "separated") & (T.tight)].round(3).to_string())
    print(pd.DataFrame(rows).round(3).to_string())
