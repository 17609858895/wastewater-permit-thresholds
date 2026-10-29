"""Adjusted inspection-targeting curves (replaces the unadjusted curve in Fig. 6d; the unadjusted one moves to Fig. S10).

Same plants, outcomes and baseline model as the external validation in 15u: FY2009-2016 plant scores, single-event
violations after 2016-09-30, logit with state fixed effects, log reports, standardized reported exceedance rate,
log design flow, major and log(1 + later inspections). Plants are ranked nationally by a score; among the top q,
observed later violations are divided by those expected from the baseline model (O/E). O/E = 1 means the score adds
nothing beyond the covariates and state. Intervals: plant bootstrap with the baseline fit held fixed; reference band:
2.5-97.5 percentiles of O/E under random rankings.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJ = Path(__file__).resolve().parents[1]
EXT = PROJ / "results" / "E5u_external"
QS = [0.05, 0.1, 0.2, 0.3, 0.5]
B = 1000

P = pd.read_csv(EXT / "permit_scores.csv", index_col=0)
rows = []
for y in ["y_integrity", "y_effluent_insp", "y_any_se"]:
    sv = P.groupby("state")[y].agg(["min", "max"]); ok = sv[(sv["min"] == 0) & (sv["max"] == 1)].index
    Q = P[P.state.isin(ok)]
    st = pd.get_dummies(Q.state, prefix="st", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([Q[["log_n", "exc_rate_s", "log_design", "major", "log_insp"]].astype(float), st], axis=1))
    p0 = sm.Logit(Q[y], X).fit(disp=0, maxiter=500, method="bfgs").predict(X).values
    yv = Q[y].values.astype(float)
    rng = np.random.default_rng(0)
    for score, kind in [("z", "shortfall"), ("z_excess", "excess")]:
        s = Q[score].values
        for qq in QS:
            k = int(qq * len(Q))
            top = np.argsort(-s)[:k]
            oe = yv[top].sum() / p0[top].sum()
            bs = []
            for _ in range(B):
                i = rng.integers(0, len(Q), len(Q))
                t = i[np.argsort(-s[i])[:k]]
                bs.append(yv[t].sum() / p0[t].sum())
            rows.append(dict(outcome=y, score=kind, q=qq, n=len(Q), k=k, O=yv[top].sum(), E=p0[top].sum(), OE=oe,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             rate_obs=yv[top].mean(), rate_exp=p0[top].mean()))
    for qq in QS:
        k = int(qq * len(Q))
        r = [(lambda t: yv[t].sum() / p0[t].sum())(rng.permutation(len(Q))[:k]) for _ in range(B)]
        rows.append(dict(outcome=y, score="random", q=qq, n=len(Q), k=k, OE=np.mean(r), lo=np.percentile(r, 2.5),
                         hi=np.percentile(r, 97.5)))
T = pd.DataFrame(rows)
T.to_csv(EXT / "targeting_adjusted.csv", index=False)
pd.set_option("display.width", 200)
print(T.round(3).to_string())
