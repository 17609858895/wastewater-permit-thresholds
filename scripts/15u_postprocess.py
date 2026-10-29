"""Post-processing of the unified CLIFF-Stack runs (results/E2u_unified -> results/E6u_aggregate, E5u_external).

1. stratum table (missing share, CI, AUC, calibration, family weights);
2. national and grouped aggregates (fiscal year, size, major/minor, sampling frequency, state, parameter x era / size);
3. bin-wise deficit profiles (O/E by r = v/L bins) under the stacked CDF;
4. retrospective regulatory-record associations (primary standardized shortfall z; exploratory 0.95L-L excess score);
5. sensitivity / placebo summary.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliffmix as cm  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "E6u_aggregate"; OUT.mkdir(parents=True, exist_ok=True)
EXT = PROJ / "results" / "E5u_external"; EXT.mkdir(parents=True, exist_ok=True)
CUT = pd.Timestamp("2016-09-30")
INTEGRITY = ["E0013", "C0014", "C0018", "C0017", "C0015"]
EDGES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0]
COLS = ["permit", "period_end", "v", "L", "nd", "r", "T", "mu_n", "s_n", "mu_l", "s_l", "major", "design_flow",
        "samples_pm", "state"]


def group_table(df, by):
    g = df.groupby(by, observed=True).agg(n=("E_win", "size"), E_win=("E_win", "sum"), O_win=("O_win", "sum"),
                                         O_abv=("O_abv", "sum"), permits=("permit", "nunique"))
    g["missing"] = g.E_win - g.O_win
    g["missing_share"] = g.missing / (g.E_win + g.O_abv)
    g["reported_exc_rate"] = (g.O_win + g.O_abv) / g.n
    g["counterfactual_exc_rate"] = (g.E_win + g.O_abv) / g.n
    return g


def boot(df, B=300, seed=0):
    t = df.groupby("permit").agg(E=("E_win", "sum"), O=("O_win", "sum"), A=("O_abv", "sum"))
    idx = np.random.default_rng(seed).integers(0, len(t), (B, len(t)))
    E, O, A = t.E.values[idx].sum(1), t.O.values[idx].sum(1), t.A.values[idx].sum(1)
    return np.percentile((E - O) / (E + A), [2.5, 97.5])


if __name__ == "__main__":
    strata, parts, prof, pre_parts = [], [], [], []
    for st in cm.CONV + ["TRC_C2", "TRC_C3"]:
        try:
            p, res = cm.load(st, columns=COLS)
        except FileNotFoundError:
            print("missing", st); continue
        q = cm.window_quantities(p, res)
        q["permit"] = p.permit.values
        lo, hi = boot(q)
        g = group_table(q.assign(all="all"), "all").iloc[0]
        strata.append(dict(stratum=st, n=res["n"], permits=res["permits"], w_normal=res["w_normal"],
                           ll_normal=res["mean_ll_normal"], ll_logistic=res["mean_ll_logistic"], ll_stack=res["mean_ll_stack"],
                           auc=res["auc_donut_entry"], auc_persistence=res["auc_persistence_lag1"],
                           calib_OE=res["donut_calibration_O_over_E"], missing=g.missing, missing_share=g.missing_share,
                           ci_lo=lo, ci_hi=hi, share_normal=res["missing_share_normal"], share_logistic=res["missing_share_logistic"],
                           reported_rate=g.reported_exc_rate, counterfactual_rate=g.counterfactual_exc_rate,
                           E_win=g.E_win, O_win=g.O_win, O_abv=g.O_abv,
                           **{f"OE_{d['c']}": d["ratio"] for d in res["ladder"]}))
        # deficit profile
        keep = p.nd.values == 0
        v, L = p.v.values, p.L.values
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            E = (cm.F(p, res, np.log(b * L)) - cm.F(p, res, np.log(a * L)))[keep].sum()
            O = ((v > a * L) & (v <= b * L) & keep).sum()
            prof.append(dict(stratum=st, lo=a, hi=b, O=int(O), E=float(E), OE=float(O / E)))
        if st.startswith("TRC"):
            continue
        param, vtype = st.split("_")
        q = q.assign(param=param, vtype=vtype, fy=(p.period_end.dt.year + (p.period_end.dt.month >= 10)).values,
                     major=p.major.values,
                     size=pd.cut(p.design_flow, [-0.1, 0.1, 1, 10, 1e6], labels=["<0.1 MGD", "0.1-1", "1-10", ">10"]).astype(str).values,
                     samples_pm=pd.cut(p.samples_pm, [0, 1.01, 4.5, 12.5, 100], labels=["<=1/mo", "2-4/mo", "5-12/mo", ">12/mo"]).astype(str).values,
                     state=p.state.astype(str).values)
        parts.append(q)
        # pre-period plant scores for external validation
        pre = (p.period_end <= CUT).values
        Lp, vp, kp = L[pre], v[pre], keep[pre]
        pe = (cm.F(p[pre], res, np.log(Lp * (1 - 1e-9))) - cm.F(p[pre], res, np.log(0.95 * Lp)))
        ob = ((vp > 0.95 * Lp) & (vp < Lp * (1 - 1e-9)) & kp).astype(float)
        pre_parts.append(pd.DataFrame({"permit": p.permit.values[pre], "E": q.E_win.values[pre], "O": q.O_win.values[pre],
                                       "A": q.O_abv.values[pre], "var": q.E_win.values[pre] * (1 - q.E_win.values[pre]),
                                       "ob": ob, "pe": pe, "vx": pe * (1 - pe)}))
        print(st, round(g.missing_share, 4), flush=True)
    S = pd.DataFrame(strata); S.to_csv(OUT / "strata.csv", index=False)
    pd.DataFrame(prof).to_csv(OUT / "deficit_profile.csv", index=False)
    print(S[["stratum", "n", "w_normal", "auc", "auc_persistence", "calib_OE", "missing_share", "ci_lo", "ci_hi"]].round(4).to_string())
    Q = pd.concat(parts, ignore_index=True)
    Q["era"] = np.where(Q.fy <= 2016, "FY2009-16", "FY2017-26")
    for by in ["fy", "size", "major", "samples_pm", "state", ["param", "vtype"], ["param", "era"], ["param", "size"], ["vtype"], ["param", "fy"], ["param", "major"]]:
        name = by if isinstance(by, str) else "_".join(by)
        group_table(Q, by).to_csv(OUT / f"by_{name}.csv")
    tot = group_table(Q.assign(all="all"), "all")
    lo, hi = boot(Q)
    tot["ci_lo"], tot["ci_hi"] = lo, hi
    tot.to_csv(OUT / "national.csv")
    print("NATIONAL", tot.round(4).to_string())
    # external validation
    pre = pd.concat(pre_parts).groupby("permit").sum()
    pre["n"] = pd.concat(pre_parts).groupby("permit").size()
    P = pre[pre.n >= 24].copy()
    P["z"] = (P.E - P.O) / np.sqrt(P["var"].clip(lower=1e-6))
    P["z_excess"] = (P.ob - P.pe) / np.sqrt(P["vx"].clip(lower=1e-6))
    P["exc_rate"] = (P.O + P.A) / P.n
    se = pd.read_parquet(PROJ / "data" / "NPDES_SE_VIOLATIONS_potw.parquet")
    se["date"] = pd.to_datetime(se.SINGLE_EVENT_VIOLATION_DATE, format="%m/%d/%Y", errors="coerce")
    post = se[se.date > CUT]
    P["y_integrity"] = P.index.isin(post.loc[post.VIOLATION_CODE.isin(INTEGRITY), "NPDES_ID"]).astype(int)
    P["y_effluent_insp"] = P.index.isin(post.loc[post.VIOLATION_CODE == "A0012", "NPDES_ID"]).astype(int)
    P["y_any_se"] = P.index.isin(post.NPDES_ID).astype(int)
    ins = pd.read_parquet(PROJ / "data" / "NPDES_INSPECTIONS_potw.parquet")
    ins["date"] = pd.to_datetime(ins.ACTUAL_END_DATE, format="%m/%d/%Y", errors="coerce")
    P["log_insp"] = np.log1p(ins[ins.date > CUT].groupby("NPDES_ID").size().reindex(P.index).fillna(0))
    pa = pd.read_parquet(PROJ / "data" / "permit_attributes.parquet").set_index("permit")
    P = P.join(pa[["state", "major", "design_flow"]]).dropna(subset=["state"])
    P["log_design"] = np.log1p(P.design_flow.fillna(P.design_flow.median()))
    P["log_n"] = np.log(P.n)
    for c in ["z", "z_excess", "exc_rate"]:
        P[c + "_s"] = (P[c] - P[c].mean()) / P[c].std()
    P.to_csv(EXT / "permit_scores.csv")
    rows = []
    for y in ["y_integrity", "y_effluent_insp", "y_any_se"]:
        sv = P.groupby("state")[y].agg(["min", "max"]); ok = sv[(sv["min"] == 0) & (sv["max"] == 1)].index
        Qy = P[P.state.isin(ok)]
        st = pd.get_dummies(Qy.state, prefix="st", drop_first=True, dtype=float)
        Xb = sm.add_constant(pd.concat([Qy[["log_n", "exc_rate_s", "log_design", "major", "log_insp"]].astype(float), st], axis=1))
        mb = sm.Logit(Qy[y], Xb).fit(disp=0, maxiter=500, method="bfgs")
        for score, kind in [("z_s", "primary"), ("z_excess_s", "exploratory")]:
            m = sm.Logit(Qy[y], Xb.assign(sc=Qy[score])).fit(disp=0, maxiter=500, method="bfgs")
            b, s_ = m.params["sc"], m.bse["sc"]
            qq = pd.qcut(Qy[score].rank(method="first"), 5, labels=False)
            rows.append(dict(outcome=y, score=kind, n=len(Qy), events=int(Qy[y].sum()), OR=np.exp(b), lo=np.exp(b - 1.96 * s_),
                             hi=np.exp(b + 1.96 * s_), p=m.pvalues["sc"], auc_base=roc_auc_score(Qy[y], mb.predict(Xb)),
                             auc_with=roc_auc_score(Qy[y], m.predict(Xb.assign(sc=Qy[score]))),
                             **{f"rate_q{i + 1}": Qy[y][qq == i].mean() for i in range(5)}))
    X = pd.DataFrame(rows); X.to_csv(EXT / "external_validation.csv", index=False)
    print(X.round(4).to_string())
    # sensitivity / placebo
    sens = []
    for tag in ["BOD5_C2_k0.8_lam2.0", "BOD5_C2_k0.7_lam2.0", "BOD5_C2_k0.8_lam1.5", "BOD5_C2_k0.8_lam1.5_placebo0.5"]:
        f = cm.E2U / f"{tag}.json"
        if f.exists():
            r = json.load(open(f))
            sens.append(dict(tag=tag, w=r["w_normal"], stack=r["missing_share_stack"], normal=r["missing_share_normal"],
                             logistic=r["missing_share_logistic"], ladder=json.dumps({d["c"]: round(d["ratio"], 3) for d in r["ladder"]})))
    pd.DataFrame(sens).to_csv(OUT / "sensitivity_placebo.csv", index=False)
    print(pd.DataFrame(sens).drop(columns="ladder").round(4).to_string())
