"""E8. Label bias of compliance ML and integrity-aware targeting.

1) For each fitted stratum, a conventional compliance classifier (LightGBM, same features, permit-grouped
   3-fold out-of-fold) is trained on the reported exceedance label O = 1[v >= T]. Its plant-level mean risk is
   compared with the CLIFF counterfactual risk P(y* >= T): rank correlation and top-decile overlap.
2) Targeting (pre-FY2016 scores, post-FY2016 outcomes): plants ranked by
     CLIFF missing-exceedance score z, reported exceedance rate, reported-label ML risk, random;
   share of plants with later integrity violations captured when inspecting the top q of plants.
"""
import glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.special import ndtr
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("fit05", str(Path(__file__).resolve().parent / "05_fit_counterfactual.py"))
fit05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit05)

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E8u_label_bias"
RES.mkdir(parents=True, exist_ok=True)
CUT = pd.Timestamp("2016-09-30")
LAM = 2.0
INTEGRITY = ["E0013", "C0014", "C0018", "C0017", "C0015"]

rows, plant_parts = [], []
import cliffmix as cm
for st in cm.CONV:
    if not (cm.E2U / f"{st}_k0.8_lam{LAM}.json").exists():
        print("missing", st); continue
    param, vtype = st.split("_")
    p, res_cm = cm.load(st)
    feat = pd.read_parquet(PROJ / "data" / "features" / f"features_{param}.parquet",
                           columns=list(dict.fromkeys(fit05.NUM + ["v", "L", "state", "statbase"])), filters=[("vtype", "==", vtype)])
    for c in ["v", "L"] + fit05.NUM:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").astype("float64")
    feat = feat[np.isfinite(feat.L) & (feat.L > 0) & feat.v.notna()].reset_index(drop=True)
    assert len(feat) == len(p)
    feat["state_code"] = feat.state.astype("category").cat.codes
    feat["statbase_code"] = feat.statbase.astype("category").cat.codes
    X = feat[fit05.NUM + fit05.CAT].to_numpy(np.float32)
    y = p.O.values.astype(int)
    perm = p.permit.astype(str).values
    rng = np.random.default_rng(0)
    up = np.unique(perm); fold = dict(zip(up, rng.integers(0, 3, len(up))))
    fo = np.array([fold[x] for x in perm])
    risk_rep = np.zeros(len(y))
    for k in range(3):
        tr, te = fo != k, fo == k
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=12), lgb.Dataset(X[tr], y[tr]), 400)
        risk_rep[te] = m.predict(X[te])
    risk_cf = 1 - cm.F(p, res_cm, np.log(p["T"].values))
    rows.append(dict(stratum=st, auc_reported_model_on_reported=roc_auc_score(y, risk_rep),
                     auc_cliff_on_reported=roc_auc_score(y, risk_cf), mean_reported=y.mean(),
                     mean_risk_rep=risk_rep.mean(), mean_risk_cf=risk_cf.mean()))
    q = pd.DataFrame({"permit": perm, "pre": (p.period_end <= CUT).values, "O": y, "risk_rep": risk_rep, "risk_cf": risk_cf,
                      "stratum": st})
    plant_parts.append(q)
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / "report_level_auc.csv", index=False)
q = pd.concat(plant_parts, ignore_index=True)
# plant-level ranks over all strata (whole period)
P = q.groupby("permit").agg(n=("O", "size"), rep_rate=("O", "mean"), risk_rep=("risk_rep", "mean"), risk_cf=("risk_cf", "mean"))
P = P[P.n >= 24]
rho = spearmanr(P.risk_rep, P.risk_cf).correlation
top = int(0.1 * len(P))
t_rep = set(P.risk_rep.nlargest(top).index); t_cf = set(P.risk_cf.nlargest(top).index); t_rate = set(P.rep_rate.nlargest(top).index)
summary = dict(plants=len(P), spearman_rep_vs_cf=rho, top10_overlap_rep_cf=len(t_rep & t_cf) / top,
               top10_overlap_rate_cf=len(t_rate & t_cf) / top, gap_mean=(P.risk_cf - P.risk_rep).mean())
pd.Series(summary).to_csv(RES / "plant_rank_summary.csv")
print(summary)
# targeting curves using pre-period scores and post-period integrity violations
pre = q[q.pre]
Pp = pre.groupby("permit").agg(n=("O", "size"), rep_rate=("O", "mean"), risk_rep=("risk_rep", "mean"))
sc = pd.read_csv(PROJ / "results" / "E5u_external" / "permit_scores.csv", index_col=0)
Pp = Pp.join(sc[["z", "y_integrity", "log_insp"]], how="inner")
Pp = Pp[Pp.log_insp > 0]   # inspected at least once after the cut
rng = np.random.default_rng(1)
curves = []
for name, score in [("cliff_z", Pp.z), ("reported_rate", Pp.rep_rate), ("reported_ML", Pp.risk_rep),
                    ("random", pd.Series(rng.random(len(Pp)), index=Pp.index))]:
    order = score.sort_values(ascending=False).index
    yv = Pp.loc[order, "y_integrity"].values
    cap = np.cumsum(yv) / yv.sum()
    for qq in [0.05, 0.1, 0.2, 0.3, 0.5]:
        curves.append(dict(score=name, q=qq, captured=cap[int(qq * len(cap)) - 1], precision=yv[:int(qq * len(cap))].mean()))
c = pd.DataFrame(curves)
c.to_csv(RES / "targeting_curves.csv", index=False)
print(c.pivot_table(index="q", columns="score", values="captured").round(3).to_string())
print(c.pivot_table(index="q", columns="score", values="precision").round(3).to_string())
