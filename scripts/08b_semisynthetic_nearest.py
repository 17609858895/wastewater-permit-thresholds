"""E3b. Nearest-method baselines on the same semi-synthetic benchmark as E3 (same series, pseudo-limits, displacement
draws and permit-grouped folds).

  xgb_aft         XGBoost survival:aft (normal), same interval/left/window censoring as CLIFF, constant scale sigma
                  tuned per fold on held-out censored likelihood (off-the-shelf interval-censored boosting;
                  Barnwal, Cho & Hocking 2022, JCGS)
  cliff_nowindow  cross-fitted HIC-Boost with rounding and non-detect censoring only: reports in [0.8L, 2L) enter
                  as rounded values, so the learned distribution absorbs the cliff
Both are scored with the two-sided window estimator of 08 (cliff_twosided there).
"""
import sys, json, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hicboost import CrossFitHIC, censored_nll  # noqa: E402
spec = importlib.util.spec_from_file_location("e3", str(Path(__file__).resolve().parent / "08_semisynthetic_benchmark.py"))
e3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e3)

PROJ, OUT, STATIC = e3.PROJ, e3.OUT, e3.STATIC
SIGMAS = [0.3, 0.45, 0.65, 0.9]
NJ = 8


def folds_like_crossfit(groups, k=3, seed=0):
    """Same permit-to-fold map as CrossFitHIC(k, seed)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gf = dict(zip(ug, rng.integers(0, k, len(ug))))
    return np.array([gf[g] for g in groups]), rng


def xgb_aft_oof(X, lo, hi, groups, k=3):
    fold, rng = folds_like_crossfit(groups, k)
    yl = np.where(np.isfinite(lo), np.exp(lo), 0.0)
    yu = np.where(np.isfinite(hi), np.exp(hi), np.inf)
    mu = np.empty(len(lo)); sig = np.empty(len(lo)); log = []
    for f in range(k):
        idx = np.flatnonzero(fold != f); rng.shuffle(idx)
        nv = int(0.1 * len(idx)); v_idx, t_idx = idx[:nv], idx[nv:]
        te = fold == f
        dt = xgb.DMatrix(X[t_idx]); dt.set_float_info("label_lower_bound", yl[t_idx]); dt.set_float_info("label_upper_bound", yu[t_idx])
        dv = xgb.DMatrix(X[v_idx]); dv.set_float_info("label_lower_bound", yl[v_idx]); dv.set_float_info("label_upper_bound", yu[v_idx])
        best = None
        for sg in SIGMAS:
            p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                     aft_loss_distribution_scale=sg, tree_method="hist", learning_rate=0.05, max_depth=6,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, nthread=NJ, seed=f)
            b = xgb.train(p, dt, num_boost_round=3000, evals=[(dv, "v")], early_stopping_rounds=50, verbose_eval=False)
            m_v = b.predict(dv, output_margin=True, iteration_range=(0, b.best_iteration + 1))
            nll = float(censored_nll("normal", m_v, np.full(len(m_v), np.log(sg)), lo[v_idx], hi[v_idx]).mean())
            log.append(dict(fold=f, sigma=sg, rounds=b.best_iteration + 1, val_nll=nll))
            if best is None or nll < best[0]:
                best = (nll, sg, b)
        _, sg, b = best
        mu[te] = b.predict(xgb.DMatrix(X[te]), output_margin=True, iteration_range=(0, b.best_iteration + 1))
        sig[te] = sg
    return mu, np.log(sig), log


if __name__ == "__main__":
    t0 = time.time()
    f = pd.read_parquet(PROJ / "data" / "features" / "features_BOD5.parquet", filters=[("vtype", "==", "C2")])
    for c in ["v", "L", "r", "h", "logv"] + STATIC:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["nd"] = f.nd.astype(int)
    smax = f.groupby("sid").r.transform("max")
    cnt = f.groupby("sid").r.transform("size")
    base = f[(smax < 0.6) & (cnt >= 36)].copy()
    del f
    rng = np.random.default_rng(42)
    sids = base.sid.unique()
    rng.shuffle(sids)
    base = base[base.sid.isin(sids[:4000])].sort_values(["sid", "period_end"]).reset_index(drop=True)
    med = base.groupby("sid").v.transform("median")
    kser = pd.Series(rng.choice([1.6, 2.0, 2.6], size=base.sid.nunique()), index=base.sid.unique())
    base["Lp"] = np.maximum(med * base.sid.map(kser), 1.0)
    print("series", base.sid.nunique(), "rows", len(base), flush=True)
    rows, logs = [], []
    for pi in [0.0, 0.25, 0.5, 0.75]:
        r_ = np.random.default_rng(int(pi * 100))
        d = base.copy()
        exc = (d.v > d.Lp) & (d.nd == 0)
        move = exc & (d.v < 2.0 * d.Lp) & (r_.random(len(d)) < pi)
        newv = r_.uniform(0.9 * d.Lp[move], d.Lp[move])
        step = 2 * d.h[move]
        d.loc[move, "v"] = np.floor(newv / step) * step
        d.loc[move & (d.v > d.Lp), "v"] = d.Lp[move & (d.v > d.Lp)]
        d["logv"] = np.log(np.maximum(d.v, 1e-6)); d["r"] = d.v / d.Lp
        X = e3.recompute(d).values
        v, h, nd, Lp = d.v.values, d.h.values, d.nd.values == 1, d.Lp.values
        O = ((v > Lp) & ~nd).astype(float)
        res = dict(pi=pi, truth=float(move.sum() / exc.sum()), n=len(d))
        lo2, hi2 = e3.bounds(v, h, nd, 0.8 * Lp, 2.0 * Lp)
        mu, s, lg = xgb_aft_oof(X, lo2, hi2, d.permit.values)
        res["xgb_aft"] = e3.missing_window(O, v, Lp, nd, mu, s, 2.0)
        res["xgb_aft_sigma"] = float(np.exp(np.median(s)))
        logs += [dict(pi=pi, **x) for x in lg]
        lo0, hi0 = e3.bounds(v, h, nd, np.full(len(v), np.inf))
        cf0 = CrossFitHIC(k=3, lr=0.05, cycles=25, min_data_in_leaf=300, n_jobs=NJ).fit(X, lo0, hi0, groups=d.permit.values,
                                                                                         recalibrate=False)
        res["cliff_nowindow"] = e3.missing_window(O, v, Lp, nd, cf0.mu_oof, cf0.s_oof, 2.0)
        res["sec"] = round(time.time() - t0)
        print(json.dumps(res), flush=True)
        rows.append(res)
        pd.DataFrame(rows).to_csv(OUT / "semisynthetic_nearest.csv", index=False)
        pd.DataFrame(logs).to_csv(OUT / "semisynthetic_nearest_xgb_log.csv", index=False)
