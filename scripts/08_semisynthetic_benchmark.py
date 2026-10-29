"""E3. Semi-synthetic benchmark on real report series that never approach their limit.

Real BOD5 monthly-average series whose reported values always stay below 0.6 L are taken as honest latent
truth. Each series gets a pseudo-limit L' = k x (series median), k in {1.6, 2.0, 2.6}. A share pi of the
reports above L' is displaced to U[0.9 L', L'] and re-rounded to the original precision. Features are
recomputed from the modified series. Estimators of the missing-exceedance fraction:
  classic   pooled polynomial bunching on r = v / L' (notch, mass-balance window)
  tobit     censored normal with linear location and linear log-scale (covariate bunching, Bertanha-type)
  hic_raw   single HIC-Boost fit, in-sample predictions
  hic_cf    cross-fitted HIC-Boost with out-of-fold distributional recalibration (proposed)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hicboost import HICBoost, CrossFitHIC, Family, nll_and_grads  # noqa: E402
from bunching_classic import classic_bunching  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "E3_semisynthetic"
OUT.mkdir(parents=True, exist_ok=True)
FAM = Family("normal")
STATIC = ["month", "year", "log_flow", "flow_util", "flow_peak_ratio", "log_inf_BOD5", "log_inf_TSS", "log_design",
          "major", "lat", "lon", "samples_pm", "grab", "comp"]


def recompute(df):
    g = df.groupby("sid", sort=False)
    out = pd.DataFrame(index=df.index)
    out["logL"] = np.log(df.Lp)
    for k in [1, 2, 3, 6, 12]:
        out[f"lag{k}"] = g.logv.shift(k)
    out["lag1_r"] = g.r.shift(1)
    sh = g.logv.shift(1)
    def roll(s, w, mp, how, q=None):
        r = s.groupby(df.sid, sort=False).rolling(w, min_periods=mp)
        o = r.quantile(q) if how == "quantile" else getattr(r, how)()
        return o.reset_index(level=0, drop=True).sort_index()
    out["roll_med12"] = roll(sh, 12, 3, "median")
    out["roll_q90_12"] = roll(sh, 12, 3, "quantile", 0.9)
    out["roll_sd12"] = roll(sh, 12, 3, "std")
    out["roll_med36"] = roll(sh, 36, 6, "median")
    shr = g.r.shift(1)
    out["past_donut12"] = roll(shr.between(0.8, 1.0).astype(float).where(shr.notna()), 12, 1, "mean")
    out["past_exc12"] = roll((shr > 1).astype(float).where(shr.notna()), 12, 1, "sum")
    out["L_rel_med"] = out.logL - out.roll_med12
    for c in STATIC:
        out[c] = df[c].astype(float)
    return out


def bounds(v, h, nd, a, b=None):
    lo = np.log(np.maximum(v - h, 1e-9)); hi = np.log(v + h)
    z = v <= 0
    lo[z], hi[z] = -np.inf, np.log(np.maximum(h[z], 1e-6))
    lo[nd], hi[nd] = -np.inf, np.log(np.maximum(v[nd], 1e-6))
    if b is None:
        don = (~nd) & (v >= a)
        lo[don], hi[don] = np.log(a[don]), np.inf
    else:
        don = (~nd) & (v >= a) & (v < b)
        lo[don], hi[don] = np.log(a[don]), np.log(b[don])
    return lo, hi


def missing_window(O_all, v, Lp, nd, mu, s, lam):
    """two-sided estimator: (E_win - O_win) / (E_win + O_above)"""
    FT = FAM.cdf((np.log(Lp) - mu) / np.exp(s)); Fb = FAM.cdf((np.log(lam * Lp) - mu) / np.exp(s))
    E_win = (Fb - FT).sum()
    O_win = ((v > Lp) & (v < lam * Lp) & ~nd).sum()
    O_abv = ((v >= lam * Lp) & ~nd).sum()
    return float((E_win - O_win) / (E_win + O_abv))


def fit_tobit(X, lo, hi):
    Xs = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9))
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    p = Xs.shape[1]
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(lo), lo, hi))
    th0 = np.zeros(2 * p); th0[0] = np.nanmedian(mid); th0[p] = np.log(np.nanstd(mid))
    def f(th):
        mu, s = Xs @ th[:p], np.clip(Xs @ th[p:], -6, 3)
        nll, gmu, gs, _ = nll_and_grads(FAM, mu, s, lo, hi)
        return nll.mean(), np.concatenate([Xs.T @ gmu, Xs.T @ gs]) / len(mu)
    th = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
    return Xs @ th[:p], np.clip(Xs @ th[p:], -6, 3)


def missing(O, p):
    return 1 - O.sum() / p.sum()


if __name__ == "__main__":
    t0 = time.time()
    f = pd.read_parquet(PROJ / "data" / "features" / "features_BOD5.parquet", filters=[("vtype", "==", "C2")])
    for c in ["v", "L", "r", "h", "logv"] + STATIC:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["nd"] = f.nd.astype(int)
    smax = f.groupby("sid").r.transform("max")
    cnt = f.groupby("sid").r.transform("size")
    base = f[(smax < 0.6) & (cnt >= 36)].copy()
    rng = np.random.default_rng(42)
    sids = base.sid.unique()
    rng.shuffle(sids)
    base = base[base.sid.isin(sids[:4000])].sort_values(["sid", "period_end"]).reset_index(drop=True)
    med = base.groupby("sid").v.transform("median")
    kser = pd.Series(rng.choice([1.6, 2.0, 2.6], size=base.sid.nunique()), index=base.sid.unique())
    base["Lp"] = np.maximum(med * base.sid.map(kser), 1.0)
    print("series", base.sid.nunique(), "rows", len(base), "true exceed share", ((base.v > base.Lp) & (base.nd == 0)).mean().round(4), flush=True)
    rows = []
    for pi in [0.0, 0.25, 0.5, 0.75]:
        for rep in range(1):
            r_ = np.random.default_rng(100 * rep + int(pi * 100))
            d = base.copy()
            exc = (d.v > d.Lp) & (d.nd == 0)
            move = exc & (d.v < 2.0 * d.Lp) & (r_.random(len(d)) < pi)  # local displacement from (L', 2L')
            newv = r_.uniform(0.9 * d.Lp[move], d.Lp[move])
            step = 2 * d.h[move]
            d.loc[move, "v"] = np.floor(newv / step) * step  # round down to reported precision
            d.loc[move & (d.v > d.Lp), "v"] = d.Lp[move & (d.v > d.Lp)]
            d["logv"] = np.log(np.maximum(d.v, 1e-6)); d["r"] = d.v / d.Lp
            X = recompute(d).values
            v, h, nd, Lp = d.v.values, d.h.values, d.nd.values == 1, d.Lp.values
            lo, hi = bounds(v, h, nd, 0.8 * Lp)
            O = ((v > Lp) & ~nd).astype(float)
            truth = move.sum() / exc.sum()
            res = dict(pi=pi, rep=rep, truth=float(truth), n=len(d), true_exceed=int(exc.sum()), observed=int(O.sum()))
            cb = classic_bunching(d.r.values[~nd], r_lo=0.8)
            res["classic"] = cb["missing_frac"]
            mu, s = fit_tobit(X, lo, hi)
            res["tobit"] = float(missing(O, 1 - FAM.cdf((np.log(Lp) - mu) / np.exp(s))))
            m = HICBoost(lr=0.05, cycles=25, min_data_in_leaf=300, n_jobs=12).fit(X, lo, hi)
            mu, s = m.predict(X)
            res["hic_raw"] = float(missing(O, 1 - FAM.cdf((np.log(Lp) - mu) / np.exp(s))))
            cf = CrossFitHIC(k=3, lr=0.05, cycles=25, min_data_in_leaf=300, n_jobs=12).fit(X, lo, hi, groups=d.permit.values)
            mu, s = cf.oof()
            res["hic_cf"] = float(missing(O, 1 - FAM.cdf((np.log(Lp) - mu) / np.exp(s))))
            mu, s = cf.mu_oof, cf.s_oof
            res["hic_cf_nocal"] = float(missing(O, 1 - FAM.cdf((np.log(Lp) - mu) / np.exp(s))))
            # proposed main estimator: two-sided window [0.8 L, 2 L), cross-fitted, no recalibration
            lo2, hi2 = bounds(v, h, nd, 0.8 * Lp, 2.0 * Lp)
            cf2 = CrossFitHIC(k=3, lr=0.05, cycles=25, min_data_in_leaf=300, n_jobs=12).fit(X, lo2, hi2, groups=d.permit.values)
            res["cliff_twosided"] = missing_window(O, v, Lp, nd, cf2.mu_oof, cf2.s_oof, 2.0)
            mu_t, s_t = fit_tobit(X, lo2, hi2)
            res["tobit_twosided"] = missing_window(O, v, Lp, nd, mu_t, s_t, 2.0)
            # truth on the same window scale: displaced / (true exceedances)

            res["sec"] = round(time.time() - t0)
            print(json.dumps(res), flush=True)
            rows.append(res)
            pd.DataFrame(rows).to_csv(OUT / "semisynthetic_results.csv", index=False)
