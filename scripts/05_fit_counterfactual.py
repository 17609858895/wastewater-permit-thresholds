"""E2. Counterfactual report distribution and missing exceedances for one stratum (parameter x statistic).

python 05_fit_counterfactual.py --param BOD5 --vtype C2 [--kappa 0.8] [--placebo 0.7] [--max_train 400000]

The latent log value y* of every report is modelled by cross-fitted HIC-Boost (permit-grouped folds,
out-of-fold recalibration). Reports at or above kappa * L_eff are right-censored at kappa * L_eff, so the
model never sees the shape of the distribution where reports can respond to the limit. With --placebo c the
pipeline is run against a fake limit c * L (no incentive at c < kappa_true) as a falsification test.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hicboost import CrossFitHIC, Family, censored_nll  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"

NUM = ["logL", "L_rel_med", "L_rel_med36", "lag1", "lag2", "lag3", "lag6", "lag12", "lag1_r", "lag2_r",
       "lag1_nd", "gap_days", "roll_med12", "roll_q90_12", "roll_q10_12", "roll_sd12", "roll_med36",
       "roll_rmax12", "past_donut12", "past_exc12", "past_nd12", "n_prior", "L_change",
       "months_since_limit_begin", "samples_pm", "grab", "comp", "month", "year", "log_flow", "flow_util",
       "flow_peak_ratio", "log_inf_BOD5", "log_inf_CBOD5", "log_inf_TSS", "log_inf_NH3N", "log_design",
       "major", "lat", "lon"]
CAT = ["state_code", "statbase_code"]


def bounds(df, kappa, Leff, lam=None):
    v, h = df.v.values, df.h.values
    lo = np.log(np.maximum(v - h, 1e-9))
    hi = np.log(v + h)
    zero = (v <= 0)
    lo[zero], hi[zero] = -np.inf, np.log(np.maximum(h[zero], 1e-6))
    nd = df.nd.values == 1
    lo[nd], hi[nd] = -np.inf, np.log(np.maximum(v[nd], 1e-6))
    a = kappa * Leff
    if lam is None:
        don = (~nd) & (v >= a)
        lo[don], hi[don] = np.log(a[don]), np.inf
    else:
        # two-sided window: reports in [kappa L, lam L) only tell that y* lies in that window
        b = lam * Leff
        don = (~nd) & (v >= a) & (v < b)
        lo[don], hi[don] = np.log(a[don]), np.log(b[don])
    return lo, hi, don


def limit_half_unit(L_raw, L_std):
    """Half of the last reported decimal unit of the limit, in standard units (legal rounding allowance)."""
    L_raw = np.asarray(L_raw, float)
    dec = np.zeros(len(L_raw))
    for k in range(6, -1, -1):
        dec = np.where(np.isclose(np.round(L_raw, k), L_raw, rtol=0, atol=1e-9), k, dec)
    scale = np.where((L_raw > 0) & np.isfinite(L_raw), L_std / L_raw, 1.0)
    return 0.5 * 10.0 ** (-dec) * scale


def summarize(df, p_exc, Leff, by=None):
    obs = (df.v.values > Leff) & (df.nd.values == 0)
    t = pd.DataFrame({"O": obs.astype(float), "E": p_exc})
    if by is not None:
        t["by"] = by
        return t.groupby("by").agg(O=("O", "sum"), E=("E", "sum"), n=("O", "size"))
    return t.sum()


def boot_missing(permit, O, E, B=300, seed=0):
    t = pd.DataFrame({"p": permit, "O": O, "E": E}).groupby("p").sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), (B, len(t)))
    Ob, Eb = t.O.values[idx].sum(1), t.E.values[idx].sum(1)
    m = 1 - Ob / Eb
    return np.percentile(m, [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--vtype", required=True)
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--placebo", type=float, default=None)
    ap.add_argument("--max_train", type=int, default=400000)
    ap.add_argument("--family", default="normal")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.param}_{a.vtype}_k{a.kappa}" + (f"_lam{a.lam}" if a.lam else "") + (f"_placebo{a.placebo}" if a.placebo else "") + (f"_{a.family}" if a.family != "normal" else "") + a.tag
    out = PROJ / "results" / "E2_counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    META = ["permit", "outfall", "sid", "period_end", "param", "vtype", "statbase", "v", "L", "L_raw", "r", "nd",
            "E90", "state", "major", "design_flow", "lim_sample", "limit_begin", "h"]
    cols = list(dict.fromkeys(META + NUM))
    f = pd.read_parquet(DATA / "features" / f"features_{a.param}.parquet", columns=cols, filters=[("vtype", "==", a.vtype)])
    for c in ["v", "L", "L_raw", "r", "h"] + NUM:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["nd"] = f.nd.astype(int)
    f = f[np.isfinite(f.L) & (f.L > 0) & f.v.notna()].reset_index(drop=True)
    f["state_code"] = f.state.astype("category").cat.codes
    f["statbase_code"] = f.statbase.astype("category").cat.codes
    Leff = f.L.values * (a.placebo if a.placebo else 1.0)
    # legal exceedance threshold: the limit plus its rounding allowance (none at a placebo limit)
    hL = limit_half_unit(f.L_raw.values, f.L.values)
    hL = np.where(np.isfinite(hL), hL, 0.0)
    T = Leff + (0.0 if a.placebo else hL)
    lo, hi, don = bounds(f, a.kappa, Leff, a.lam)
    X = f[NUM + CAT].to_numpy(dtype=np.float32)
    model = CrossFitHIC(k=a.k, family=a.family, max_train=a.max_train, lr=0.05, cycles=a.cycles,
                        min_data_in_leaf=300, rounds_per_cycle=20, n_jobs=a.n_jobs).fit(X, lo, hi, groups=f.permit.values)
    mu_c, s_c = model.oof()            # recalibrated (ablation)
    mu, s = model.mu_oof, model.s_oof  # main: raw out-of-fold predictions
    fam = Family(a.family)
    sig = np.exp(s)
    lL = np.log(Leff)
    p_exc = 1 - fam.cdf((np.log(T) - mu) / sig)
    p_don = 1 - fam.cdf((np.log(a.kappa * Leff) - mu) / sig)
    obs_don = ((f.nd.values == 0) & (f.v.values >= a.kappa * Leff)).astype(float)
    # discrimination / calibration of the fully observed donut-entry event
    from sklearn.metrics import roc_auc_score, brier_score_loss
    keep = f.nd.values == 0
    auc_don = roc_auc_score(obs_don[keep], p_don[keep])
    cal_don = obs_don[keep].sum() / p_don[keep].sum()
    persist = f.lag1_r.values
    ok = keep & np.isfinite(persist)
    auc_persist = roc_auc_score(obs_don[ok], persist[ok])
    O = ((f.v.values >= T - 1e-12) & (f.v.values > Leff) & keep).astype(float)
    E = p_exc
    E_cal = (1 - fam.cdf((np.log(T) - mu_c) / np.exp(s_c))).sum()
    win = {}
    if a.lam:
        Fb = fam.cdf((np.log(a.lam * Leff) - mu) / sig)
        FT = fam.cdf((np.log(T) - mu) / sig)
        Fa = fam.cdf((np.log(a.kappa * Leff) - mu) / sig)
        vv = f.v.values
        O_win = ((vv >= T - 1e-12) & (vv > Leff) & (vv < a.lam * Leff) & keep).sum()
        E_win = (Fb - FT).sum()
        O_above = ((vv >= a.lam * Leff) & keep).sum()
        X_obs = ((vv >= a.kappa * Leff) & ~((vv >= T - 1e-12) & (vv > Leff)) & keep).sum()
        X_exp = (FT - Fa).sum()
        win = dict(O_window=float(O_win), E_window=float(E_win), O_above_window=float(O_above),
                   missing_in_window=float(E_win - O_win), missing_frac_window=float(1 - O_win / E_win),
                   missing_frac_total_window=float((E_win - O_win) / (E_win + O_above)),
                   excess_below=float(X_obs - X_exp), excess_obs=float(X_obs), excess_exp=float(X_exp))
    miss = 1 - O.sum() / E.sum()
    ci = boot_missing(f.permit.values, O, E)
    # deficit / calibration ladder relative to L_eff
    ladder = []
    for c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        pc = 1 - fam.cdf((np.log(c * Leff) - mu) / sig)
        oc = ((f.v.values > c * Leff) & keep).sum()
        ladder.append(dict(c=c, O=int(oc), E=float(pc.sum()), ratio=float(oc / pc.sum())))
    res = dict(tag=tag, param=a.param, vtype=a.vtype, kappa=a.kappa, placebo=a.placebo, family=a.family,
               n=int(len(f)), permits=int(f.permit.nunique()), donut_share=float(obs_don.mean()),
               nd_share=float(f.nd.mean()), O=float(O.sum()), E=float(E.sum()), missing_frac=float(miss),
               missing_ci=[float(x) for x in ci], missing_count=float(E.sum() - O.sum()),
               missing_frac_recal=float(1 - O.sum() / E_cal), O_strict=float(((f.v.values > Leff) & keep).sum()),
               auc_donut_entry=float(auc_don), auc_persistence_lag1=float(auc_persist),
               donut_calibration_O_over_E=float(cal_don), cal_params=[float(x) for x in model.cal],
               oof_nll_raw=float(model.nll_raw), oof_nll_cal=float(model.nll_cal),
               ladder=ladder, seconds=round(time.time() - t0, 1), **win)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    keep_cols = ["permit", "outfall", "sid", "period_end", "param", "vtype", "statbase", "v", "L", "r", "nd", "E90",
                 "state", "major", "design_flow", "samples_pm", "lim_sample", "limit_begin", "L_change", "year"]
    pred = f[keep_cols].copy()
    pred["mu"], pred["s"], pred["p_exc"], pred["p_don"], pred["T"], pred["O"] = mu, s, p_exc, p_don, T, O
    pred.to_parquet(out / f"{tag}_pred.parquet", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "ladder"}, indent=1))
    print(pd.DataFrame(ladder).round(3).to_string())
