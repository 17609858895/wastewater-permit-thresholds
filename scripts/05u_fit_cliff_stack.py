"""E2u. Unified CLIFF-Stack fit for one stratum (production settings).

python 05u_fit_cliff_stack.py --param BOD5 --vtype C2 [--kappa 0.8] [--lam 2.0] [--placebo 0.5]

Settings fixed for every stratum: permit-grouped K = 5 cross-fitting, at most 150,000 training rows per fold,
25 boosting cycles x 20 rounds, learning rate 0.06, min 300 rows per leaf. Normal and logistic latent-error
families are fitted on identical folds; the stacking weight w maximises the mean out-of-fold log probability of
each report's censoring set under w * P_normal + (1 - w) * P_logistic. All reported quantities use the stack.
The fold-1 boosters of both families are saved for SHAP.
"""
import argparse, json, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hicboost import CrossFitHIC  # noqa: E402
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("fit05", str(Path(__file__).resolve().parent / "05_fit_counterfactual.py"))
fit05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit05)

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
OUT = PROJ / "results" / "E2u_unified"
OUT.mkdir(parents=True, exist_ok=True)
K, MAX_TRAIN, CYCLES, LR = 5, 150_000, 25, 0.06


def iprob(Fn, lo, hi):
    a = np.where(np.isfinite(lo), Fn(lo), 0.0)
    b = np.where(np.isfinite(hi), Fn(hi), 1.0)
    return np.clip(b - a, 1e-300, None)


def boot_share(permit, E, O, A, B=300, seed=0):
    t = pd.DataFrame({"p": permit, "E": E, "O": O, "A": A}).groupby("p").sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), (B, len(t)))
    Eb, Ob, Ab = t.E.values[idx].sum(1), t.O.values[idx].sum(1), t.A.values[idx].sum(1)
    return np.percentile((Eb - Ob) / (Eb + Ab), [2.5, 97.5])


def dump_models(cf):
    m = cf.models[0]
    return dict(family=m.family, mu0=m.mu0, s0=m.s0, mu=[b.model_to_string() for b in m.mu_b],
                s=[b.model_to_string() for b in m.s_b])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True)
    ap.add_argument("--vtype", required=True)
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--placebo", type=float, default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.param}_{a.vtype}_k{a.kappa}_lam{a.lam}" + (f"_placebo{a.placebo}" if a.placebo else "")
    NUM, CAT = fit05.NUM, fit05.CAT
    META = ["permit", "outfall", "sid", "period_end", "param", "vtype", "statbase", "v", "L", "L_raw", "r", "nd",
            "E90", "state", "major", "design_flow", "lim_sample", "limit_begin", "h"]
    f = pd.read_parquet(DATA / "features" / f"features_{a.param}.parquet", columns=list(dict.fromkeys(META + NUM)),
                        filters=[("vtype", "==", a.vtype)])
    for c in ["v", "L", "L_raw", "r", "h"] + NUM:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["nd"] = f.nd.astype(int)
    f = f[np.isfinite(f.L) & (f.L > 0) & f.v.notna()].reset_index(drop=True)
    f["state_code"] = f.state.astype("category").cat.codes
    f["statbase_code"] = f.statbase.astype("category").cat.codes
    Leff = f.L.values * (a.placebo if a.placebo else 1.0)
    hL = fit05.limit_half_unit(f.L_raw.values, f.L.values)
    hL = np.where(np.isfinite(hL), hL, 0.0)
    T = Leff + (0.0 if a.placebo else hL)
    lo, hi, don = fit05.bounds(f, a.kappa, Leff, a.lam)
    X = f[NUM + CAT].to_numpy(dtype=np.float32)
    keep_cols = ["permit", "outfall", "sid", "period_end", "param", "vtype", "statbase", "v", "L", "r", "nd", "E90",
                 "state", "major", "design_flow", "samples_pm", "lim_sample", "limit_begin", "L_change", "year", "lag1_r"]
    pred = f[keep_cols].copy()
    groups = f.permit.values
    del f
    fits, models = {}, {}
    for fam in ["normal", "logistic"]:
        cf = CrossFitHIC(k=K, family=fam, max_train=MAX_TRAIN, lr=LR, cycles=CYCLES, min_data_in_leaf=300,
                         rounds_per_cycle=20, n_jobs=a.n_jobs).fit(X, lo, hi, groups=groups, recalibrate=False)
        fits[fam] = (cf.mu_oof, cf.s_oof)
        models[fam] = dump_models(cf)
        print(fam, "done", round(time.time() - t0), "s", flush=True)
    del X
    (mu_n, s_n), (mu_l, s_l) = fits["normal"], fits["logistic"]
    Fn = lambda x: ndtr((x - mu_n) / np.exp(s_n))
    Fl = lambda x: expit((x - mu_l) / np.exp(s_l))
    Pn, Pl = iprob(Fn, lo, hi), iprob(Fl, lo, hi)
    ws = np.linspace(0, 1, 101)
    ll = np.array([np.log(w * Pn + (1 - w) * Pl).mean() for w in ws])
    w = float(ws[ll.argmax()])
    Fs = lambda x: w * Fn(x) + (1 - w) * Fl(x)
    v, nd = pred.v.values, pred.nd.values
    keep = nd == 0
    obs_don = (keep & (v >= a.kappa * Leff)).astype(float)
    p_don = 1 - Fs(np.log(a.kappa * Leff))
    auc_don = roc_auc_score(obs_don[keep], p_don[keep])
    persist = pred.lag1_r.values
    okp = keep & np.isfinite(persist)
    auc_persist = roc_auc_score(obs_don[okp], persist[okp])
    exc = (v >= T - 1e-12) & (v > Leff) & keep
    out = dict(tag=tag, param=a.param, vtype=a.vtype, kappa=a.kappa, lam=a.lam, placebo=a.placebo,
               settings=dict(K=K, max_train=MAX_TRAIN, cycles=CYCLES, rounds_per_cycle=20, lr=LR, min_leaf=300),
               n=int(len(pred)), permits=int(pred.permit.nunique()), nd_share=float(nd.mean()),
               w_normal=w, mean_ll_normal=float(np.log(Pn).mean()), mean_ll_logistic=float(np.log(Pl).mean()),
               mean_ll_stack=float(ll.max()), auc_donut_entry=float(auc_don), auc_persistence_lag1=float(auc_persist),
               donut_calibration_O_over_E=float(obs_don[keep].sum() / p_don[keep].sum()))
    for name, Fx in [("normal", Fn), ("logistic", Fl), ("stack", Fs)]:
        FT, Fb, Fa = Fx(np.log(T)), Fx(np.log(a.lam * Leff)), Fx(np.log(a.kappa * Leff))
        E_win = Fb - FT
        O_win = (exc & (v < a.lam * Leff)).astype(float)
        O_abv = (keep & (v >= a.lam * Leff)).astype(float)
        miss = (E_win.sum() - O_win.sum()) / (E_win.sum() + O_abv.sum())
        out[f"missing_share_{name}"] = float(miss)
        if name == "stack":
            ci = boot_share(pred.permit.values, E_win, O_win, O_abv)
            out.update(E_window=float(E_win.sum()), O_window=float(O_win.sum()), O_above_window=float(O_abv.sum()),
                       missing_in_window=float(E_win.sum() - O_win.sum()), missing_ci=[float(x) for x in ci],
                       excess_below=float((keep & (v >= a.kappa * Leff) & ~exc).sum() - (FT - Fa).sum()),
                       reported_exc_rate=float(exc.mean()),
                       counterfactual_exc_rate=float((E_win.sum() + O_abv.sum()) / len(v)))
            lad = []
            for c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]:
                Ec = (1 - Fx(np.log(c * Leff))).sum()
                Oc = (keep & (v > c * Leff)).sum()
                lad.append(dict(c=c, O=int(Oc), E=float(Ec), ratio=float(Oc / Ec)))
            out["ladder"] = lad
    out["seconds"] = round(time.time() - t0, 1)
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1))
    pred["mu_n"], pred["s_n"], pred["mu_l"], pred["s_l"], pred["T"], pred["O"] = mu_n, s_n, mu_l, s_l, T, exc.astype(float)
    pred.drop(columns=["lag1_r"]).to_parquet(OUT / f"{tag}_pred.parquet", index=False)
    pickle.dump(models, open(OUT / f"{tag}_fold1_models.pkl", "wb"))
    print(json.dumps({k: v for k, v in out.items() if k != "ladder"}, indent=1), flush=True)
