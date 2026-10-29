"""E3c. Repeated semi-synthetic recovery with the final national model (normal-logistic mixture).

Same 1,724 BOD5 C2 series, pseudo-limits and feature recomputation as E3 (08_semisynthetic_benchmark.py), but each
replicate fits the production configuration of 05u_fit_cliff_stack.py: permit-grouped K = 5 cross-fitting, at most
150,000 training rows per fold, 25 cycles x 20 rounds, learning rate 0.06, min 300 rows per leaf, normal and
logistic components on identical folds, mixture weight chosen on the out-of-fold interval log score, two-sided
window [0.8L, 2L) and no recalibration. Replicates differ in the displacement draw and in the permit-to-fold map.

Displacement mechanisms
  iid     each true exceedance in (L', 2L') is displaced with probability pi (as in E3; rep 0 reproduces its draw)
  permit  a share q of permits displaces; within them each exceedance in (L', 2L') is displaced with probability 0.9,
          so displacement is concentrated in some report histories rather than spread across all of them

python 08c_semisynthetic_mixture_repeats.py --job iid 0.5 0 [--nj 8]
python 08c_semisynthetic_mixture_repeats.py --queue 3 0.8 8      # parallel jobs, min available GB, threads per job
python 08c_semisynthetic_mixture_repeats.py --collect
"""
import sys, json, time, argparse, subprocess, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hicboost import CrossFitHIC  # noqa: E402
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("e3", str(Path(__file__).resolve().parent / "08_semisynthetic_benchmark.py"))
e3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e3)

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "results" / "E3_semisynthetic" / "repeats"
BASE = PROJ / "data" / "features" / "semisyn_base_BOD5_C2.parquet"
LOGS = PROJ / "logs"
K, MAX_TRAIN, CYCLES, LR = 5, 150_000, 25, 0.06
JOBS = ([("iid", p, 0) for p in [0.0, 0.25, 0.5, 0.75]] +
        [("permit", q, 0) for q in [0.25, 0.5, 0.75]] +
        [("iid", p, 1) for p in [0.0, 0.5]])


def build_base():
    cols = ["sid", "permit", "period_end", "v", "L", "r", "h", "logv", "nd"] + e3.STATIC
    f = pd.read_parquet(PROJ / "data" / "features" / "features_BOD5.parquet", columns=cols, filters=[("vtype", "==", "C2")])
    for c in ["v", "L", "r", "h", "logv"] + e3.STATIC:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f["nd"] = f.nd.astype(int)
    smax = f.groupby("sid").r.transform("max")
    cnt = f.groupby("sid").r.transform("size")
    base = f[(smax < 0.6) & (cnt >= 36)].copy()
    del f
    rng = np.random.default_rng(42)  # identical selection and pseudo-limits to E3
    sids = base.sid.unique()
    rng.shuffle(sids)
    base = base[base.sid.isin(sids[:4000])].sort_values(["sid", "period_end"]).reset_index(drop=True)
    med = base.groupby("sid").v.transform("median")
    kser = pd.Series(rng.choice([1.6, 2.0, 2.6], size=base.sid.nunique()), index=base.sid.unique())
    base["Lp"] = np.maximum(med * base.sid.map(kser), 1.0)
    base.to_parquet(BASE, index=False)
    return base


def displace(base, mech, p, rep):
    d = base.copy()
    exc = (d.v > d.Lp) & (d.nd == 0)
    if mech == "iid":
        r_ = np.random.default_rng(100 * rep + int(p * 100))
        move = exc & (d.v < 2.0 * d.Lp) & (r_.random(len(d)) < p)
    else:
        r_ = np.random.default_rng(5000 + 100 * rep + int(p * 100))
        perms = d.permit.unique()
        disp = pd.Series(r_.random(len(perms)) < p, index=perms)
        move = exc & (d.v < 2.0 * d.Lp) & d.permit.map(disp).astype(bool).values & (r_.random(len(d)) < 0.9)
    newv = r_.uniform(0.9 * d.Lp[move], d.Lp[move])
    step = 2 * d.h[move]
    d.loc[move, "v"] = np.floor(newv / step) * step
    d.loc[move & (d.v > d.Lp), "v"] = d.Lp[move & (d.v > d.Lp)]
    d["logv"] = np.log(np.maximum(d.v, 1e-6)); d["r"] = d.v / d.Lp
    return d, float(move.sum() / exc.sum()), int(exc.sum()), int(move.sum())


def iprob(F, lo, hi):
    a = np.where(np.isfinite(lo), F(lo), 0.0)
    b = np.where(np.isfinite(hi), F(hi), 1.0)
    return np.clip(b - a, 1e-300, None)


def run_job(mech, p, rep, nj):
    t0 = time.time()
    base = pd.read_parquet(BASE)
    d, truth, n_exc, n_move = displace(base, mech, p, rep)
    X = e3.recompute(d).values.astype(np.float32)
    v, h, nd, Lp = d.v.values, d.h.values, d.nd.values == 1, d.Lp.values
    lo, hi = e3.bounds(v, h, nd, 0.8 * Lp, 2.0 * Lp)
    fits = {}
    for fam in ["normal", "logistic"]:
        cf = CrossFitHIC(k=K, family=fam, seed=rep, max_train=MAX_TRAIN, lr=LR, cycles=CYCLES, min_data_in_leaf=300,
                         rounds_per_cycle=20, n_jobs=nj).fit(X, lo, hi, groups=d.permit.values, recalibrate=False)
        fits[fam] = (cf.mu_oof, cf.s_oof)
        print(mech, p, rep, fam, "done", round(time.time() - t0), "s", flush=True)
    (mu_n, s_n), (mu_l, s_l) = fits["normal"], fits["logistic"]
    Fn = lambda x: ndtr((x - mu_n) / np.exp(s_n))
    Fl = lambda x: expit((x - mu_l) / np.exp(s_l))
    Pn, Pl = iprob(Fn, lo, hi), iprob(Fl, lo, hi)
    ws = np.linspace(0, 1, 101)
    ll = np.array([np.log(w * Pn + (1 - w) * Pl).mean() for w in ws])
    w = float(ws[ll.argmax()])
    Fs = lambda x: w * Fn(x) + (1 - w) * Fl(x)
    keep = ~nd
    O_win = ((v > Lp) & (v < 2 * Lp) & keep).sum()
    O_abv = ((v >= 2 * Lp) & keep).sum()
    res = dict(mech=mech, p=p, rep=rep, truth=truth, n=len(d), true_exceed=n_exc, displaced=n_move,
               permits=int(d.permit.nunique()), w_normal=w, ll_normal=float(np.log(Pn).mean()),
               ll_logistic=float(np.log(Pl).mean()), ll_stack=float(ll.max()))
    for name, F in [("normal", Fn), ("logistic", Fl), ("stack", Fs)]:
        E_win = (F(np.log(2 * Lp)) - F(np.log(Lp))).sum()
        res[f"est_{name}"] = float((E_win - O_win) / (E_win + O_abv))
    res["seconds"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{mech}_{p}_{rep}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def queue(P, min_gb, nj):
    import psutil
    if not BASE.exists():
        b = build_base()
        print("base", b.sid.nunique(), "series", len(b), "rows", flush=True)
    def external():
        """Jobs already running in processes this runner did not start (e.g. after a runner restart)."""
        out = set()
        for pr in psutil.process_iter(["cmdline"]):
            c = pr.info["cmdline"] or []
            if any("08c_semisynthetic" in x for x in c) and "--job" in c:
                k = c.index("--job"); out.add((c[k + 1], float(c[k + 2]), int(c[k + 3])))
        return out
    todo = [j for j in JOBS if not (OUT / f"{j[0]}_{j[1]}_{j[2]}.json").exists()]
    running = []
    env = dict(os.environ, PYTHONUTF8="1")
    while todo or running or external():
        running = [(j, pr) for j, pr in running if pr.poll() is None]
        ext = external() - {j for j, _ in running}
        todo = [j for j in todo if j not in ext]
        if not todo and not running:
            time.sleep(30); continue
        if todo and len(running) + len(ext) < P and psutil.virtual_memory().available / 2**30 >= min_gb:
            j = todo.pop(0)
            log = open(LOGS / f"08c_{j[0]}_{j[1]}_{j[2]}.log", "w", encoding="utf-8")
            pr = subprocess.Popen([sys.executable, __file__, "--job", j[0], str(j[1]), str(j[2]), "--nj", str(nj)],
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((j, pr))
            print(time.strftime("%H:%M:%S"), "started", j, flush=True)
            time.sleep(90)  # let the job load before checking memory again
        else:
            time.sleep(30)
    print("queue done", flush=True)


def collect():
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    t = pd.DataFrame(rows).sort_values(["mech", "p", "rep"])
    t.to_csv(OUT.parent / "semisynthetic_mixture_repeats.csv", index=False)
    for c in ["stack", "normal", "logistic"]:
        t[f"err_{c}"] = 100 * (t[f"est_{c}"] - t.truth)
    s = t.groupby(["mech", "p"]).agg(reps=("rep", "size"), truth=("truth", "mean"), est=("est_stack", "mean"),
                                      est_sd=("est_stack", "std"), err=("err_stack", "mean"), err_sd=("err_stack", "std"),
                                      err_normal=("err_normal", "mean"), err_logistic=("err_logistic", "mean"),
                                      w_normal=("w_normal", "mean")).reset_index()
    s.to_csv(OUT.parent / "semisynthetic_mixture_summary.csv", index=False)
    print(s.round(4).to_string())
    print("MAE stack (pp):", t.groupby("mech").err_stack.apply(lambda x: x.abs().mean()).round(3).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", nargs=3)
    ap.add_argument("--queue", nargs=3)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--nj", type=int, default=8)
    a = ap.parse_args()
    if a.job:
        run_job(a.job[0], float(a.job[1]), int(a.job[2]), a.nj)
    elif a.queue:
        queue(int(a.queue[0]), float(a.queue[1]), int(a.queue[2]))
    elif a.collect:
        collect()
