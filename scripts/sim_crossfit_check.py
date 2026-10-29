"""Synthetic check of cross-fitted, recalibrated HIC-Boost.
Design A: limit depends only on observed covariates (technology-based limits).
Design B: limit also reflects the unobserved plant level (water-quality-based limits); log L is a feature.
pi = share of true exceedances displaced into [0.9 L, L)."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from hicboost import CrossFitHIC, Family
F = Family('normal')
def make(n, seed, pi, design):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)); g = rng.integers(0, 3000, n)
    u = rng.normal(size=3000)[g] * 0.3                      # unobserved plant effect
    mu_t = 1.0 + 0.6*X[:,0] - 0.4*X[:,1]**2 + u; s_t = -0.8 + 0.4*X[:,2]
    y = mu_t + np.exp(s_t)*rng.normal(size=n)
    if design == "A":
        logL = 2.0 + 0.3*X[:,3]
    else:
        logL = 2.0 + 0.3*X[:,3] + 0.7*u
    L = np.exp(logL); a = 0.8*L
    v = np.exp(y)
    exc = v > L; move = exc & (rng.random(n) < pi)
    v[move] = rng.uniform(0.9*L[move], L[move])
    v = np.round(v, 1)
    lo = np.log(np.maximum(v-0.05,1e-6)); hi = np.log(v+0.05)
    nd = v < 0.5; lo[nd] = -np.inf; hi[nd] = np.log(0.5)
    don = v >= a; lo[don] = np.log(a[don]); hi[don] = np.inf
    Xf = np.column_stack([X, logL])
    return Xf, g, lo, hi, L, v, (y>logL).sum(), move.sum()
out = []
for design in ["A", "B"]:
    for pi in [0.0, 0.5]:
        Xf, g, lo, hi, L, v, ntrue, nmoved = make(200_000, 1, pi, design)
        t0 = time.time()
        m = CrossFitHIC(k=4, lr=0.05, cycles=30, min_data_in_leaf=500, n_jobs=12).fit(Xf, lo, hi, groups=g)
        mu, s = m.oof(); mur, sr = m.mu_oof, m.s_oof
        E = (1-F.cdf((np.log(L)-mu)/np.exp(s))).sum(); Er = (1-F.cdf((np.log(L)-mur)/np.exp(sr))).sum()
        O = (v > L).sum()
        r = dict(design=design, pi=pi, true_exceed=int(ntrue), moved=int(nmoved), observed=int(O), expected_raw=round(float(Er)), expected_cal=round(float(E)),
                 cal=[round(float(x),4) for x in m.cal], miss_raw=round(float(1-O/Er),4), miss_cal=round(float(1-O/E),4), miss_true=round(float(nmoved/ntrue),4), sec=round(time.time()-t0))
        print(json.dumps(r), flush=True); out.append(r)
json.dump(out, open('../results/sim_crossfit_check.json','w'), indent=1)
