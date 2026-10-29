"""Location-scale distributional boosting for log-concentration intervals.

The caller supplies rounding, nondetect or coarsened interval bounds. Production
fitting uses a two-sided permit-relative window; simulation variants can use
one-sided coarsening. Bounds and numerical precision are defined in the driver.
"""
import numpy as np
import lightgbm as lgb
from scipy.special import ndtr, log_ndtr

SQ2PI = np.sqrt(2 * np.pi)


class Family:
    """Standardized location-scale density with pdf f, cdf F, log cdf/sf and score f'/f."""

    def __init__(self, name):
        self.name = name

    def pdf(self, z):
        if self.name == "normal":
            return np.exp(-0.5 * z * z) / SQ2PI
        e = np.exp(-np.abs(z))
        return e / (1 + e) ** 2

    def cdf(self, z):
        if self.name == "normal":
            return ndtr(z)
        return 1 / (1 + np.exp(-z))

    def logcdf(self, z):
        if self.name == "normal":
            return log_ndtr(z)
        return -np.logaddexp(0, -z)

    def logsf(self, z):
        return self.logcdf(-z)  # both families are symmetric

    def dlogpdf(self, z):  # f'(z)/f(z)
        if self.name == "normal":
            return -z
        return -np.tanh(z / 2)

    def logpdf(self, z):
        if self.name == "normal":
            return -0.5 * z * z - np.log(SQ2PI)
        return -np.abs(z) - 2 * np.log1p(np.exp(-np.abs(z)))


def _interval_prob(fam, zl, zh):
    """log P(zl < Z <= zh) computed stably."""
    # use the tail that keeps precision
    lp_h, lp_l = fam.logcdf(zh), fam.logcdf(zl)
    ls_h, ls_l = fam.logsf(zh), fam.logsf(zl)
    use_sf = zl > 0
    out = np.where(use_sf, ls_l + np.log1p(-np.exp(np.minimum(ls_h - ls_l, -1e-300))),
                   lp_h + np.log1p(-np.exp(np.minimum(lp_l - lp_h, -1e-300))))
    return out


def nll_and_grads(fam, mu, s, lo, hi):
    """Per-observation negative log-likelihood and gradients wrt mu and s=log sigma.
    Exact observations (lo == hi) use the density; others the interval probability."""
    sig = np.exp(s)
    exact = lo == hi
    nll = np.empty_like(mu)
    gmu = np.empty_like(mu)
    gs = np.empty_like(mu)
    hmu = np.empty_like(mu)
    # exact
    if exact.any():
        z = (lo[exact] - mu[exact]) / sig[exact]
        nll[exact] = s[exact] - fam.logpdf(z)
        r = fam.dlogpdf(z)
        gmu[exact] = r / sig[exact]
        gs[exact] = 1 + z * r
        hmu[exact] = 1 / sig[exact] ** 2
    ne = ~exact
    if ne.any():
        m, sg = mu[ne], sig[ne]
        zl = np.where(np.isfinite(lo[ne]), (lo[ne] - m) / sg, -np.inf)
        zh = np.where(np.isfinite(hi[ne]), (hi[ne] - m) / sg, np.inf)
        logP = np.where(np.isinf(zl), fam.logcdf(zh), np.where(np.isinf(zh), fam.logsf(zl), _interval_prob(fam, zl, zh)))
        logP = np.maximum(logP, -700)
        P = np.exp(logP)
        fl = np.where(np.isfinite(zl), fam.pdf(np.where(np.isfinite(zl), zl, 0)), 0.0)
        fh = np.where(np.isfinite(zh), fam.pdf(np.where(np.isfinite(zh), zh, 0)), 0.0)
        with np.errstate(invalid="ignore"):
            zfl = np.where(np.isfinite(zl), zl * fl, 0.0)
            zfh = np.where(np.isfinite(zh), zh * fh, 0.0)
        Pc = np.maximum(P, 1e-300)
        A = fh - fl
        nll[ne] = -logP
        gmu[ne] = A / (sg * Pc)
        gs[ne] = (zfh - zfl) / Pc
        # curvature in mu (exact for the normal family; Gauss-Newton style otherwise)
        h = ((zfh - zfl) / Pc + (A / Pc) ** 2) / sg ** 2
        hmu[ne] = np.clip(h, 1e-6 / sg ** 2, None)
    return nll, gmu, gs, hmu


class HICBoost:
    def __init__(self, family="normal", cycles=30, rounds_per_cycle=20, lr=0.08, num_leaves=63,
                 min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=10.0,
                 patience=4, verbose=False, seed=0, n_jobs=16):
        self.fam = Family(family)
        self.family = family
        self.cycles, self.rpc, self.patience, self.verbose = cycles, rounds_per_cycle, patience, verbose
        self.params = dict(learning_rate=lr, num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
                           feature_fraction=feature_fraction, bagging_fraction=bagging_fraction, bagging_freq=1,
                           lambda_l2=lambda_l2, verbose=-1, seed=seed, num_threads=n_jobs)

    def _init(self, lo, hi):
        mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(lo), lo, hi))
        return float(np.nanmedian(mid)), float(np.log(np.nanstd(mid) + 1e-3))

    def fit(self, X, lo, hi, Xv=None, lov=None, hiv=None, categorical=None):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.mu0, self.s0 = self._init(lo, hi)
        mu = np.full(len(lo), self.mu0)
        s = np.full(len(lo), self.s0)
        dtr = lgb.Dataset(X, free_raw_data=False, categorical_feature=categorical or "auto", params={"max_bin": 127, "verbose": -1})
        dtr.construct()
        self.mu_b, self.s_b = [], []
        has_val = Xv is not None
        if has_val:
            lov, hiv = np.asarray(lov, float), np.asarray(hiv, float)
            muv = np.full(len(lov), self.mu0)
            sv = np.full(len(lov), self.s0)
        best, best_c, wait, self.history = np.inf, 0, 0, []
        fam = self.fam
        for c in range(self.cycles):
            # --- location step (scale fixed)
            s_fixed = s.copy()
            def obj_mu(preds, data, s_fixed=s_fixed):
                _, g, _, h = nll_and_grads(fam, preds, s_fixed, lo, hi)
                return g, h
            dtr.set_init_score(mu)
            b = lgb.train({**self.params, "objective": obj_mu}, dtr, num_boost_round=self.rpc)
            mu = mu + b.predict(X, raw_score=True)
            self.mu_b.append(b)
            # --- scale step (location fixed)
            mu_fixed = mu.copy()
            def obj_s(preds, data, mu_fixed=mu_fixed):
                _, _, g, _ = nll_and_grads(fam, mu_fixed, preds, lo, hi)
                return g, np.full_like(g, 2.0)
            dtr.set_init_score(s)
            b = lgb.train({**self.params, "objective": obj_s, "learning_rate": self.params["learning_rate"] * 0.5},
                          dtr, num_boost_round=self.rpc)
            s = np.clip(s + b.predict(X, raw_score=True), -6, 3)
            self.s_b.append(b)
            tr = nll_and_grads(fam, mu, s, lo, hi)[0].mean()
            if has_val:
                muv = muv + self.mu_b[-1].predict(Xv, raw_score=True)
                sv = np.clip(sv + self.s_b[-1].predict(Xv, raw_score=True), -6, 3)
                va = nll_and_grads(fam, muv, sv, lov, hiv)[0].mean()
            else:
                va = tr
            self.history.append((c, tr, va))
            if self.verbose:
                print(f"cycle {c}: train nll {tr:.4f}  val nll {va:.4f}", flush=True)
            if va < best - 1e-5:
                best, best_c, wait = va, c, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.n_cycles = best_c + 1
        self.mu_b, self.s_b = self.mu_b[:self.n_cycles], self.s_b[:self.n_cycles]
        return self

    def predict(self, X):
        mu = np.full(X.shape[0], self.mu0)
        s = np.full(X.shape[0], self.s0)
        for bm, bs in zip(self.mu_b, self.s_b):
            mu = mu + bm.predict(X, raw_score=True)
            s = np.clip(s + bs.predict(X, raw_score=True), -6, 3)
        return mu, s

    def cdf(self, X, y, mu=None, s=None):
        if mu is None:
            mu, s = self.predict(X)
        return self.fam.cdf((y - mu) / np.exp(s))


def censored_nll(family, mu, s, lo, hi):
    return nll_and_grads(Family(family), mu, s, np.asarray(lo, float), np.asarray(hi, float))[0]


if __name__ == "__main__":
    # synthetic self-test: heteroscedastic log-normal with rounding, non-detects and a limit donut
    rng = np.random.default_rng(0)
    n = 200_000
    X = rng.normal(size=(n, 5))
    mu_t = 1.0 + 0.6 * X[:, 0] - 0.4 * X[:, 1] ** 2
    s_t = -0.8 + 0.4 * X[:, 2]
    y = mu_t + np.exp(s_t) * rng.normal(size=n)
    v = np.round(np.exp(y), 1)
    lo = np.log(np.maximum(v - 0.05, 1e-6)); hi = np.log(v + 0.05)
    nd = v < 0.5; lo[nd] = -np.inf; hi[nd] = np.log(0.5)
    L = np.exp(mu_t + 1.3 * np.exp(s_t)); a = 0.8 * L
    don = v >= a; lo[don] = np.log(a[don]); hi[don] = np.inf
    m = HICBoost(verbose=True, cycles=25).fit(X[:150000], lo[:150000], hi[:150000], X[150000:], lo[150000:], hi[150000:])
    mu, s = m.predict(X[150000:])
    print("corr mu", np.corrcoef(mu, mu_t[150000:])[0, 1], "corr s", np.corrcoef(s, s_t[150000:])[0, 1])
    pe = 1 - m.fam.cdf((np.log(L[150000:]) - mu) / np.exp(s))
    print("expected exceed", pe.sum(), "true", (y[150000:] > np.log(L[150000:])).sum())


class CrossFitHIC:
    """K-fold HIC-Boost with out-of-fold distributional recalibration.

    Tail probabilities are convex in the location error, so an in-sample scale that ignores it
    inflates expected exceedances. Each training row receives out-of-fold (mu, s); a three-parameter
    map mu' = mu + a, s' = b + c*s is then fitted by maximising the same censored likelihood on those
    out-of-fold predictions, which folds the location error into the predictive scale.
    """

    def __init__(self, k=4, family="normal", seed=0, max_train=None, **hic_kw):
        self.k, self.family, self.seed, self.hic_kw, self.max_train = k, family, seed, hic_kw, max_train

    def fit(self, X, lo, hi, groups=None, Xv=None, lov=None, hiv=None, recalibrate=True):
        from scipy.optimize import minimize
        n = len(lo)
        rng = np.random.default_rng(self.seed)
        if groups is None:
            fold = rng.integers(0, self.k, n)
        else:
            ug = np.unique(groups)
            gf = dict(zip(ug, rng.integers(0, self.k, len(ug))))
            fold = np.array([gf[g] for g in groups])
        self.models = []
        mu_oof, s_oof = np.empty(n), np.empty(n)
        for f in range(self.k):
            tr, te = fold != f, fold == f
            # early stopping on the held-out fold of this split is avoided: use a random 10% of tr
            idx = np.flatnonzero(tr)
            rng.shuffle(idx)
            nv = max(int(0.1 * len(idx)), 1)
            v_idx, t_idx = idx[:nv], idx[nv:]
            if self.max_train and len(t_idx) > self.max_train:
                t_idx = t_idx[:self.max_train]
            v_idx = v_idx[:max(self.max_train // 5, 1000)] if self.max_train else v_idx
            m = HICBoost(family=self.family, seed=self.seed + f, **self.hic_kw).fit(
                X[t_idx], lo[t_idx], hi[t_idx], X[v_idx], lo[v_idx], hi[v_idx])
            mu_oof[te], s_oof[te] = m.predict(X[te])
            self.models.append(m)
        self.mu_oof, self.s_oof, self.fold = mu_oof, s_oof, fold
        if not recalibrate:
            self.cal = np.array([0.0, 0.0, 1.0])
            self.nll_raw = self.nll_cal = float(censored_nll(self.family, mu_oof, s_oof, lo, hi).mean())
            return self

        def obj(th):
            a, b, c = th
            return censored_nll(self.family, mu_oof + a, b + c * s_oof, lo, hi).mean()
        res = minimize(obj, x0=[0.0, 0.0, 1.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=400))
        self.cal = res.x
        self.nll_raw = obj([0.0, 0.0, 1.0])
        self.nll_cal = res.fun
        return self

    def predict_raw(self, X):
        mus, ss = zip(*(m.predict(X) for m in self.models))
        return np.mean(mus, axis=0), np.mean(ss, axis=0)

    def predict(self, X):
        mu, s = self.predict_raw(X)
        a, b, c = self.cal
        return mu + a, b + c * s

    def oof(self):
        a, b, c = self.cal
        return self.mu_oof + a, b + c * self.s_oof
