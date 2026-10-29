"""Classic pooled bunching estimator for a notch at r = 1 (Chetty et al. 2011; Kleven & Waseem 2013 style).

Counts of r = v / L in bins are fitted with a degree-p polynomial outside an excluded window
[r_lo, r_hi]; the upper bound r_hi is chosen so that the excess mass below 1 equals the missing mass
above 1 (mass conservation). Returns the missing fraction of exceedances.
"""
import numpy as np


def classic_bunching(r, r_lo=0.8, width=0.025, rmin=0.2, rmax=3.0, deg=7, r_hi_grid=None):
    r = np.asarray(r, float)
    r = r[(r >= rmin) & (r < rmax)]
    edges = np.arange(rmin, rmax + 1e-9, width)
    c, _ = np.histogram(r, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    if r_hi_grid is None:
        r_hi_grid = np.arange(1.05, 2.01, 0.05)
    best = None
    for r_hi in r_hi_grid:
        out = (mids < r_lo) | (mids > r_hi)
        z = (mids - 1.0)
        V = np.vander(z, deg + 1)
        beta, *_ = np.linalg.lstsq(V[out], c[out], rcond=None)
        cf = np.maximum(V @ beta, 0)
        below = (mids >= r_lo) & (mids <= 1.0)
        above = (mids > 1.0) & (mids <= r_hi)
        excess = (c[below] - cf[below]).sum()
        missing = (cf[above] - c[above]).sum()
        gap = abs(excess - missing)
        if best is None or gap < best[0]:
            exp_exc = cf[mids > 1.0].sum()
            best = (gap, r_hi, excess, missing, exp_exc)
    gap, r_hi, excess, missing, exp_exc = best
    obs_exc = (r > 1.0).sum()
    return dict(r_hi=float(r_hi), excess=float(excess), missing=float(missing),
                expected_exceed=float(exp_exc), observed_exceed=float(obs_exc),
                missing_frac=float(1 - obs_exc / exp_exc) if exp_exc > 0 else np.nan)
