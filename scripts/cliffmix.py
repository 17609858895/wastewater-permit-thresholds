"""Helpers for the unified CLIFF-Stack outputs (normal + logistic latent errors, mixture weight w).

Exceedance threshold (2026-11-23): exceedances are counted at the permit limit itself, v > L, which matches the
effluent violation code E90 that EPA attaches to the reports (99.9% of detected TP C2 reports with L < v < T carry it).
The precision-adjusted threshold T = L + half the limit's last-place unit, stored in the prediction files by 05u, is
kept as a sensitivity analysis: set the environment variable CLIFF_THRESHOLD=T to reproduce it. Placebo runs keep the
stored threshold, which already equals the placebo cut-point.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, expit

PROJ = Path(__file__).resolve().parents[1]
E2U = PROJ / "results" / "E2u_unified"
CONV = ["BOD5_C2", "BOD5_C3", "CBOD5_C2", "CBOD5_C3", "TSS_C2", "TSS_C3", "NH3N_C2", "NH3N_C3", "TP_C2", "TP_C3",
        "FecalColi_C2", "FecalColi_C3", "Ecoli_C2", "Ecoli_C3"]


def cdf(x, mu_n, s_n, mu_l, s_l, w):
    """Stacked predictive CDF of the latent log value at x."""
    return w * ndtr((x - mu_n) / np.exp(s_n)) + (1 - w) * expit((x - mu_l) / np.exp(s_l))


THRESHOLD = os.environ.get("CLIFF_THRESHOLD", "L").upper()


def load(stratum, kappa=0.8, lam=2.0, extra="", columns=None):
    tag = f"{stratum}_k{kappa}_lam{lam}{extra}"
    res = json.load(open(E2U / f"{tag}.json"))
    at_limit = THRESHOLD == "L" and "placebo" not in extra
    cols = columns
    if at_limit and columns is not None and ({"T", "O"} & set(columns)):
        cols = list(dict.fromkeys(list(columns) + ["v", "L", "nd"]))
    p = pd.read_parquet(E2U / f"{tag}_pred.parquet", columns=cols)
    if at_limit:
        if "T" in p.columns:
            p["T"] = p["L"].values
        if "O" in p.columns:
            p["O"] = ((p.v.values > p.L.values) & (p.nd.values == 0)).astype(float)
    return p, res


def F(p, res, x):
    return cdf(x, p.mu_n.values, p.s_n.values, p.mu_l.values, p.s_l.values, res["w_normal"])


def with_h(p, stratum):
    """Attach each report's rounding half-width h (reported precision) from the feature table."""
    param, vtype = stratum.split("_")
    f = pd.read_parquet(PROJ / "data" / "features" / f"features_{param}.parquet", columns=["sid", "period_end", "h"],
                        filters=[("vtype", "==", vtype)])
    return p.merge(f, on=["sid", "period_end"], how="left", validate="one_to_one")


def frac(v, h, a, b):
    """Share of the rounding interval (max(v - h, 0), v + h] that falls in (a, b]: observed counts on the latent scale.
    A reported zero stands for (0, h], as in the fitting bounds."""
    lo, hi = np.maximum(v - h, 0.0), v + h
    w = np.maximum(hi - lo, 1e-12)
    return np.clip((np.minimum(hi, b) - np.maximum(lo, a)) / w, 0, 1)


def window_quantities(p, res, lam=2.0):
    """Per-report E_win = P(T <= y* < lam L), O_win, O_abv (reported large exceedances)."""
    T, L, v = p["T"].values, p.L.values, p.v.values
    keep = p.nd.values == 0
    E_win = F(p, res, np.log(lam * L)) - F(p, res, np.log(T))
    exc = (v >= T - 1e-12) & (v > L)
    return pd.DataFrame({"E_win": E_win, "O_win": (exc & (v < lam * L) & keep).astype(float),
                         "O_abv": ((v >= lam * L) & keep).astype(float)})
