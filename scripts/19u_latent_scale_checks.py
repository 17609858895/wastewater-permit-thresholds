"""Rounding-consistent (latent-scale) observed counts for the unified CLIFF-Stack runs.

The predictive CDF describes the latent value y*, while a detected report v stands for the rounding interval
(v - h, v + h]. Counting reports by v in narrow bins around a round limit shifts mass across bin edges (for L = 30 and
integer reports, (28.5, 30] holds two reporting units but only 1.5 latent units), which manufactures an O/E jump at
the limit even without displacement. Here every detected report is spread uniformly over its rounding interval
before it is compared with E:

1. deficit profiles by v/L bin (replaces 15u deficit_profile.csv for Fig. 4b and Fig. S6);
2. the observed/expected ladder above c * L for the true-limit and fake-limit BOD5 runs (Fig. 3e);
3. the missing share with latent-scale O_win and O_abv, a robustness check on the main estimator.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliffmix as cm  # noqa: E402

AG = cm.PROJ / "results" / "E6u_aggregate"
EDGES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0]
LADDER = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]
COLS = ["sid", "permit", "period_end", "v", "L", "T", "nd", "mu_n", "s_n", "mu_l", "s_l"]


def ladder(p, res, Leff):
    keep = p.nd.values == 0
    v, h = p.v.values, p.h.values
    out = []
    for c in LADDER:
        E = (1 - cm.F(p, res, np.log(c * Leff)))[keep].sum()
        O = cm.frac(v, h, c * Leff, np.inf)[keep].sum()
        O_raw = ((v > c * Leff) & keep).sum()
        out.append(dict(c=c, E=E, O=O, ratio=O / E, ratio_raw=O_raw / E))
    return pd.DataFrame(out)


if __name__ == "__main__":
    prof, strata = [], []
    for st in cm.CONV + ["TRC_C2", "TRC_C3"]:
        p, res = cm.load(st, columns=COLS)
        p = cm.with_h(p, st)
        keep = p.nd.values == 0
        v, h, L, T = p.v.values, p.h.values, p.L.values, p["T"].values
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            E = (cm.F(p, res, np.log(b * L)) - cm.F(p, res, np.log(a * L)))[keep].sum()
            O = cm.frac(v, h, a * L, b * L)[keep].sum()
            O_raw = ((v > a * L) & (v <= b * L) & keep).sum()
            prof.append(dict(stratum=st, lo=a, hi=b, O=float(O), E=float(E), OE=float(O / E), OE_raw=float(O_raw / E)))
        E_win = cm.F(p, res, np.log(2 * L)) - cm.F(p, res, np.log(T))
        q = cm.window_quantities(p, res)
        O_win_l = cm.frac(v, h, T, 2 * L) * keep
        O_abv_l = cm.frac(v, h, 2 * L, np.inf) * keep
        strata.append(dict(stratum=st, E_win=E_win.sum(), O_win=q.O_win.sum(), O_abv=q.O_abv.sum(),
                           O_win_latent=O_win_l.sum(), O_abv_latent=O_abv_l.sum(),
                           share=(E_win.sum() - q.O_win.sum()) / (E_win.sum() + q.O_abv.sum()),
                           share_latent=(E_win.sum() - O_win_l.sum()) / (E_win.sum() + O_abv_l.sum())))
        print(st, round(strata[-1]["share"], 4), round(strata[-1]["share_latent"], 4), flush=True)
    pd.DataFrame(prof).to_csv(AG / "deficit_profile_latent.csv", index=False)
    S = pd.DataFrame(strata)
    conv = S[~S.stratum.str.startswith("TRC")]
    nat = dict(stratum="national", **{c: conv[c].sum() for c in ["E_win", "O_win", "O_abv", "O_win_latent", "O_abv_latent"]})
    nat["share"] = (nat["E_win"] - nat["O_win"]) / (nat["E_win"] + nat["O_abv"])
    nat["share_latent"] = (nat["E_win"] - nat["O_win_latent"]) / (nat["E_win"] + nat["O_abv_latent"])
    S = pd.concat([S, pd.DataFrame([nat])], ignore_index=True)
    S.to_csv(AG / "strata_latent_check.csv", index=False)
    print(S.round(4).to_string())
    lad = []
    for tag, scale in [("BOD5_C2_k0.8_lam2.0", 1.0), ("BOD5_C2_k0.8_lam1.5_placebo0.5", 0.5)]:
        res = json.load(open(cm.E2U / f"{tag}.json"))
        p = pd.read_parquet(cm.E2U / f"{tag}_pred.parquet", columns=COLS)
        p = cm.with_h(p, "BOD5_C2")
        lad.append(ladder(p, res, scale * p.L.values).assign(run=tag, scale=scale))
    lad = pd.concat(lad, ignore_index=True)
    lad.to_csv(cm.E2U / "ladder_latent_BOD5_C2.csv", index=False)
    print(lad.round(3).to_string())
