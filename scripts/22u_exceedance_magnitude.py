"""Size of reported exceedances and headroom composition by constituent (exceedances counted at v > L).

For each primary stratum, detected exceedances are split into those inside the window (L < v < 2L) and those at
or above 2L. The excess above the limit is measured in multiples of the limit, v/L - 1. Because every missing
exceedance lies inside the window, its excess is below one limit; the upper-bound column assigns one limit to each.
The second table gives, by constituent, the share of reports whose historical headroom (Eq. S8) is below 2.

usage: CLIFF_THRESHOLD=L python 22u_exceedance_magnitude.py
"""
import numpy as np
import pandas as pd

import cliffmix as cm

AG = cm.PROJ / "results" / "E6u_aggregate"


def magnitude(ag):
    strata = pd.read_csv(ag / "strata.csv").set_index("stratum")
    rows = []
    for st in cm.CONV:
        p, _ = cm.load(st, columns=["v", "L", "nd"])
        r = p.v / p.L
        exc = (p.nd == 0) & (r > 1)
        small, big = exc & (r < 2), exc & (r >= 2)
        ex_small, ex_big = float((r[small] - 1).sum()), float((r[big] - 1).sum())
        miss = float(strata.loc[st, "missing"])
        rows.append(dict(stratum=st, n_exc_window=int(small.sum()), n_exc_2L=int(big.sum()),
                         share_exc_2L=big.sum() / exc.sum(), excess_window=ex_small, excess_2L=ex_big,
                         share_excess_2L=ex_big / (ex_small + ex_big), missing=miss,
                         share_excess_2L_lower_bound=ex_big / (ex_small + ex_big + miss)))
        print(st, flush=True)
    M = pd.DataFrame(rows)
    tot = M[["n_exc_window", "n_exc_2L"]].sum()
    M.loc[len(M)] = dict(stratum="pooled", n_exc_window=tot.n_exc_window, n_exc_2L=tot.n_exc_2L,
                         share_exc_2L=tot.n_exc_2L / tot.sum())
    M.to_csv(ag / "exceedance_magnitude.csv", index=False)
    return M


def headroom_composition(ag):
    G = pd.read_csv(ag / "gradients.csv")
    h = G[G["var"] == "headroom"]
    low = {"<1.25", "1.25-1.5", "1.5-2"}
    rows = []
    for g, s in h.groupby("group"):
        rows.append(dict(group=g, n=int(s.n.sum()), n_headroom_below2=int(s[s.bin.isin(low)].n.sum())))
    H = pd.DataFrame(rows)
    H["share_headroom_below2"] = H.n_headroom_below2 / H.n
    tp = h[h.group == "TP"].set_index("bin").missing_share
    conv = h[h.group == "BOD5, CBOD5, TSS"].set_index("bin").missing_share
    ratio = (tp / conv).rename("tp_to_bod_tss_shortfall_ratio").reset_index()
    H.to_csv(ag / "headroom_composition.csv", index=False)
    ratio.to_csv(ag / "headroom_tp_ratio.csv", index=False)
    return H, ratio


if __name__ == "__main__":
    ag = AG
    pd.set_option("display.width", 200)
    print(magnitude(ag).round(3).to_string())
    H, ratio = headroom_composition(ag)
    print(H.round(3).to_string())
    print(ratio.round(2).to_string())
