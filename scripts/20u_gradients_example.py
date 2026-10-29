"""Historical gradients and an illustrative report sequence.

Headroom is exp(log L - median(previous log v)), using up to twelve previous
reports. The count estimand combines expected window counts and observed
extremes. The TP example uses the source concentration field and analytical
threshold; the output does not reconstruct permit-specific legal determinations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliffmix as cm  # noqa: E402

AG = cm.PROJ / "results" / "E6u_aggregate"
HEAD = [0, 1.25, 1.5, 2, 3, 5, np.inf]
HLAB = ["<1.25", "1.25-1.5", "1.5-2", "2-3", "3-5", ">5"]
PEXC = [-0.5, 0.5, 1.5, 3.5, 13]
PLAB = ["0", "1", "2-3", "4+"]
COLS = ["sid", "permit", "period_end", "v", "L", "T", "nd", "mu_n", "s_n", "mu_l", "s_l"]
B = 300


def share_ci(g, rng):
    t = g.groupby("permit")[["E_win", "O_win", "O_abv"]].sum()
    idx = rng.integers(0, len(t), (B, len(t)))
    E, O, A = t.E_win.values[idx].sum(1), t.O_win.values[idx].sum(1), t.O_abv.values[idx].sum(1)
    s = (E - O) / (E + A)
    tot = t.sum()
    return (tot.E_win - tot.O_win) / (tot.E_win + tot.O_abv), np.percentile(s, 2.5), np.percentile(s, 97.5), len(t)


def quantile(res, row, qs):
    out = []
    for q in qs:
        f = lambda x: cm.cdf(x, row.mu_n, row.s_n, row.mu_l, row.s_l, res["w_normal"]) - q
        out.append(np.exp(brentq(f, min(row.mu_n, row.mu_l) - 15, max(row.mu_n, row.mu_l) + 15)))
    return out


def gradients():
    rng = np.random.default_rng(0)
    parts = []
    for st in cm.CONV:
        p, res = cm.load(st, columns=COLS)
        param, vtype = st.split("_")
        f = pd.read_parquet(cm.PROJ / "data" / "features" / f"features_{param}.parquet",
                            columns=["sid", "period_end", "L_rel_med", "past_exc12"], filters=[("vtype", "==", vtype)])
        p = p.merge(f, on=["sid", "period_end"], how="left", validate="one_to_one")
        q = cm.window_quantities(p, res)
        parts.append(pd.DataFrame({"param": param, "vtype": vtype, "permit": p.permit.values,
                                   "head": np.exp(p.L_rel_med.values), "pexc": p.past_exc12.values,
                                   "E_win": q.E_win.values, "O_win": q.O_win.values, "O_abv": q.O_abv.values}))
        print(st, flush=True)
    Q = pd.concat(parts, ignore_index=True)
    Q["hbin"] = pd.cut(Q["head"], HEAD, labels=HLAB, right=False)
    Q["pbin"] = pd.cut(Q.pexc, PEXC, labels=PLAB)
    Q["group"] = np.where(Q.param.isin(["TP", "NH3N", "Ecoli"]), "TP, NH3-N, E. coli",
                          np.where(Q.param == "FecalColi", "F. coli", "BOD5, CBOD5, TSS"))
    rows = []
    for var, col in [("headroom", "hbin"), ("past_exceedances", "pbin")]:
        for grp_name, sub in [("All", Q)] + list(Q.groupby("group")) + list(Q.groupby("param")):
            for b, g in sub.groupby(col, observed=True):
                sh, lo, hi, npm = share_ci(g, rng)
                rows.append(dict(var=var, group=grp_name, bin=str(b), n=len(g), permits=npm, E_win=g.E_win.sum(),
                                 O_win=g.O_win.sum(), O_abv=g.O_abv.sum(), missing_share=sh, ci_lo=lo, ci_hi=hi,
                                 exc_rate=(g.E_win.sum() + g.O_abv.sum()) / len(g)))
    G = pd.DataFrame(rows)
    G.to_csv(AG / "gradients.csv", index=False)
    pd.set_option("display.width", 200)
    print(G[G.group.isin(["All", "TP", "BOD5", "TSS", "NH3N"])].round(3).to_string())


def example():
    """A typical rather than extreme bunched series: pile share closest to 0.4 among 100-150 reports."""
    p, res = cm.load("TP_C2", columns=COLS)
    p = p.sort_values(["sid", "period_end"])
    r = p.v / p.L
    g = p.assign(r=r, pile=((r > 0.9) & (p.v < p["T"])).astype(float), above=(p.v >= p["T"]).astype(float),
                 nd=p.nd).groupby("sid").agg(n=("v", "size"), nL=("L", "nunique"), pile=("pile", "mean"),
                                              above=("above", "mean"), nd=("nd", "mean"), med=("r", "median"))
    cand = g[(g.n >= 100) & (g.n <= 150) & (g.nL == 1) & (g.nd < 0.05) & (g.above < 0.04) & (g.pile >= 0.15) & (g.med > 0.5)]
    cand = cand.iloc[np.argsort(np.abs(cand.pile.values - 0.4))]
    print(cand.head(5).round(3).to_string())
    sid = cand.index[0]
    s = p[p.sid == sid].copy()
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    Qs = np.array([quantile(res, row, qs) for row in s.itertuples()])
    for j, qq in enumerate(qs):
        s[f"q{int(qq * 100)}"] = Qs[:, j]
    s["p_exc"] = 1 - cm.F(s, res, np.log(s["T"].values))
    s = s.assign(month=np.arange(len(s)))
    s[["month", "v", "L", "T", "nd", "p_exc"] + [f"q{int(qq * 100)}" for qq in qs]].to_csv(
        cm.PROJ / "figure_data" / "Fig3_example_series.csv", index=False)
    print("example series: reports", len(s), "pile share", round(cand.pile.iloc[0], 3), "E exceed", round(s.p_exc.sum(), 2),
          "reported", int((s.v >= s["T"]).sum()))


if __name__ == "__main__":
    if "example" not in sys.argv:
        gradients()
    example()
