"""E4c. Unchanged-limit controls for the model-free limit-change contrasts (E4 without TRC).

Treated events follow 07_limit_change_events.py exactly: the first within-series change of the limit by a factor of
at least 1.2, with at least 12 detected reports on each side within 36 months. For each event, up to M control
series of the same parameter, concentration field and statistical basis are drawn from other permits whose limit
stayed within 5% over their whole record and which have at least 12 detected reports on each side of the same
calendar date t0. A control receives the placebo new limit L_c x (L_new / L_old) of its matched event, so it
experiences the same calendar period and the same relative cut-point without a limit change. For treated and
control series the E4 quantities are computed identically:
  did  change (post - pre) in the new-limit contrast (share in (0.9, 1.0] minus share in (1.0, 1.1] of the
       cut-point) minus the change in the same contrast at the old limit
  dq   change in the 25th, 50th and 75th percentiles of log(v / L_old)
Net change = treated minus the mean of its matched controls; 95% intervals resample events (1,000 draws).
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results" / "E4_limit_change"
KEY = ["permit", "outfall", "param", "statbase", "vtype"]
PARAMS = ["BOD5", "CBOD5", "TSS", "NH3N", "TP", "FecalColi", "Ecoli"]
M, W, B = 20, 36, 1000


def unit_metrics(r):
    """r: rows with unit, et, v, Lold, Lnew -> per-unit did and quantile changes."""
    r = r.copy()
    r["post"] = (r.et >= 0).astype(int)
    for lim, tag in [("Lold", "o"), ("Lnew", "n")]:
        q = r.v / r[lim]
        r[f"b{tag}"] = ((q > 0.9) & (q <= 1.0)).astype(float)
        r[f"a{tag}"] = ((q > 1.0) & (q <= 1.1)).astype(float)
    r["lr"] = np.log(np.maximum(r.v, 1e-6) / r.Lold)
    g = r.groupby(["unit", "post"])
    s = g[["bn", "an", "bo", "ao"]].mean()
    s["cn"], s["co"] = s.bn - s.an, s.bo - s.ao
    s["q25"], s["q50"], s["q75"] = g.lr.quantile(0.25), g.lr.median(), g.lr.quantile(0.75)
    s["share_new"] = s.bn  # share just below the new cut-point
    w = s[["cn", "co", "q25", "q50", "q75", "share_new"]].unstack("post").dropna()
    out = pd.DataFrame(index=w.index)
    out["did"] = (w[("cn", 1)] - w[("cn", 0)]) - (w[("co", 1)] - w[("co", 0)])
    out["dnew"] = w[("share_new", 1)] - w[("share_new", 0)]
    for q in ["q25", "q50", "q75"]:
        out[f"d{q}"] = w[(q, 1)] - w[(q, 0)]
    return out


def run_param(p, rng):
    d = pd.read_parquet(PROJ / "data" / "analysis_sample.parquet", columns=KEY + ["period_end", "v", "L", "nd"],
                        filters=[("param", "==", p)])
    for c in ["statbase", "vtype", "outfall"]:
        d[c] = d[c].astype(str)
    d = d[d.nd == 0].sort_values(KEY + ["period_end"]).reset_index(drop=True)
    d["ser"] = d.groupby(KEY, sort=False).ngroup()
    # --- treated events (E4 definition)
    d["Lprev"] = d.groupby("ser").L.shift(1)
    chg = d[(d.Lprev.notna()) & (np.abs(np.log(d.L / d.Lprev)) >= np.log(1.2))].groupby("ser").head(1)
    ev = chg[["ser", "permit", "vtype", "statbase", "period_end", "L", "Lprev"]].rename(
        columns={"period_end": "t0", "L": "L_new", "Lprev": "L_old"})
    t = d[["ser", "period_end", "v"]].merge(ev[["ser", "t0", "L_new", "L_old"]], on="ser")
    t["et"] = ((t.period_end - t.t0).dt.days / 30.44).round().astype(int)
    t = t[(t.et >= -W) & (t.et <= W)]
    sup = t.groupby("ser").et.agg(pre=lambda x: (x < 0).sum(), post=lambda x: (x >= 0).sum())
    ok = sup[(sup.pre >= 12) & (sup.post >= 12)].index
    ev = ev[ev.ser.isin(ok)].reset_index(drop=True)
    ev["tight"] = ev.L_new < ev.L_old
    ev["unit"] = "T" + ev.ser.astype(str)
    t = t[t.ser.isin(ok)].rename(columns={"L_old": "Lold", "L_new": "Lnew"})
    t["unit"] = "T" + t.ser.astype(str)
    # --- stable-limit control pool
    st = d.groupby("ser").agg(permit=("permit", "first"), vtype=("vtype", "first"), statbase=("statbase", "first"),
                              Lmin=("L", "min"), Lmax=("L", "max"), Lc=("L", "median"))
    st = st[st.Lmax / st.Lmin <= 1.05]
    cd = d[d.ser.isin(st.index)][["ser", "period_end", "v"]]
    m0 = d.period_end.min()
    mi = ((cd.period_end.dt.year - m0.year) * 12 + cd.period_end.dt.month - m0.month).to_numpy()
    sidx = pd.Series(np.arange(len(st)), index=st.index)
    nm = mi.max() + W + 2
    cnt = np.zeros((len(st), nm + 1), dtype=np.int32)
    np.add.at(cnt, (sidx[cd.ser].to_numpy(), mi + 1), 1)
    cum = cnt.cumsum(1)  # cum[:, k] = reports with month index < k
    pairs = []
    for (vt, sb), g in ev.groupby(["vtype", "statbase"]):
        pool = st[(st.vtype == vt) & (st.statbase == sb)]
        if pool.empty:
            continue
        pi_ = sidx[pool.index].to_numpy()
        for e in g.itertuples():
            k0 = (e.t0.year - m0.year) * 12 + e.t0.month - m0.month
            lo_, hi_ = max(k0 - W, 0), min(k0 + W + 1, nm)
            pre = cum[pi_, k0] - cum[pi_, lo_]
            post = cum[pi_, hi_] - cum[pi_, k0]
            elig = pool.index[(pre >= 12) & (post >= 12) & (pool.permit.to_numpy() != e.permit)]
            if len(elig) == 0:
                continue
            pick = rng.choice(elig, size=min(M, len(elig)), replace=False)
            for j, c in enumerate(pick):
                pairs.append((e.unit, f"C{e.ser}_{j}", c, e.t0, st.at[c, "Lc"], e.L_new / e.L_old))
    pr = pd.DataFrame(pairs, columns=["event", "unit", "cser", "t0", "Lold", "f"])
    pr["Lnew"] = pr.Lold * pr.f
    rows = []
    for i in range(0, len(pr), 4000):
        c = pr.iloc[i:i + 4000].merge(cd.rename(columns={"ser": "cser"}), on="cser")
        c["et"] = ((c.period_end - c.t0).dt.days / 30.44).round().astype(int)
        rows.append(c[(c.et >= -W) & (c.et <= W)][["unit", "et", "v", "Lold", "Lnew"]])
    cr = pd.concat(rows) if rows else pd.DataFrame(columns=["unit", "et", "v", "Lold", "Lnew"])
    mt = unit_metrics(t[["unit", "et", "v", "Lold", "Lnew"]])
    mc = unit_metrics(cr)
    mc = mc.join(pr.drop_duplicates("unit").set_index("unit")[["event"]])
    cagg = mc.groupby("event").agg(n_ctrl=("did", "size"), did_c=("did", "mean"), dnew_c=("dnew", "mean"),
                                   dq25_c=("dq25", "mean"), dq50_c=("dq50", "mean"), dq75_c=("dq75", "mean"))
    out = ev.set_index("unit")[["ser", "tight", "t0", "L_old", "L_new"]].join(mt).join(cagg)
    out["param"] = p
    # event-time profile of the share just below the new cut-point (treated vs matched controls)
    def prof(r, grp):
        q = r.v / r.Lnew
        r = r.assign(below_new=((q > 0.9) & (q <= 1.0)).astype(float), above_new=((q > 1.0) & (q <= 1.1)).astype(float))
        return r.groupby("et")[["below_new", "above_new"]].mean().assign(group=grp, param=p)
    tight_units = set(ev.unit[ev.tight])
    pt = prof(t[t.unit.isin(tight_units)], "treated")
    ctl_t = pr[pr.event.isin(tight_units)].unit
    pc = prof(cr[cr.unit.isin(set(ctl_t))], "control") if len(ctl_t) else None
    return out.reset_index(), pd.concat([x for x in [pt, pc] if x is not None]).reset_index()


def boot(x, rng):
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    evs, profs = [], []
    for p in PARAMS:
        e, pf = run_param(p, rng)
        evs.append(e); profs.append(pf)
        print(p, "events", len(e), "with controls", int(e.n_ctrl.notna().sum()), flush=True)
    E = pd.concat(evs, ignore_index=True)
    E.to_csv(RES / "limit_change_controls_events.csv", index=False)
    pd.concat(profs, ignore_index=True).to_csv(RES / "limit_change_controls_profile.csv", index=False)
    rows = []
    groups = [((t, "All"), g) for t, g in E.groupby("tight")] + [((t, p), g) for (t, p), g in E.groupby(["tight", "param"])]
    for (tight, param), g in groups:
        m = g[g.n_ctrl.notna() & g.did.notna()]
        if len(m) < 30:
            continue
        rec = dict(tight=tight, param=param, events=len(g), matched=len(m), controls=int(m.n_ctrl.sum()))
        for name, x in [("did_treated", m.did), ("did_control", m.did_c), ("did_net", m.did - m.did_c),
                        ("dnew_treated", m.dnew), ("dnew_control", m.dnew_c), ("dnew_net", m.dnew - m.dnew_c)]:
            rec[name], rec[name + "_lo"], rec[name + "_hi"] = boot(100 * x, rng)  # percentage points
        for q in ["q25", "q50", "q75"]:
            for name, x in [("treated", m[f"d{q}"]), ("control", m[f"d{q}_c"]), ("net", m[f"d{q}"] - m[f"d{q}_c"])]:
                mu, lo, hi = boot(x, rng)
                rec[f"d{q}_{name}_pct"] = 100 * (np.exp(mu) - 1)
                rec[f"d{q}_{name}_lo"], rec[f"d{q}_{name}_hi"] = 100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)
        rows.append(rec)
    S = pd.DataFrame(rows)
    S.to_csv(RES / "limit_change_controls_summary.csv", index=False)
    cols = ["tight", "param", "events", "matched", "controls", "did_treated", "did_control", "did_net", "did_net_lo",
            "did_net_hi", "dq50_treated_pct", "dq50_control_pct", "dq50_net_pct", "dq50_net_lo", "dq50_net_hi"]
    print(S[cols].round(3).to_string())
