"""Feature table for the counterfactual-distribution model (one row per report in the analysis sample).

Only information that does not depend on the current report's value is used:
past values of the same series, limit design, sampling regime, same-month flow/influent and plant
attributes. Same-month values of other effluent statistics are excluded because they face the same limit
incentive.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
SERIES = ["permit", "outfall", "limset", "param", "statbase", "vtype"]


def freq_per_month(code):
    """'01/30' -> 1 sample per 30 days -> samples per month."""
    m = re.match(r"^(\d+)/(\d+)$", str(code))
    if not m:
        return np.nan
    a, b = int(m.group(1)), int(m.group(2))
    return a * 30.0 / b if b > 0 else np.nan


def precision(vstr, v_std, v_raw):
    """Half-width of the rounding interval in standard units, from the decimals of the reported number."""
    s = vstr.astype(str)
    dec = s.str.extract(r"\.(\d+)$")[0].str.len().fillna(0).astype(float)
    step = 10.0 ** (-dec)
    v_raw = np.asarray(pd.to_numeric(v_raw, errors="coerce"), dtype=float)
    v_std = np.asarray(v_std, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where((v_raw > 0) & np.isfinite(v_raw), v_std / v_raw, 1.0)
    return 0.5 * step.to_numpy(dtype=float) * scale


def build(d, cov, pa):
    d = d.sort_values(SERIES + ["period_end"]).reset_index(drop=True)
    d["v_raw"] = pd.to_numeric(d.v_raw_str, errors="coerce").astype("float64")
    for c in ["v", "L", "L_raw", "r"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    d["h"] = precision(d.v_raw_str, d.v, d.v_raw)
    d["logv"] = np.log(np.maximum(d.v, 1e-6))
    d["logL"] = np.log(d.L)
    g = d.groupby(SERIES, sort=False, observed=True)
    d["sid"] = g.ngroup()
    gl = d.groupby("sid", sort=False)
    # lags of the same series (report order)
    for k in [1, 2, 3, 6, 12]:
        d[f"lag{k}"] = gl.logv.shift(k)
        d[f"lag{k}_r"] = gl.r.shift(k)
    d["lag1_nd"] = gl.nd.shift(1)
    d["gap_days"] = (d.period_end - gl.period_end.shift(1)).dt.days
    # rolling history (last 12 reports, excluding current)
    def roll(s, w, mp, how, q=None):
        r = s.groupby(d.sid, sort=False).rolling(w, min_periods=mp)
        out = r.quantile(q) if how == "quantile" else getattr(r, how)()
        return out.reset_index(level=0, drop=True).sort_index()
    sh = gl.logv.shift(1)
    d["roll_med12"] = roll(sh, 12, 3, "median")
    d["roll_q90_12"] = roll(sh, 12, 3, "quantile", 0.9)
    d["roll_q10_12"] = roll(sh, 12, 3, "quantile", 0.1)
    d["roll_sd12"] = roll(sh, 12, 3, "std")
    d["roll_med36"] = roll(sh, 36, 6, "median")
    shr = gl.r.shift(1)
    d["roll_rmax12"] = roll(shr, 12, 1, "max")
    d["past_donut12"] = roll(shr.between(0.8, 1.0).astype(float).where(shr.notna()), 12, 1, "mean")
    d["past_exc12"] = roll((shr > 1).astype(float).where(shr.notna()), 12, 1, "sum")
    d["past_nd12"] = roll(gl.nd.shift(1).astype(float), 12, 1, "mean")
    d["n_prior"] = gl.cumcount()
    d["L_rel_med"] = d.logL - d.roll_med12
    d["L_rel_med36"] = d.logL - d.roll_med36
    # limit regime
    d["L_change"] = (gl.L.shift(1) != d.L).astype(int)
    d.loc[gl.cumcount() == 0, "L_change"] = 0
    d["months_since_limit_begin"] = (d.period_end - d.limit_begin).dt.days / 30.4
    d["samples_pm"] = d.lim_freq.fillna("").astype(object).map(freq_per_month)
    d["grab"] = (d.lim_sample.fillna("") == "GR").astype(int)
    d["comp"] = d.lim_sample.fillna("").str.startswith("C").astype(int)
    d["month"] = d.period_end.dt.month
    d["year"] = d.period_end.dt.year + d.period_end.dt.month / 12
    # covariates
    d["pm"] = d.period_end.dt.to_period("M").dt.to_timestamp("M")
    d = d.merge(cov, on=["permit", "pm"], how="left")
    d = d.merge(pa, on="permit", how="left")
    d["state"] = d.state.fillna(d.permit.str[:2])
    d["log_design"] = np.log1p(d.design_flow)
    d["flow_util"] = d.flow_avg / d.design_flow.replace(0, np.nan)
    d["flow_peak_ratio"] = d.flow_max / d.flow_avg.replace(0, np.nan)
    d["log_flow"] = np.log1p(d.flow_avg)
    for c in ["inf_BOD5", "inf_CBOD5", "inf_TSS", "inf_NH3N"]:
        d["log_" + c] = np.log1p(d[c])
    d = d.drop(columns=["inf_BOD5", "inf_CBOD5", "inf_TSS", "inf_NH3N", "pm"])
    return d


if __name__ == "__main__":
    import sys
    cov = pd.read_parquet(DATA / "monthly_covariates.parquet")
    pa = pd.read_parquet(DATA / "permit_attributes.parquet")[["permit", "state", "major", "design_flow", "pretreat", "lat", "lon", "permit_type"]]
    params = sys.argv[1:] or ["BOD5", "CBOD5", "TSS", "NH3N", "TP", "FecalColi", "Ecoli", "TRC"]
    (DATA / "features").mkdir(exist_ok=True)
    for p in params:
        d = pd.read_parquet(DATA / "analysis_sample.parquet", filters=[("param", "==", p)])
        d = build(d, cov, pa)
        d.to_parquet(DATA / "features" / f"features_{p}.parquet", index=False)
        print(p, "rows", len(d), "series", d.sid.nunique(), flush=True)
        print(d[["lag1", "roll_med12", "flow_avg", "log_inf_BOD5", "samples_pm", "design_flow"]].notna().mean().round(3).to_dict(), flush=True)
