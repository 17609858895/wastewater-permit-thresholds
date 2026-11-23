"""Compare TP reports near a revised limit with a within-series median-scaling reference."""
from importlib import import_module

import numpy as np
import pandas as pd

events = import_module("07c_limit_change_controls")
precision = import_module("04_features").precision
BANDS = {"lower": (0.9, 1.0), "upper": (1.0, 1.1)}


def make_reports():
    # Redraw the stable controls within TP, using the original eligibility rules.
    ev, pairs = events.run_param("TP", np.random.default_rng(2026), return_matches=True)
    ev = ev[ev.tight & ev.unit.isin(pairs.event)].copy()
    pairs = pairs[pairs.event.isin(ev.unit)]
    published = pd.read_csv(events.RES / "limit_change_controls_events.csv")
    published = published[(published.param == "TP") & published.tight & published.n_ctrl.notna()]
    assert set(ev.unit) == set(published.unit)
    cols = events.KEY + ["period_end", "nd", "v", "L", "v_raw_str"]
    data = pd.read_parquet(events.PROJ / "data/analysis_sample.parquet", columns=cols,
                           filters=[("param", "==", "TP")])
    for col in ["statbase", "vtype", "outfall"]:
        data[col] = data[col].astype(str)
    data = data[data.nd == 0].sort_values(events.KEY + ["period_end"]).reset_index(drop=True)
    data["ser"] = data.groupby(events.KEY, sort=False).ngroup()
    data["h"] = precision(data.v_raw_str, data.v, data.v_raw_str)
    data = data[["ser", "period_end", "v", "h", "L"]]
    treated = ev.assign(event=ev.unit, cut=ev.L_new)[["event", "unit", "ser", "t0", "cut"]]
    controls = pairs.rename(columns={"cser": "ser", "Lnew": "cut"})[treated.columns]
    reports = pd.concat([treated, controls]).merge(data, on="ser")
    reports["month"] = ((reports.period_end - reports.t0).dt.days / 30.44).round().astype(int)
    reports = reports[reports.month.between(-36, 36)].copy()
    stable = reports[(reports.unit == reports.event) & (reports.month >= 0)].copy()
    stable["same"] = np.isclose(stable.L, stable.cut, rtol=1e-12, atol=0)
    stable = stable.groupby("event").same.all()
    return reports, ev.set_index("unit").permit, stable, published.set_index("unit")


def band_mass(value, width, low, high, spread):
    # Values and rounding half-widths are relative to the new or placebo limit.
    value, width = np.asarray(value), np.asarray(width)
    point = (value > low + 1e-12) & (value <= high + 1e-12)
    if not spread:
        return point.astype(float)
    a, b = np.maximum(0, value - width), value + width
    mass = np.maximum(0, np.minimum(b, high) - np.maximum(a, low))
    return np.divide(mass, b - a, out=point.astype(float), where=b > a)


def compare(data, preperiod, spread=False):
    cut_month = -18 if preperiod else 0
    r = data[data.month < 0].copy() if preperiod else data.copy()
    r["late"] = r.month >= cut_month
    groups = ["unit", "late"]
    r["logv"] = np.log(np.maximum(r.v, 1e-6))
    g = r.groupby(groups).logv
    med = g.median().unstack()
    count = g.size().unstack(fill_value=0)
    min_count = 6 if preperiod else 1
    ok = med.index[count.min(axis=1) >= min_count]
    ratio = np.exp(med.loc[ok, True] - med.loc[ok, False])
    r = r[r.unit.isin(ok)].copy()
    scale = r.unit.map(ratio)
    out = pd.DataFrame(index=ok)
    out["ratio"] = ratio
    for side, (low, high) in BANDS.items():
        val = r.v.to_numpy(dtype=float) / r.cut
        width = r.h.to_numpy(dtype=float) / r.cut
        observed = pd.Series(band_mass(val, width, low, high, spread), index=r.index)
        predicted = pd.Series(band_mass(val * scale, width * scale, low, high, spread), index=r.index)
        shares = observed.groupby([r.unit, r.late]).mean().unstack() * 100
        out[side + "_before"] = shares[False]
        out[side + "_scaled"] = predicted[~r.late].groupby(r.unit).mean() * 100
        out[side + "_after"] = shares[True]
        out[side + "_residual"] = out[side + "_after"] - out[side + "_scaled"]
    return out


def summarize(result, mapping, permits, keep, label):
    treated = result.loc[result.index.intersection(keep)].drop(columns="ratio")
    ctl = result.drop(columns="ratio").join(mapping.rename("event")).groupby("event").mean()
    common = treated.index.intersection(ctl.index)
    treated, ctl = treated.loc[common], ctl.loc[common]
    permit = permits.loc[common]
    boot_rng = np.random.default_rng(20261123)
    sizes = permit.value_counts().sort_index()
    draws = boot_rng.multinomial(len(sizes), np.repeat(1 / len(sizes), len(sizes)), 5000)
    denom = draws @ sizes.to_numpy()
    rows = []
    for group, frame in [("tightened", treated), ("stable_control", ctl), ("net", treated - ctl)]:
        sums = frame.groupby(permit).sum().reindex(sizes.index)
        boot = (draws @ sums.to_numpy()) / denom[:, None]
        for col, mean, lo, hi in zip(frame.columns, frame.mean(), *np.percentile(boot, [2.5, 97.5], axis=0)):
            rows.append([label, group, col, len(common), len(sizes), mean, lo, hi])
    return rows


def main():
    reports, permits, stable, published = make_reports()
    mapping = reports.drop_duplicates("unit").set_index("unit").event
    mapping = mapping[mapping.index != mapping]
    rows = []
    for tag, pilot, spread in [("point", False, False), ("rounding_intervals", False, True),
                               ("preperiod_halves", True, False)]:
        result = compare(reports, pilot, spread)
        if tag == "point":
            assert np.allclose(np.log(result.loc[published.index, "ratio"]), published.dq50,
                               atol=1e-6)
            rows.extend(summarize(result, mapping, permits, stable[stable].index, "unchanged_new_limit"))
        rows.extend(summarize(result, mapping, permits, permits.index, tag))
    out = pd.DataFrame(rows, columns=["analysis", "group", "measure", "events", "treated_permits",
                                     "estimate_pct_or_pp", "ci_low", "ci_high"])
    out.to_csv(events.RES / "tp_band_changes.csv", index=False)
    print(out[out.measure.str.endswith("residual")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
