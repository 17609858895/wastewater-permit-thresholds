"""Build analysis tables from the per-year POTW DMR extracts.

Outputs (data/):
  effluent_panel.parquet  one row per (permit, outfall, limit set, parameter, stat base, value type, period)
                          for 8 conventional effluent parameters at monitoring location 1 (effluent gross)
  monthly_covariates.parquet  permit-month flow (50050) and influent BOD/TSS (location G), never used as targets
  permit_attributes.parquet   current permit/facility attributes (POTW, version 0)
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
from project_paths import RAW_ROOT as ROOT
SRC = PROJ / "data" / "dmr_potw"
OUT = PROJ / "data"

PARAMS = {"00310": "BOD5", "80082": "CBOD5", "00530": "TSS", "00610": "NH3N",
          "00665": "TP", "74055": "FecalColi", "51040": "Ecoli", "50060": "TRC"}
KEY = ["EXTERNAL_PERMIT_NMBR", "PERM_FEATURE_NMBR", "LIMIT_SET_DESIGNATOR", "PARAMETER_CODE",
       "STATISTICAL_BASE_CODE", "VALUE_TYPE_CODE", "MONITORING_PERIOD_END_DATE"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def dt(s):
    return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")


def effluent_year(f):
    d = pd.read_parquet(f)
    d = d[(d.MONITORING_LOCATION_CODE == "1") & d.PARAMETER_CODE.isin(PARAMS)
          & d.VALUE_TYPE_CODE.isin(["C2", "C3"])].copy()
    d["E90"] = (d.VIOLATION_CODE == "E90").astype(np.int8)
    d["D90"] = (d.VIOLATION_CODE == "D90").astype(np.int8)
    agg = d.groupby(KEY, sort=False)[["E90", "D90"]].max()
    d = d.drop_duplicates(KEY).set_index(KEY)
    d[["E90", "D90"]] = agg
    d = d.reset_index()
    out = pd.DataFrame({
        "permit": d.EXTERNAL_PERMIT_NMBR, "outfall": d.PERM_FEATURE_NMBR,
        "limset": d.LIMIT_SET_DESIGNATOR, "param": d.PARAMETER_CODE.map(PARAMS),
        "statbase": d.STATISTICAL_BASE_CODE, "stattype": d.STATISTICAL_BASE_TYPE_CODE,
        "vtype": d.VALUE_TYPE_CODE, "period_end": dt(d.MONITORING_PERIOD_END_DATE),
        "limit_id": d.LIMIT_ID, "limit_begin": dt(d.LIMIT_BEGIN_DATE), "limit_end": dt(d.LIMIT_END_DATE),
        "L": num(d.LIMIT_VALUE_STANDARD_UNITS), "L_raw": num(d.LIMIT_VALUE_NMBR),
        "L_qual": d.LIMIT_VALUE_QUALIFIER_CODE, "unit_std": d.STANDARD_UNIT_CODE,
        "limit_type": d.LIMIT_TYPE_CODE, "optional": d.OPTIONAL_MONITORING_FLAG, "stay": d.STAY_TYPE_CODE,
        "lim_freq": d.LIMIT_FREQ_OF_ANALYSIS_CODE, "lim_sample": d.LIMIT_SAMPLE_TYPE_CODE,
        "dmr_freq": d.DMR_FREQ_OF_ANALYSIS_CODE, "dmr_sample": d.DMR_SAMPLE_TYPE_CODE,
        "v": num(d.DMR_VALUE_STANDARD_UNITS), "v_raw_str": d.DMR_VALUE_NMBR, "v_unit": d.DMR_UNIT_CODE,
        "qual": d.DMR_VALUE_QUALIFIER_CODE, "nodi": d.NODI_CODE,
        "received": dt(d.VALUE_RECEIVED_DATE), "days_late": num(d.DAYS_LATE),
        "excursions": num(d.REPORTED_EXCURSION_NMBR), "exceed_pct": num(d.EXCEEDENCE_PCT),
        "E90": d.E90, "D90": d.D90,
    })
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("string[pyarrow]")
    return out


def covariates_year(f):
    d = pd.read_parquet(f, columns=["EXTERNAL_PERMIT_NMBR", "PARAMETER_CODE", "MONITORING_LOCATION_CODE",
                                    "VALUE_TYPE_CODE", "MONITORING_PERIOD_END_DATE", "DMR_VALUE_STANDARD_UNITS",
                                    "STATISTICAL_BASE_CODE"])
    d["v"] = num(d.DMR_VALUE_STANDARD_UNITS)
    d["period_end"] = dt(d.MONITORING_PERIOD_END_DATE)
    d = d[d.v.notna() & d.period_end.notna()]
    d["pm"] = d.period_end.dt.to_period("M")
    rows = []
    flow = d[(d.PARAMETER_CODE == "50050") & d.VALUE_TYPE_CODE.isin(["Q1", "Q2"])]
    for vt, name in [("Q1", "flow_avg"), ("Q2", "flow_max")]:
        g = flow[flow.VALUE_TYPE_CODE == vt].groupby(["EXTERNAL_PERMIT_NMBR", "pm"]).v.max().rename(name)
        rows.append(g)
    infl = d[(d.MONITORING_LOCATION_CODE == "G") & (d.VALUE_TYPE_CODE == "C2")]
    for p, name in [("00310", "inf_BOD5"), ("80082", "inf_CBOD5"), ("00530", "inf_TSS"), ("00610", "inf_NH3N")]:
        g = infl[infl.PARAMETER_CODE == p].groupby(["EXTERNAL_PERMIT_NMBR", "pm"]).v.median().rename(name)
        rows.append(g)
    return pd.concat(rows, axis=1).reset_index()


def permit_attributes():
    p = pd.read_csv(ROOT / "icis" / "ICIS_PERMITS.csv", dtype=str, encoding="latin-1",
                    usecols=["EXTERNAL_PERMIT_NMBR", "VERSION_NMBR", "FACILITY_TYPE_INDICATOR", "PERMIT_TYPE_CODE",
                             "MAJOR_MINOR_STATUS_FLAG", "PERMIT_STATUS_CODE", "TOTAL_DESIGN_FLOW_NMBR",
                             "ACTUAL_AVERAGE_FLOW_NMBR", "EDMR_AUTHORIZATION_FLAG", "PRETREATMENT_INDICATOR_CODE",
                             "ORIGINAL_ISSUE_DATE", "RAD_WBD_HUC12S", "PERMIT_NAME"])
    p = p[p.FACILITY_TYPE_INDICATOR == "POTW"].copy()
    p["VERSION_NMBR"] = num(p.VERSION_NMBR)
    p = p.sort_values("VERSION_NMBR").drop_duplicates("EXTERNAL_PERMIT_NMBR", keep="first")
    f = pd.read_csv(ROOT / "icis" / "ICIS_FACILITIES.csv", dtype=str, encoding="latin-1",
                    usecols=["NPDES_ID", "STATE_CODE", "COUNTY_CODE", "GEOCODE_LATITUDE", "GEOCODE_LONGITUDE",
                             "IMPAIRED_WATERS"])
    p = p.merge(f, left_on="EXTERNAL_PERMIT_NMBR", right_on="NPDES_ID", how="left")
    out = pd.DataFrame({
        "permit": p.EXTERNAL_PERMIT_NMBR, "state": p.STATE_CODE.fillna(p.EXTERNAL_PERMIT_NMBR.str[:2]),
        "permit_type": p.PERMIT_TYPE_CODE, "major": (p.MAJOR_MINOR_STATUS_FLAG == "M").astype(int),
        "status": p.PERMIT_STATUS_CODE, "design_flow": num(p.TOTAL_DESIGN_FLOW_NMBR),
        "actual_flow": num(p.ACTUAL_AVERAGE_FLOW_NMBR), "edmr": p.EDMR_AUTHORIZATION_FLAG,
        "pretreat": p.PRETREATMENT_INDICATOR_CODE, "orig_issue": dt(p.ORIGINAL_ISSUE_DATE),
        "huc12": p.RAD_WBD_HUC12S, "lat": num(p.GEOCODE_LATITUDE), "lon": num(p.GEOCODE_LONGITUDE),
        "impaired": p.IMPAIRED_WATERS, "name": p.PERMIT_NAME,
    })
    return out


if __name__ == "__main__":
    files = sorted(SRC.glob("dmr_potw_fy*.parquet"))
    eff = pd.concat([effluent_year(f) for f in files], ignore_index=True)
    # the same period can appear in two fiscal-year files (late or revised reports); keep the last file
    n0 = len(eff)
    key = pd.util.hash_pandas_object(eff[["permit", "outfall", "limset", "param", "statbase", "vtype", "period_end"]], index=False)
    eff = eff.loc[~key.duplicated(keep="last").values].reset_index(drop=True)
    del key
    eff.to_parquet(OUT / "effluent_panel.parquet", index=False)
    print("effluent rows", n0, "->", len(eff), "permits", eff.permit.nunique())
    cov = pd.concat([covariates_year(f) for f in files], ignore_index=True)
    cov = cov.groupby(["EXTERNAL_PERMIT_NMBR", "pm"]).max().reset_index().rename(columns={"EXTERNAL_PERMIT_NMBR": "permit"})
    cov["pm"] = cov.pm.dt.to_timestamp("M")
    cov.to_parquet(OUT / "monthly_covariates.parquet", index=False)
    print("covariate permit-months", len(cov))
    pa = permit_attributes()
    pa.to_parquet(OUT / "permit_attributes.parquet", index=False)
    print("permits with attributes", len(pa))
