"""Stream-extract POTW rows from the annual ECHO DMR ZIPs without unzipping to disk.

Filter: EXTERNAL_PERMIT_NMBR in the set of permits ever flagged FACILITY_TYPE_INDICATOR == 'POTW'
in ICIS_PERMITS.csv (any version). All parameters are kept; a reduced column set is written to
one parquet file per fiscal year under data/dmr_potw/.

Usage: python 01_extract_potw_dmr.py [year ...]   (default: all 2009-2026, run in parallel)
"""
import sys, io, zipfile, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq

from project_paths import RAW_ROOT as ROOT
PROJ = Path(__file__).resolve().parents[1]
ZIPDIR = ROOT / "annual_dmr"
OUT = PROJ / "data" / "dmr_potw"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = [
    "EXTERNAL_PERMIT_NMBR", "VERSION_NMBR", "PERM_FEATURE_NMBR", "PERM_FEATURE_TYPE_CODE",
    "LIMIT_SET_DESIGNATOR", "LIMIT_ID", "LIMIT_BEGIN_DATE", "LIMIT_END_DATE",
    "PARAMETER_CODE", "MONITORING_LOCATION_CODE", "STAY_TYPE_CODE",
    "LIMIT_VALUE_TYPE_CODE", "LIMIT_VALUE_NMBR", "LIMIT_UNIT_CODE", "STANDARD_UNIT_CODE",
    "LIMIT_VALUE_STANDARD_UNITS", "STATISTICAL_BASE_CODE", "STATISTICAL_BASE_TYPE_CODE",
    "LIMIT_VALUE_QUALIFIER_CODE", "OPTIONAL_MONITORING_FLAG", "LIMIT_SAMPLE_TYPE_CODE",
    "LIMIT_FREQ_OF_ANALYSIS_CODE", "LIMIT_TYPE_CODE", "DMR_EVENT_ID",
    "MONITORING_PERIOD_END_DATE", "DMR_SAMPLE_TYPE_CODE", "DMR_FREQ_OF_ANALYSIS_CODE",
    "REPORTED_EXCURSION_NMBR", "VALUE_TYPE_CODE", "DMR_VALUE_NMBR", "DMR_UNIT_CODE",
    "DMR_VALUE_STANDARD_UNITS", "DMR_VALUE_QUALIFIER_CODE", "VALUE_RECEIVED_DATE",
    "DAYS_LATE", "NODI_CODE", "EXCEEDENCE_PCT", "VIOLATION_CODE", "RNC_DETECTION_CODE",
]


def potw_permits():
    t = pacsv.read_csv(
        ROOT / "icis" / "ICIS_PERMITS.csv",
        read_options=pacsv.ReadOptions(encoding="latin1"),
        convert_options=pacsv.ConvertOptions(
            include_columns=["EXTERNAL_PERMIT_NMBR", "FACILITY_TYPE_INDICATOR"],
            column_types={"EXTERNAL_PERMIT_NMBR": pa.string(), "FACILITY_TYPE_INDICATOR": pa.string()},
        ),
    )
    m = pc.equal(t["FACILITY_TYPE_INDICATOR"], "POTW")
    return pa.array(sorted(set(t.filter(m)["EXTERNAL_PERMIT_NMBR"].to_pylist())))


def extract_year(year):
    t0 = time.time()
    permits = potw_permits()
    zp = ZIPDIR / f"npdes_dmrs_fy{year}.zip"
    z = zipfile.ZipFile(zp)
    name = z.namelist()[0]
    out = OUT / f"dmr_potw_fy{year}.parquet"
    tmp = out.with_suffix(".parquet.partial")
    n_in = n_out = 0
    writer = None
    with z.open(name) as raw:
        reader = pacsv.open_csv(
            raw,
            read_options=pacsv.ReadOptions(encoding="latin1", block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(
                include_columns=KEEP, column_types={c: pa.string() for c in KEEP}
            ),
        )
        for batch in reader:
            n_in += batch.num_rows
            tb = pa.Table.from_batches([batch])
            tb = tb.filter(pc.is_in(tb["EXTERNAL_PERMIT_NMBR"], value_set=permits))
            if tb.num_rows == 0:
                continue
            n_out += tb.num_rows
            if writer is None:
                writer = pq.ParquetWriter(tmp, tb.schema, compression="zstd")
            writer.write_table(tb)
    if writer is not None:
        writer.close()
        tmp.replace(out)
    info = dict(year=year, rows_in=n_in, rows_potw=n_out, seconds=round(time.time() - t0, 1),
                out=str(out.name), bytes=out.stat().st_size if out.exists() else 0)
    (OUT / f"dmr_potw_fy{year}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(range(2009, 2027))
    todo = [y for y in years if not (OUT / f"dmr_potw_fy{y}.parquet").exists()]
    with ProcessPoolExecutor(max_workers=min(6, len(todo) or 1)) as ex:
        res = list(ex.map(extract_year, todo))
    print("done", sum(r["rows_potw"] for r in res))
