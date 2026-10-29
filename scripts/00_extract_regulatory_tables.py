"""Select POTW inspection and single-event-violation rows from EPA CSVs."""
from pathlib import Path
import importlib.util
import pandas as pd
from project_paths import RAW_ROOT

if __name__ == '__main__':
    base=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('extract',Path(__file__).with_name('01_extract_potw_dmr.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    permits=set(module.potw_permits().to_pylist())
    for name in ['NPDES_INSPECTIONS','NPDES_SE_VIOLATIONS']:
        parts=[]
        for chunk in pd.read_csv(RAW_ROOT/'icis'/(name+'.csv'),dtype=str,encoding='latin-1',chunksize=200000,low_memory=False):
            parts.append(chunk.loc[chunk.NPDES_ID.isin(permits)])
        output=base/'data'/(name+'_potw.parquet');output.parent.mkdir(exist_ok=True)
        data=pd.concat(parts,ignore_index=True);data.to_parquet(output,index=False)
        print(name,len(data))
