"""Sequential production fits; no fitting is started on import."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    driver=Path(__file__).with_name('05u_fit_cliff_stack.py')
    for parameter in ['BOD5','CBOD5','TSS','NH3N','TP','FecalColi','Ecoli','TRC']:
        for field in ['C2','C3']:
            subprocess.run([sys.executable,str(driver),'--param',parameter,'--vtype',field],check=True)
