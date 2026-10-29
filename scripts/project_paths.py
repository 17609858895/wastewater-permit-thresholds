"""Local raw-data root; derived files remain under this repository."""
import os
from pathlib import Path
RAW_ROOT = Path(os.environ.get("ECHO_RAW_ROOT", Path(__file__).resolve().parents[1] / "raw_data")).expanduser().resolve()
