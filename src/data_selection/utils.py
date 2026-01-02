import json
from pathlib import Path
import pandas as pd
import numpy as np

def project_root_from(__file_path: str) -> Path:
    return Path(__file_path).resolve().parents[2]

def load_dataset(excel_path: Path) -> pd.DataFrame:
    """
    Excel has the real header on row 1 , so header=1.
    """
    df = pd.read_excel(excel_path, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _jsonable(obj):
    # numpy/pandas scalars into Python scalars
    try:

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass

    # handle containers
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]

    # paths
    if isinstance(obj, Path):
        return str(obj)

    return obj

def save_meta(path: Path, meta: dict):
    meta = _jsonable(meta)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")