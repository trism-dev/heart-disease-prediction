"""UCI Cleveland file -> patients_raw. Idempotent: skips when row count and hash match."""

import hashlib

import pandas as pd
import requests
from sqlalchemy import delete, select

from src.config import FEATURES, RAW_FILE, RAW_URL
from src.db import PatientRaw, get_engine, get_session_factory, init_db


def _hash(df: pd.DataFrame) -> str:
    cols = FEATURES + ["num"]
    canon = df[cols].sort_values(cols).round(6).to_csv(index=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def load_raw_file() -> pd.DataFrame:
    """Read the untouched UCI file ('?' -> NaN), downloading it first if absent."""
    if not RAW_FILE.exists():
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(RAW_URL, timeout=30)
        r.raise_for_status()
        RAW_FILE.write_bytes(r.content)
    return pd.read_csv(RAW_FILE, header=None, names=FEATURES + ["num"], na_values="?")


def ingest(engine=None) -> str:
    """Load patients_raw. Returns 'skipped' or 'loaded'."""
    engine = init_db(engine or get_engine())
    df = load_raw_file()
    with get_session_factory(engine)() as s:
        existing = pd.read_sql(select(PatientRaw), s.connection())
        if len(existing) == len(df) and _hash(existing) == _hash(df):
            return "skipped"
        s.execute(delete(PatientRaw))  # delete + insert commit together
        rows = df.astype(object).where(df.notna(), None).to_dict("records")
        s.add_all(PatientRaw(**r) for r in rows)
        s.commit()
    return "loaded"


if __name__ == "__main__":
    print(f"ingest: {ingest()}")
