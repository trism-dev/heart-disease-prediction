"""Read patients_raw -> validate -> binarise -> dedupe -> stratified split."""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from src.config import (
    EXPECTED_MISSING,
    EXPECTED_ROWS,
    FEATURES,
    NUM_CODES,
    SEED,
    VALID_CODES,
)
from src.db import PatientRaw, get_engine

RAW_COLS = FEATURES + ["num"]


def read_raw(engine=None) -> pd.DataFrame:
    """patients_raw as a DataFrame (feature columns + num)."""
    return pd.read_sql(select(PatientRaw), engine or get_engine())[RAW_COLS]


def validate(df: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS) -> None:
    """Raise on wrong row count, missing columns or unexpected codes. Never coerces."""
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")
    for col, ok in VALID_CODES.items():
        bad = set(df[col].dropna().unique()) - ok
        if bad:
            raise ValueError(f"unexpected codes in {col}: {sorted(bad)}")
    bad = set(df["num"].dropna().unique()) - NUM_CODES
    if bad or df["num"].isna().any():
        raise ValueError(f"unexpected/missing num: {sorted(bad)}")
    for col, n in EXPECTED_MISSING.items():
        if expected_rows is not None and df[col].isna().sum() != n:
            raise ValueError(f"{col}: expected {n} NULLs, got {df[col].isna().sum()}")
    other = df[FEATURES].isna().sum().drop(list(EXPECTED_MISSING))
    if other.any():
        raise ValueError(f"unexpected NULLs: {other[other > 0].to_dict()}")


def binarise(df: pd.DataFrame) -> pd.DataFrame:
    """target = (num > 0); drops num."""
    out = df.copy()
    out["target"] = (out["num"] > 0).astype(int)
    return out.drop(columns="num")


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows. Returns (frame, number removed)."""
    out = df.drop_duplicates().reset_index(drop=True)
    return out, len(df) - len(out)


def load_clean(engine=None, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Full path: read -> validate -> binarise -> dedupe."""
    raw = read_raw(engine)
    validate(raw, expected_rows)
    df, n = dedupe(binarise(raw))
    if n:
        print(f"dedupe: removed {n} duplicate rows")
    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified 80/20 split -> X_train, X_test, y_train, y_test."""
    X, y = df[FEATURES], df["target"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def data_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the sorted frame; ties a run to its exact rows."""
    cols = list(df.columns)
    canon = df.sort_values(cols).round(6).to_csv(index=False)
    return hashlib.sha256(canon.encode()).hexdigest()
