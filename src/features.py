"""Shared preprocessing: impute -> scale numerics, impute -> one-hot nominals, passthrough binaries."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BINARY, NOMINAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """Unfitted ColumnTransformer. Must only ever be fit on training data."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("scale", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("nom", nominal, NOMINAL), ("bin", "passthrough", BINARY)],
        sparse_threshold=0.0,
    )
