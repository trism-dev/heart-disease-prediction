"""External validation: score the frozen shipped model on the other three UCI sites.

These rows never touch training or threshold selection. They are read straight from the UCI files
(not Postgres) because patients_raw is the Cleveland training table.
"""

import json

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FEATURES, METADATA_PATH, PIPELINE_PATH, RAW_FILE, REPORTS_DIR

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SITES = {
    "Hungarian": "processed.hungarian.data",
    "Swiss": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
IMPUTED = ["ca", "thal", "slope", "chol", "fbs", "trestbps"]


def load_site(fname: str) -> pd.DataFrame:
    """One UCI site file, '?' -> NaN, with impossible zeros (chol/trestbps = 0) set to NaN."""
    path = RAW_FILE.parent / fname
    if not path.exists():
        r = requests.get(BASE + fname, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path, header=None, names=FEATURES + ["num"], na_values="?")
    for c in ("chol", "trestbps"):
        df.loc[df[c] == 0, c] = np.nan
    return df


def evaluate_site(pipe, thr: float, df: pd.DataFrame) -> dict:
    y = (df["num"] > 0).astype(int).to_numpy()
    # binary columns are passthrough in the pipeline (spec), so fill their gaps with the Cleveland
    # training mode here; everything else is imputed inside the pipeline
    X = df[FEATURES].copy()
    X[["sex", "fbs", "exang"]] = X[["sex", "fbs", "exang"]].fillna(0)
    p = pipe.predict_proba(X)[:, 1]
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(df),
        "pos": int(y.sum()),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "fn": int(fn),
        "fp": int(fp),
        "missing": {c: float(df[c].isna().mean()) for c in IMPUTED},
    }


def main() -> None:
    meta = json.loads(METADATA_PATH.read_text())
    pipe, thr = joblib.load(PIPELINE_PATH), meta["threshold"]
    frames = {name: load_site(f) for name, f in SITES.items()}
    res = {name: evaluate_site(pipe, thr, df) for name, df in frames.items()}
    res["All three pooled"] = evaluate_site(
        pipe, thr, pd.concat(frames.values(), ignore_index=True)
    )

    rows = [
        f"| {k} | {r['n']} | {r['pos']} ({r['pos'] / r['n']:.0%}) | {r['recall']:.3f} | "
        f"{r['precision']:.3f} | {r['specificity']:.3f} | {r['auc']:.3f} | {r['fn']} |"
        for k, r in res.items()
    ]
    miss = [
        f"| {k} | " + " | ".join(f"{r['missing'][c]:.0%}" for c in IMPUTED) + " |"
        for k, r in res.items()
    ]
    md = f"""# External validation

The frozen shipped model ({meta["model_name"].replace("_", " ")}, threshold {thr:.3f}) scored on the
other three UCI sites. None of these rows were used for training, tuning or the threshold.

| Site | n | Disease | Recall | Precision | Specificity | ROC-AUC | Missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## How to read this

- **Prevalence differs a lot by site** (Swiss and VA are mostly disease-positive), so precision and
  specificity are not comparable with the Cleveland test set. Recall and ROC-AUC travel better.
- **Heavy missingness.** Share of rows with the value missing (or an impossible zero for `chol` and
  `trestbps`). The pipeline fills these with Cleveland's most frequent value (`fbs` and `exang` gaps are
  filled with 0 outside the pipeline, because binary columns are passthrough). That pulls scores
  toward the typical Cleveland patient, and for `ca` and `thal` it can hide exactly the signal the
  model relies on. Treat these as a stress test, not a benchmark.

| Site | ca | thal | slope | chol | fbs | trestbps |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(miss)}

- Different clinics code and measure things differently, and the cohorts are small. No claim of
  clinical generalisation follows from this.
"""
    (REPORTS_DIR / "external_validation.md").write_text(md, encoding="utf-8")
    for k, r in res.items():
        print(
            f"{k}: n={r['n']} recall={r['recall']:.3f} precision={r['precision']:.3f} "
            f"auc={r['auc']:.3f}"
        )


if __name__ == "__main__":
    main()
