"""Frozen pipeline inference: no training, database, or prediction persistence."""

import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


@lru_cache(maxsize=1)
def load_bundle():
    meta = json.loads((ROOT / "models/metadata.json").read_text())
    pipeline = joblib.load(ROOT / "models/pipeline.joblib")
    if not hasattr(pipeline[-1], "coef_"):
        raise RuntimeError("This deployment requires the verified logistic-regression pipeline")
    return {"meta": meta, "pipeline": pipeline}


def validate_inputs(value):
    if not isinstance(value, dict):
        return {"request": "Send a JSON object containing the 13 input fields."}
    errors = {}
    extra = set(value) - set(FEATURES)
    if extra:
        errors["request"] = "Unexpected input fields."
    ranges = load_bundle()["meta"]["input_ranges"]
    for key in FEATURES:
        if key not in value:
            errors[key] = "This field is required."
            continue
        v = value[key]
        if v is None and key in {"ca", "thal"}:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors[key] = "Enter a finite number."
        elif key in CODES and v not in CODES[key]:
            errors[key] = "Select one of the listed options."
        elif not ranges[key][0] <= v <= ranges[key][1]:
            errors[key] = f"Use a value between {ranges[key][0]:g} and {ranges[key][1]:g}."
        elif key != "oldpeak" and not float(v).is_integer():
            errors[key] = "Use a whole number."
    return errors


def calibrated(p, meta):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(1 / (1 + np.exp(-(meta["platt"]["a"] * np.log(p / (1 - p)) + meta["platt"]["b"]))))


def predict(inputs):
    bundle = load_bundle()
    pipe, meta = bundle["pipeline"], bundle["meta"]
    frame = pd.DataFrame([{c: np.nan if inputs[c] is None else float(inputs[c]) for c in FEATURES}])
    raw = float(pipe.predict_proba(frame)[0, 1])
    transformed = pipe[:-1].transform(frame)[0] - np.asarray(meta["transformed_mean"])
    contributions = {}
    for name, z, coefficient in zip(
        pipe[:-1].get_feature_names_out(), transformed, pipe[-1].coef_[0]
    ):
        raw_name = name.split("__", 1)[1]
        feature = next(
            (f for f in FEATURES if raw_name == f or raw_name.startswith(f + "_")), raw_name
        )
        contributions[feature] = contributions.get(feature, 0.0) + float(z * coefficient)
    elevated = raw >= meta["threshold"]
    return {
        "probability": calibrated(raw, meta),
        "raw_probability": raw,
        "threshold": calibrated(meta["threshold"], meta),
        "raw_threshold": meta["threshold"],
        "elevated": elevated,
        "label": "Elevated risk — recommend clinical follow-up" if elevated else "Lower risk",
        "contributions": contributions,
        "top": sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3],
        "edge_fields": [
            c
            for c in ("age", "trestbps", "chol", "thalach", "oldpeak")
            if inputs[c] in meta["input_ranges"][c]
        ],
    }
