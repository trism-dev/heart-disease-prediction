"""Serving logic for the app: load the active run, score one patient, explain, optionally log."""

import json

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from src.config import FEATURES, LOG_PREDICTIONS, METADATA_PATH, PIPELINE_PATH, ROOT
from src.db import ModelRun, Prediction, get_engine, get_session_factory

HIGH = "Elevated risk — recommend clinical follow-up"
LOW = "Lower risk"


def load_bundle() -> dict:
    """Active model_runs row + its joblib; falls back to models/ files if Postgres is down."""
    meta = json.loads(METADATA_PATH.read_text())
    bundle = {
        "meta": meta,
        "run_id": meta["run_id"],
        "db_ok": False,
        "threshold": meta["threshold"],
        "test_metrics": meta["test_metrics"],
        "model_name": meta["model_name"],
        "path": PIPELINE_PATH,
    }
    try:
        with get_session_factory(get_engine())() as s:
            run = s.scalars(select(ModelRun).where(ModelRun.is_active)).first()
        if run is not None and run.test_metrics is not None:
            bundle.update(
                run_id=run.run_id,
                threshold=run.threshold,
                test_metrics=run.test_metrics,
                model_name=run.model_name,
                path=ROOT / run.artifact_path,
                db_ok=True,
            )
    except (
        Exception
    ):  # ponytail: any DB failure -> offline mode; narrow if it ever hides a real bug
        pass
    bundle["pipeline"] = joblib.load(bundle["path"])
    return bundle


def to_frame(inputs: dict) -> pd.DataFrame:
    """One-row frame in training column order; None -> NaN (the pipeline's imputer handles it)."""
    return pd.DataFrame(
        [{c: (np.nan if inputs.get(c) is None else float(inputs[c])) for c in FEATURES}]
    )


def _contributions(bundle: dict, X: pd.DataFrame) -> dict[str, float]:
    pipe, meta = bundle["pipeline"], bundle["meta"]
    clf = pipe[-1]
    if hasattr(clf, "coef_"):  # LR: coefficient x transformed value, summed per raw feature
        Z = pipe[:-1].transform(X)[0]
        names = pipe[:-1].get_feature_names_out()
        out: dict[str, float] = {}
        for n, z, w in zip(names, Z, clf.coef_[0]):
            raw = n.split("__", 1)[1]
            raw = next((f for f in FEATURES if raw == f or raw.startswith(f + "_")), raw)
            out[raw] = out.get(raw, 0.0) + float(z * w)
        return out
    # tree models: occlusion, replace one feature with its training reference value
    base = pipe.predict_proba(X)[0, 1]
    out = {}
    for c in FEATURES:
        Xo = X.copy()
        Xo[c] = meta["reference_values"][c]
        out[c] = float(base - pipe.predict_proba(Xo)[0, 1])
    return out


def predict(bundle: dict, inputs: dict) -> dict:
    """Probability, label, threshold and top-3 contributing raw features for one patient."""
    X = to_frame(inputs)
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    contrib = _contributions(bundle, X)
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    thr = bundle["threshold"]
    return {
        "probability": p,
        "threshold": thr,
        "elevated": p >= thr,
        "label": HIGH if p >= thr else LOW,
        "top": top,
    }


def log_prediction(bundle: dict, inputs: dict, result: dict) -> bool:
    """Write inputs + output to predictions (no identifiers). No-op unless enabled and DB is up."""
    if not (LOG_PREDICTIONS and bundle["db_ok"]):
        return False
    row = {c: inputs.get(c) for c in FEATURES}
    with get_session_factory(get_engine())() as s:
        s.add(
            Prediction(
                run_id=bundle["run_id"],
                probability=result["probability"],
                label=result["label"],
                **row,
            )
        )
        s.commit()
    return True
