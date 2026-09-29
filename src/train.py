"""CV-tune 3 models on train only, pick thresholds on OOF probabilities, save artifacts + model_runs."""

import json
import platform
import sys

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.config import (
    MODELS_DIR,
    N_ITER,
    N_SPLITS,
    PRECISION_FLOOR,
    RECALL_TARGET,
    SEED,
)
from src.data import data_hash, load_clean, split
from src.db import ModelRun, get_engine, get_session_factory, init_db
from src.models import MODEL_NAMES, SEARCH_SPACES, make_pipeline


def lib_versions() -> dict:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
    }


def _refit(cv_results) -> int:
    """Best mean recall among candidates with mean precision >= floor (else best recall)."""
    rec = np.asarray(cv_results["mean_test_recall"])
    prec = np.asarray(cv_results["mean_test_precision"])
    ok = prec >= PRECISION_FLOOR
    if not ok.any():
        print(f"  warning: no candidate reaches precision >= {PRECISION_FLOOR}; using best recall")
        return int(np.argmax(rec))
    score = np.where(ok, rec + 1e-6 * prec, -1.0)  # ties -> higher precision
    return int(np.argmax(score))


def choose_threshold(y, proba, target: float = RECALL_TARGET) -> float:
    """Highest threshold whose recall is >= target (positive if p >= t)."""
    y = np.asarray(y)
    best = 0.0
    for t in np.unique(proba):
        if recall_score(y, proba >= t) >= target:
            best = max(best, float(t))
    return best


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def summarize(vals) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


def select_model(runs: dict) -> tuple[str, str]:
    """LR is the bar; a tree model ships only if CV recall beats LR by > 1 CV std at comparable precision."""
    lr = runs["logistic_regression"]["search"]
    winner, reason = (
        "logistic_regression",
        "no tree model beat LR by more than one CV std; ties go to LR",
    )
    best_gain = 0.0
    for name in ("random_forest", "xgboost"):
        s = runs[name]["search"]
        gain = s["recall"]["mean"] - lr["recall"]["mean"]
        comparable = s["precision"]["mean"] >= lr["precision"]["mean"] - 0.05
        if gain > lr["recall"]["std"] and comparable and gain > best_gain:
            winner, best_gain = name, gain
            reason = (
                f"{name} CV recall {s['recall']['mean']:.3f} beats LR {lr['recall']['mean']:.3f} "
                f"by more than one LR CV std ({lr['recall']['std']:.3f}) at comparable precision"
            )
    return winner, reason


def main() -> None:
    engine = init_db(get_engine())
    df = load_clean(engine)
    dhash = data_hash(df)
    X_train, _, y_train, _ = split(df)  # test set is not touched here
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    MODELS_DIR.mkdir(exist_ok=True)

    runs, oof = {}, {}
    for name in MODEL_NAMES:
        print(f"tuning {name} ...")
        search = RandomizedSearchCV(
            make_pipeline(name, y_train),
            SEARCH_SPACES[name],
            n_iter=N_ITER,
            cv=cv,
            scoring={"recall": "recall", "precision": "precision"},
            refit=_refit,
            random_state=SEED,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        r = search.cv_results_
        i = search.best_index_
        proba = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[
            :, 1
        ]
        thr = choose_threshold(y_train, proba)
        folds = {"recall": [], "precision": [], "f2": []}
        for _, val in cv.split(X_train, y_train):
            yv, pv = y_train.iloc[val], proba[val] >= thr
            folds["recall"].append(recall_score(yv, pv))
            folds["precision"].append(precision_score(yv, pv, zero_division=0))
            folds["f2"].append(fbeta_score(yv, pv, beta=2))
        cv_metrics = {
            "search": {
                "recall": {
                    "mean": float(r["mean_test_recall"][i]),
                    "std": float(r["std_test_recall"][i]),
                },
                "precision": {
                    "mean": float(r["mean_test_precision"][i]),
                    "std": float(r["std_test_precision"][i]),
                },
            },
            "at_threshold": {k: summarize(v) for k, v in folds.items()},
            "oof_recall": float(recall_score(y_train, proba >= thr)),
            "oof_precision": float(precision_score(y_train, proba >= thr)),
        }
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(best, path)
        oof[name] = proba
        runs[name] = {
            "search": cv_metrics["search"],
            "row": dict(
                model_name=name,
                threshold=thr,
                cv_metrics=cv_metrics,
                data_hash=dhash,
                params={k: _jsonable(v) for k, v in search.best_params_.items()},
                artifact_path=str(path.relative_to(MODELS_DIR.parent)),
                lib_versions=lib_versions(),
                is_active=False,
            ),
        }
        print(
            f"  cv recall {cv_metrics['search']['recall']['mean']:.3f} "
            f"precision {cv_metrics['search']['precision']['mean']:.3f} | "
            f"threshold {thr:.3f} -> OOF recall {cv_metrics['oof_recall']:.3f} "
            f"precision {cv_metrics['oof_precision']:.3f}"
        )

    winner, reason = select_model(runs)
    with get_session_factory(engine)() as s:
        objs = {n: ModelRun(**runs[n]["row"]) for n in MODEL_NAMES}
        s.add_all(objs.values())
        s.commit()
        run_ids = {n: o.run_id for n, o in objs.items()}
    joblib.dump({"oof": oof, "y_train": y_train.to_numpy()}, MODELS_DIR / "oof.joblib")
    (MODELS_DIR / "selection.json").write_text(
        json.dumps(
            {"selected": winner, "reason": reason, "run_ids": run_ids, "data_hash": dhash}, indent=2
        )
    )
    print(f"selected: {winner} ({reason})")


if __name__ == "__main__":
    sys.exit(main())
