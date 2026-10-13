"""Extra report sections: Brier scores, LR odds ratios, permutation importance, recall by sex."""

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score

from src.config import MODELS_DIR, SEED
from src.models import LABELS


def _row(label, y, pred) -> str:
    n, pos = len(y), int(np.sum(y))
    rec = recall_score(y, pred) if pos else float("nan")
    prec = precision_score(y, pred, zero_division=0)
    return f"| {label} | {n} | {pos} | {rec:.3f} | {prec:.3f} |"


def _table(header: str, rows: list[str]) -> str:
    cols = header.count("|") - 1
    return "\n".join([header, "|" + " --- |" * cols, *rows])


def extras_markdown(
    X_train, y_train, X_test, y_test, probs, cal_probs, thresholds, oof_blob, shipped
) -> str:
    """Markdown sections appended to reports/model_comparison.md."""
    out = []

    lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    names = lr[:-1].get_feature_names_out()
    coefs = lr[-1].coef_[0]
    rows = [
        f"| {names[i].split('__', 1)[1]} | {np.exp(coefs[i]):.2f} |"
        for i in np.argsort(-np.abs(coefs))
        if coefs[i] != 0
    ]
    out.append(
        "## Logistic regression odds ratios\n\n"
        "Per 1 SD for numeric features (age, trestbps, chol, thalach, oldpeak, ca), per category "
        "for one-hot features, per unit for binary features. Above 1 raises the odds of disease.\n\n"
        + _table("| Feature | Odds ratio |", rows)
        + "\n"
    )

    for n in ("random_forest", "xgboost"):
        pipe = joblib.load(MODELS_DIR / f"{n}.joblib")
        pi = permutation_importance(
            pipe, X_train, y_train, scoring="roc_auc", n_repeats=10, random_state=SEED
        )
        rows = [
            f"| {X_train.columns[i]} | {pi.importances_mean[i]:.3f} ± {pi.importances_std[i]:.3f} |"
            for i in np.argsort(-pi.importances_mean)[:5]
        ]
        out.append(
            f"## {LABELS[n]}: permutation importance (top 5)\n\n"
            "Drop in train ROC-AUC when the feature is shuffled.\n\n"
            + _table("| Feature | Importance |", rows)
            + "\n"
        )

    thr = thresholds[shipped]
    oof_pred = oof_blob["oof"][shipped] >= thr
    test_pred = probs[shipped] >= thr
    ytr = y_train.to_numpy()
    rows = []
    for code, label in ((1, "Male"), (0, "Female")):
        mt = (X_train["sex"] == code).to_numpy()
        ms = (X_test["sex"] == code).to_numpy()
        rows.append(_row(f"{label}, train OOF", ytr[mt], oof_pred[mt]))
        rows.append(_row(f"{label}, test", y_test[ms], test_pred[ms]))
    out.append(
        "## Subgroups: recall by sex (shipped model)\n\n"
        "One global threshold is used for both groups. Women are about a third of the data, so the "
        "test rows are tiny; the train OOF rows are the steadier view.\n\n"
        + _table("| Group | n | Disease | Recall | Precision |", rows)
        + "\n"
    )
    return "\n".join(out) + "\n"
