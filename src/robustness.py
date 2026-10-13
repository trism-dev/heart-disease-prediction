"""Threshold selection, repeated CV, threshold stability and Platt (sigmoid) calibration.

Platt scaling is monotone, so it never changes who is flagged: recall and precision at the mapped
threshold are identical. It only changes the number shown to the user.
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import N_SPLITS, RECALL_TARGET, SEED

N_REPEATS = 5
N_BOOT_THR = 500
TARGETS = (0.85, 0.90, 0.95)


def choose_threshold(y, proba, target: float = RECALL_TARGET) -> float:
    """Highest threshold whose recall is >= target (positive if p >= t)."""
    y, proba = np.asarray(y), np.asarray(proba)
    u = np.unique(proba)
    rec = ((proba[None, :] >= u[:, None]) & (y[None, :] == 1)).sum(1) / max(int(y.sum()), 1)
    ok = u[rec >= target]
    return float(ok.max()) if len(ok) else 0.0


def summarize(vals) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}


def repeated_cv(pipe, X, y, n_repeats: int = N_REPEATS) -> dict:
    """Precision at recall >= target and the chosen threshold, over n_repeats x 5-fold OOF runs."""
    thrs, precs = [], []
    for r in range(n_repeats):
        cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + 1 + r)
        p = cross_val_predict(clone(pipe), X, y, cv=cv, method="predict_proba")[:, 1]
        t = choose_threshold(y, p)
        thrs.append(t)
        precs.append(float(precision_score(y, p >= t)))
    return {"precision_at_recall": summarize(precs), "threshold": summarize(thrs)}


def threshold_bootstrap(y, proba, n: int = N_BOOT_THR) -> dict:
    """5th / 50th / 95th percentile of the recall-target threshold over bootstrap resamples."""
    rng = np.random.default_rng(SEED)
    y, proba = np.asarray(y), np.asarray(proba)
    t = [
        choose_threshold(y[i], proba[i])
        for i in (rng.integers(0, len(y), len(y)) for _ in range(n))
        if y[i].sum()
    ]
    lo, mid, hi = np.percentile(t, [5, 50, 95])
    return {"p05": float(lo), "p50": float(mid), "p95": float(hi)}


def target_table(y, proba) -> dict:
    """What each recall target costs on the out-of-fold predictions."""
    y, proba = np.asarray(y), np.asarray(proba)
    out = {}
    for tgt in TARGETS:
        t = choose_threshold(y, proba, tgt)
        pred = proba >= t
        out[f"{tgt:.2f}"] = {
            "threshold": t,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "specificity": float(((~pred) & (y == 0)).sum() / (y == 0).sum()),
            "false_alarms": int((pred & (y == 0)).sum()),
        }
    return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y, proba) -> dict:
    """Sigmoid recalibration fitted on out-of-fold train probabilities."""
    lr = LogisticRegression(C=1e6).fit(_logit(proba)[:, None], np.asarray(y))
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(proba, platt: dict):
    """Map raw probabilities to recalibrated ones (monotone)."""
    return 1 / (1 + np.exp(-(platt["a"] * _logit(proba) + platt["b"])))
