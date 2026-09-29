"""Model factories and RandomizedSearchCV spaces (parameters keyed for the 'clf' pipeline step)."""

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import SEED
from src.features import build_preprocessor

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def make_pipeline(name: str, y_train=None, **params) -> Pipeline:
    """Preprocessor + classifier. y_train is needed for XGBoost's scale_pos_weight."""
    if name == "logistic_regression":
        clf = LogisticRegression(
            class_weight="balanced", solver="liblinear", random_state=SEED, max_iter=1000
        )
    elif name == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif name == "xgboost":
        pos = int(np.sum(y_train))
        clf = XGBClassifier(
            scale_pos_weight=(len(y_train) - pos) / pos,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=1,
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)]).set_params(**params)


SEARCH_SPACES = {
    "logistic_regression": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__l1_ratio": [0, 1],  # 0 = l2, 1 = l1 (penalty= is deprecated in sklearn 1.8),
    },
    "random_forest": {
        "clf__n_estimators": randint(200, 801),
        "clf__max_depth": [3, 4, 5, 6, 7, 8, 9, 10, None],
        "clf__min_samples_leaf": randint(1, 11),
        "clf__max_features": ["sqrt", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": randint(100, 601),
        "clf__max_depth": randint(2, 7),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__reg_lambda": loguniform(0.1, 10),
    },
}
