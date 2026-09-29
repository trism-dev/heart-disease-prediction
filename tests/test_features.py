import numpy as np

from src.config import FEATURES
from src.data import split
from src.features import build_preprocessor


def test_fit_on_train_only(clean):
    Xtr, Xte, _, _ = split(clean)
    prep = build_preprocessor().fit(Xtr)
    scaler = prep.named_transformers_["num"]["scale"]
    assert np.allclose(scaler.mean_[0], Xtr["age"].mean())
    assert not np.allclose(scaler.mean_[0], clean["age"].mean())


def test_no_nan_and_stable_columns(clean):
    Xtr, Xte, _, _ = split(clean)
    prep = build_preprocessor().fit(Xtr)
    a, b = prep.transform(Xtr), prep.transform(Xte)
    assert not np.isnan(a).any() and not np.isnan(b).any()
    assert (
        a.shape[1] == b.shape[1] == 6 + (4 + 3 + 3 + 3) + 3
    )  # num + onehot(cp,restecg,thal,slope) + bin


def test_unknown_category_and_nan_do_not_crash(clean):
    Xtr, _, _, _ = split(clean)
    prep = build_preprocessor().fit(Xtr)
    row = Xtr.iloc[[0]].copy()
    row["cp"] = 99.0
    row["ca"] = np.nan
    row["thal"] = np.nan
    out = prep.transform(row[FEATURES])
    assert not np.isnan(out).any()
