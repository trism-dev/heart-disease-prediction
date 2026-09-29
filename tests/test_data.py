import pytest

from src.data import binarise, dedupe, split, validate


def test_raw_passes_validation(raw):
    validate(raw)
    assert raw["ca"].isna().sum() == 4 and raw["thal"].isna().sum() == 2


def test_label_direction(clean):
    """Catches the inverted Kaggle target: disease rows must have higher ca and oldpeak."""
    pos, neg = clean[clean.target == 1], clean[clean.target == 0]
    assert pos["ca"].mean() > neg["ca"].mean()
    assert pos["oldpeak"].mean() > neg["oldpeak"].mean()
    assert clean.target.sum() == 139


def test_rejects_unknown_code(raw):
    bad = raw.copy()
    bad.loc[0, "ca"] = 4  # the Kaggle fake code
    with pytest.raises(ValueError, match="ca"):
        validate(bad)
    bad = raw.copy()
    bad.loc[0, "thal"] = 0
    with pytest.raises(ValueError, match="thal"):
        validate(bad)


def test_rejects_wrong_rows_and_columns(raw):
    with pytest.raises(ValueError, match="rows"):
        validate(raw.iloc[:-1])
    with pytest.raises(ValueError, match="columns"):
        validate(raw.drop(columns="chol"))


def test_no_duplicates_after_cleaning(clean):
    assert not clean.duplicated().any()
    assert dedupe(clean.assign())[1] == 0


def test_binarise_drops_num(raw):
    out = binarise(raw)
    assert "num" not in out and set(out.target) == {0, 1}


def test_split_stratified_and_disjoint(clean):
    Xtr, Xte, ytr, yte = split(clean)
    assert abs(ytr.mean() - yte.mean()) < 0.03
    assert len(Xte) == round(len(clean) * 0.2) and not set(Xtr.index) & set(Xte.index)
