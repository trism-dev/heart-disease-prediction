import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, PIPELINE_PATH
from src.inference import HIGH, LOW, load_bundle, predict, to_frame

PROFILES = [
    dict(
        age=63,
        sex=1,
        cp=1,
        trestbps=145,
        chol=233,
        fbs=1,
        restecg=2,
        thalach=150,
        exang=0,
        oldpeak=2.3,
        slope=3,
        ca=0,
        thal=6,
    ),
    dict(
        age=67,
        sex=1,
        cp=4,
        trestbps=160,
        chol=286,
        fbs=0,
        restecg=2,
        thalach=108,
        exang=1,
        oldpeak=1.5,
        slope=2,
        ca=3,
        thal=3,
    ),
    dict(
        age=41,
        sex=0,
        cp=2,
        trestbps=130,
        chol=204,
        fbs=0,
        restecg=2,
        thalach=172,
        exang=0,
        oldpeak=1.4,
        slope=1,
        ca=0,
        thal=3,
    ),
]


@pytest.fixture(scope="module")
def bundle():
    if not PIPELINE_PATH.exists():
        pytest.skip("run src.train and src.evaluate first")
    return load_bundle()


@pytest.mark.parametrize("profile", PROFILES)
def test_matches_pipeline(bundle, profile):
    ref = joblib.load(PIPELINE_PATH).predict_proba(pd.DataFrame([profile])[FEATURES])[0, 1]
    res = predict(bundle, profile)
    assert res["probability"] == pytest.approx(ref)
    assert res["label"] == (HIGH if ref >= bundle["threshold"] else LOW)
    assert len(res["top"]) == 3


def test_unknown_inputs_work(bundle):
    p = {**PROFILES[0], "ca": None, "thal": None}
    assert np.isnan(to_frame(p)[["ca", "thal"]]).all().all()
    assert 0 <= predict(bundle, p)["probability"] <= 1
