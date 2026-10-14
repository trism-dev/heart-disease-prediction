import json

import joblib
import numpy as np
import pytest

from src.config import METADATA_PATH, PIPELINE_PATH
from src.inference import predict as original_predict
from tests.test_app_inference import PROFILES
from vercel_app.app import app
from vercel_app.scoring import ROOT


@pytest.fixture
def client():
    return app.test_client()


def test_deployment_artifacts_match_canonical():
    assert (ROOT / "models/pipeline.joblib").read_bytes() == PIPELINE_PATH.read_bytes()
    assert (ROOT / "models/metadata.json").read_bytes() == METADATA_PATH.read_bytes()


@pytest.mark.parametrize("profile", PROFILES + [{**PROFILES[0], "ca": None, "thal": None}])
def test_api_matches_original_inference(client, profile):
    meta = json.loads(METADATA_PATH.read_text())
    bundle = {
        "meta": meta,
        "pipeline": joblib.load(PIPELINE_PATH),
        "threshold": meta["threshold"],
        "platt": meta["platt"],
    }
    expected = original_predict(bundle, profile)
    response = client.post("/api/predict", json=profile)
    assert response.status_code == 200
    actual = response.get_json()
    for key in ("probability", "raw_probability", "threshold", "raw_threshold"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert actual["elevated"] == expected["elevated"]
    assert actual["label"] == expected["label"]
    assert actual["contributions"] == pytest.approx(expected["contributions"], abs=1e-12)
    assert [p[0] for p in actual["top"]] == [p[0] for p in expected["top"]]
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "changes",
    [
        {"age": 28},
        {"sex": 3},
        {"thal": 0},
        {"oldpeak": -1},
        {"chol": None},
        {"sex": True},
        {"age": np.nan},
        {"ca": 2.5},
    ],
)
def test_rejects_invalid_inputs(client, changes):
    response = client.post("/api/predict", json={**PROFILES[0], **changes})
    assert response.status_code == 400
    assert next(iter(changes)) in response.get_json()["errors"]


def test_rejects_missing_extra_and_malformed_inputs(client):
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", json={**PROFILES[0], "extra": 1}).status_code == 400
    assert (
        client.post("/api/predict", data="bad", content_type="application/json").status_code == 400
    )
    assert (
        client.post("/api/predict", data="x" * 9000, content_type="application/json").status_code
        == 413
    )


def test_metadata_has_ranges_defaults_and_no_private_runtime_configuration(client):
    response = client.get("/api/model")
    assert response.status_code == 200
    assert len(response.get_json()["input_ranges"]) == 13
    assert "DATABASE_URL" not in response.get_data(as_text=True)
