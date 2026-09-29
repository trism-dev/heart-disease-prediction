import pytest
from sqlalchemy import func, select, text

from src.data import read_raw
from src.db import ModelRun, PatientRaw, get_engine, get_session_factory, init_db
from src.ingest import ingest

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def engine():
    e = init_db(get_engine())
    ingest(e)
    return e


def test_ingest_303_rows_6_nulls(engine):
    df = read_raw(engine)
    assert len(df) == 303
    assert df["ca"].isna().sum() == 4 and df["thal"].isna().sum() == 2


def test_ingest_rerun_is_noop(engine):
    assert ingest(engine) == "skipped"
    with get_session_factory(engine)() as s:
        assert s.scalar(select(func.count()).select_from(PatientRaw)) == 303


def test_model_runs_jsonb_roundtrip_and_single_active(engine):
    payload = {"a": [1, 2.5, None], "b": {"c": "d"}}
    with get_session_factory(engine)() as s:
        run = ModelRun(
            model_name="test",
            params=payload,
            threshold=0.4,
            cv_metrics=payload,
            data_hash="x",
            artifact_path="x",
            lib_versions={},
            is_active=False,
        )
        s.add(run)
        s.commit()
        rid = run.run_id
    try:
        with get_session_factory(engine)() as s:
            assert s.get(ModelRun, rid).params == payload
            assert (
                s.scalar(select(func.count()).select_from(ModelRun).where(ModelRun.is_active)) == 1
            )
    finally:
        with get_session_factory(engine)() as s:
            s.execute(text("DELETE FROM model_runs WHERE run_id = :i"), {"i": rid})
            s.commit()
