"""SQLAlchemy engine, session factory and ORM models (patients_raw, model_runs, predictions)."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import DATABASE_URL

# JSONB on Postgres, plain JSON elsewhere
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class PatientRaw(Base):
    """UCI values as-is; '?' stored as NULL."""

    __tablename__ = "patients_raw"

    id: Mapped[int] = mapped_column(primary_key=True)
    age: Mapped[float] = mapped_column(Float)
    sex: Mapped[float] = mapped_column(Float)
    cp: Mapped[float] = mapped_column(Float)
    trestbps: Mapped[float] = mapped_column(Float)
    chol: Mapped[float] = mapped_column(Float)
    fbs: Mapped[float] = mapped_column(Float)
    restecg: Mapped[float] = mapped_column(Float)
    thalach: Mapped[float] = mapped_column(Float)
    exang: Mapped[float] = mapped_column(Float)
    oldpeak: Mapped[float] = mapped_column(Float)
    slope: Mapped[float] = mapped_column(Float)
    ca: Mapped[float | None] = mapped_column(Float)
    thal: Mapped[float | None] = mapped_column(Float)
    num: Mapped[int] = mapped_column(Integer)
    source_site: Mapped[str] = mapped_column(String(32), default="cleveland")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelRun(Base):
    """One row per model per training run; the app loads the is_active one."""

    __tablename__ = "model_runs"

    run_id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_name: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict] = mapped_column(JSONType)
    threshold: Mapped[float] = mapped_column(Float)
    cv_metrics: Mapped[dict] = mapped_column(JSONType)
    test_metrics: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    data_hash: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(String(256))
    lib_versions: Mapped[dict] = mapped_column(JSONType)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    """Optional log (LOG_PREDICTIONS=true). No identifiers, ever."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.run_id"))
    age: Mapped[float] = mapped_column(Float)
    sex: Mapped[float] = mapped_column(Float)
    cp: Mapped[float] = mapped_column(Float)
    trestbps: Mapped[float] = mapped_column(Float)
    chol: Mapped[float] = mapped_column(Float)
    fbs: Mapped[float] = mapped_column(Float)
    restecg: Mapped[float] = mapped_column(Float)
    thalach: Mapped[float] = mapped_column(Float)
    exang: Mapped[float] = mapped_column(Float)
    oldpeak: Mapped[float] = mapped_column(Float)
    slope: Mapped[float] = mapped_column(Float)
    ca: Mapped[float | None] = mapped_column(Float)
    thal: Mapped[float | None] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(64))


def get_engine(url: str = DATABASE_URL):
    return create_engine(url, future=True)


def init_db(engine=None):
    """Create all tables (idempotent)."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None) -> sessionmaker:
    return sessionmaker(engine or get_engine())
