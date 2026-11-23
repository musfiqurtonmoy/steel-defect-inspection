"""SQL schema for the dataset manifest, dataset versions and served predictions.

SQLite by default. Any SQLAlchemy URL works, e.g. postgresql+psycopg://...
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DatasetVersion(Base):
    """One frozen snapshot of the cleaned dataset and its split."""

    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # content hash
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    raw_dir: Mapped[str] = mapped_column(Text)
    n_images: Mapped[int] = mapped_column(Integer)
    n_rejected: Mapped[int] = mapped_column(Integer)
    classes: Mapped[str] = mapped_column(Text)  # comma-separated, sorted
    report_json: Mapped[str] = mapped_column(Text)


class ImageRecord(Base):
    """One raw file and the outcome of its quality checks."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str | None] = mapped_column(String(8), nullable=True)
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    reject_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)


class Prediction(Base):
    """One request served by the API. Used for monitoring."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)


def make_session(db_url: str):
    if db_url.startswith("sqlite:///"):
        Path(db_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
