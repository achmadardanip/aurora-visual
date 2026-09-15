import time
from pathlib import Path

from sqlalchemy import JSON, Float, Integer, String, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"
    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner: Mapped[str] = mapped_column(String)
    revision: Mapped[int] = mapped_column(Integer)
    bundle: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class Snapshot(Base):
    __tablename__ = "snapshots"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(String, index=True)
    revision: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    bundle: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Media(Base):
    __tablename__ = "media"
    asset_id: Mapped[str] = mapped_column(String, primary_key=True)
    ref: Mapped[dict] = mapped_column(JSON)
    owners: Mapped[list] = mapped_column(JSON)
    transform: Mapped[dict] = mapped_column(JSON)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (UniqueConstraint("owner", "endpoint", "key", name="uq_job_idempotency"),)
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String)
    case_id: Mapped[str] = mapped_column(String, index=True)
    owner: Mapped[str] = mapped_column(String)
    endpoint: Mapped[str] = mapped_column(String, default="analyze")
    key: Mapped[str] = mapped_column(String)
    payload_hash: Mapped[str] = mapped_column(String)
    snapshot_hash: Mapped[str] = mapped_column(String)
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String, default="queued")
    progress: Mapped[str] = mapped_column(String, default="Menunggu worker")
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Audit(Base):
    __tablename__ = "audit"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(String, index=True)
    event: Mapped[str] = mapped_column(String)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class WorkerState(Base):
    __tablename__ = "worker_state"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    heartbeat: Mapped[float] = mapped_column(Float)


def database(data_dir: Path):
    engine = create_engine(
        f"sqlite:///{data_dir / 'aurora.db'}", connect_args={"check_same_thread": False, "timeout": 20}
    )

    @event.listens_for(engine, "connect")
    def pragmas(connection, _):
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=20000")

    return engine, sessionmaker(engine, expire_on_commit=False)
