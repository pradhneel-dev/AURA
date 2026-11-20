"""Database engine/session wiring for SQLite development and PostgreSQL deployment."""
import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


def database_url() -> str:
    return os.getenv("AURA_DATABASE_URL", "sqlite:///./data/aura.db")


def build_engine(url: str | None = None):
    selected = url or database_url()
    if selected.startswith("sqlite:///") and ":memory:" not in selected:
        path = selected.removeprefix("sqlite:///")
        db_path = Path(path)
        if not db_path.is_absolute(): db_path = Path.cwd() / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        selected = f"sqlite:///{db_path}"
    options = {"pool_pre_ping": True}
    if selected.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False}
    engine = create_engine(selected, **options)
    if selected.startswith("sqlite:"):
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    from backend.database.models import Base
    Base.metadata.create_all(bind=engine)
