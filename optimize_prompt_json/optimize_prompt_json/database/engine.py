"""Database engine and session setup."""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from optimize_prompt_json.database.models import Base

_engine = None
_SessionLocal = None


def init_db(db_url="sqlite:///optimize_prompt_json.db"):
    """Initialize the database engine and create all tables."""
    global _engine, _SessionLocal
    _engine = create_engine(
        db_url,
        echo=False,
        connect_args={"check_same_thread": False},
    )
    _SessionLocal = sessionmaker(bind=_engine)
    Base.metadata.create_all(bind=_engine)


def get_engine():
    """Return the current SQLAlchemy engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


def get_session():
    """Return a new database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
