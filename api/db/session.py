import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

_engine = None
_SessionLocal = None


def init_engine():
    global _engine, _SessionLocal
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@db:5432/reco")
    _engine = create_engine(db_url, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, future=True
    )


def get_engine():
    if _engine is None:
        init_engine()
    return _engine


def get_sessionmaker():
    if _SessionLocal is None:
        init_engine()
    return _SessionLocal


def get_db() -> Session:
    SessionLocal = get_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
