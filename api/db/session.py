import os
import time
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

_engine = None
_SessionLocal = None
_query_logging_attached = False
_sql_logger = logging.getLogger("api.db.sql")


def _format_statement(statement: str, *, max_length: int = 120) -> str:
    condensed = " ".join(statement.strip().split())
    if len(condensed) <= max_length:
        return condensed
    return condensed[: max_length - 1] + "…"


def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.perf_counter()


def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    if not _sql_logger.isEnabledFor(logging.INFO):
        return
    start = getattr(context, "_query_start_time", None)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 if start else None
    summary = _format_statement(statement)
    rowcount = cursor.rowcount if cursor.rowcount is not None else "?"
    prefix = summary.split(" ", 1)[0].upper() if summary else "SQL"
    if elapsed_ms is not None:
        _sql_logger.info(
            "%s query took %.1f ms | rows=%s | %s",
            prefix,
            elapsed_ms,
            rowcount,
            summary,
        )
    else:
        _sql_logger.info("%s query | rows=%s | %s", prefix, rowcount, summary)


def _handle_error(context):
    if not _sql_logger.isEnabledFor(logging.WARNING):
        return
    statement = getattr(context, "statement", "") or ""
    summary = _format_statement(statement)
    _sql_logger.warning(
        "SQL error during '%s': %s", summary, context.original_exception
    )


def _attach_sql_logging(engine):
    global _query_logging_attached
    if _query_logging_attached:
        return
    try:
        event.listen(engine, "before_cursor_execute", _before_cursor_execute)
        event.listen(engine, "after_cursor_execute", _after_cursor_execute)
        event.listen(engine, "handle_error", _handle_error)
    except InvalidRequestError:
        # Tests may stub create_engine with a simple placeholder; skip wiring in that case.
        return
    _query_logging_attached = True


def init_engine():
    global _engine, _SessionLocal
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@db:5432/reco")
    _engine = create_engine(db_url, pool_pre_ping=True, future=True)
    _attach_sql_logging(_engine)
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
