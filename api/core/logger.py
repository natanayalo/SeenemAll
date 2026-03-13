import time
import uuid
import logging
import json
from contextvars import ContextVar
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log_json(self, level: int, msg: str, **kwargs: Any) -> None:
        if not self.logger.isEnabledFor(level):
            return

        record = {"message": msg, "request_id": request_id_ctx.get(), **kwargs}
        # Dump as JSON for structured logging. We could integrate structlog fully here,
        # but stdlib json logging is enough for a lightweight setup.
        try:
            log_str = json.dumps(record, default=str)
        except Exception:
            # Fallback
            log_str = f"{msg} | {kwargs}"

        self.logger.log(level, log_str)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log_json(logging.INFO, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log_json(logging.DEBUG, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log_json(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log_json(logging.ERROR, msg, **kwargs)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx.set(req_id)

        start_time = time.perf_counter()
        # Define logging context early
        path = request.url.path
        method = request.method
        logger = StructuredLogger("api.request")
        status_code = 500  # Default fallback

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            status_code = response.status_code
            return response
        except Exception as e:
            # Capture status code from common FastAPI/Starlette exceptions if available
            status_code = getattr(e, "status_code", 500)
            raise e
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if not path.startswith("/health"):  # Ignore health check spam
                logger.info(
                    "Request complete",
                    method=method,
                    path=path,
                    latency_ms=round(elapsed_ms, 2),
                    status_code=status_code,
                )

            request_id_ctx.reset(token)
