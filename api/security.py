from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    """
    Simple API key auth for local deployments.

    - Enable with API_AUTH_ENABLED=1 (default).
    - Set API_AUTH_KEY to a non-empty secret value.
    - Send it via the X-API-Key header.
    """

    if not _env_flag("API_AUTH_ENABLED", default="1"):
        return

    expected = (os.getenv("API_AUTH_KEY") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API auth enabled but API_AUTH_KEY is not set.",
        )

    provided = (x_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
