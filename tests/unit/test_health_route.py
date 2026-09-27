from types import SimpleNamespace

from fastapi.responses import JSONResponse

from api.core.fast_intent_parser import FastIntentParser
from api.routes.health import healthz


def test_healthz_returns_ok(monkeypatch):
    monkeypatch.setattr(
        FastIntentParser, "get_instance", lambda: SimpleNamespace(gliner_failed=False)
    )
    assert healthz() == {"status": "ok"}


def test_healthz_reports_degraded_intent_runtime(monkeypatch):
    monkeypatch.setattr(
        FastIntentParser, "get_instance", lambda: SimpleNamespace(gliner_failed=True)
    )

    response = healthz()

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
