from api.routes.health import healthz


def test_healthz_returns_ok():
    assert healthz() == {"status": "ok"}
