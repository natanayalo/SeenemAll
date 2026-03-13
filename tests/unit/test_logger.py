from __future__ import annotations

import json
from unittest.mock import Mock

from api.core import logger as logger_module


def test_structured_logger_fallback_remains_json(monkeypatch):
    structured_logger = logger_module.StructuredLogger("test.logger")
    sink = Mock()
    sink.isEnabledFor.return_value = True
    structured_logger.logger = sink

    original_dumps = logger_module.json.dumps
    state = {"calls": 0}

    def flaky_dumps(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise TypeError("boom")
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(logger_module.json, "dumps", flaky_dumps)

    structured_logger.info("message", payload={"x": 1})

    log_payload = sink.log.call_args.args[1]
    parsed = json.loads(log_payload)
    assert parsed["message"] == "message"
    assert parsed["error"] == "Failed to serialize full record"
    assert "payload" in parsed["kwargs_repr"]
