# Observability and Telemetry Checklist

## Logging Checks
- Structured logs remain valid JSON.
- Request id is preserved end to end.
- Error fallback remains parseable.

## Metrics Checks
- New metric names follow existing naming style.
- Empty snapshots are semantically correct.
- Timer measurements are still emitted for key stages.

## Validation Commands
```bash
pre-commit run --all-files
pytest tests/unit/test_metrics.py tests/unit/test_logger.py
pytest tests/unit/test_recommend_route.py
```

## Report Template
- Logging updates: <summary>
- Metrics updates: <summary>
- Compatibility impact: <none / details>
- Tests run: <commands>
