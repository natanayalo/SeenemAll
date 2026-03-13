# API Contract Safety Checklist

## Scope
- List endpoints touched.
- List request/response fields changed.
- Mark each change as additive or breaking.

## Validation Commands
```bash
pre-commit run --all-files
pytest tests/integration/test_routes.py
pytest tests/unit/
```

## Contract Checks
- Status code unchanged unless intentional.
- Response keys unchanged unless intentional.
- Pagination cursor behavior remains compatible.
- Error payload structure remains stable.

## Report Template
- Endpoints touched: <list>
- Breaking changes: <none or details>
- Tests run: <commands>
- Compatibility status: <compatible / breaking>
