---
description: Run the full test suite with coverage and cache isolation.
---
// turbo-all

1. Ensure the local environment is ready:
```bash
# Activate .venv if not already active
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Run all tests with coverage:
```bash
pytest --cov=api tests/
```

3. Check specific integration routes:
```bash
pytest tests/integration/test_routes.py
```

4. Verify timing-sensitive metrics tests:
```bash
pytest tests/unit/test_metrics.py
```
