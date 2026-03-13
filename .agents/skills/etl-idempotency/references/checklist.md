# ETL Idempotency Checklist

## Idempotency Checks
- Run the same ETL step twice; verify row counts are stable.
- Verify duplicates are not introduced for existing records.
- Verify updates overwrite expected fields only.

## Failure-Handling Checks
- Simulate an upstream/API failure path.
- Confirm rerun continues safely after failure.

## Validation Commands
```bash
pre-commit run --all-files
pytest tests/
make etl-tmdb
make etl-justwatch
```

## Report Template
- Job touched: <name>
- Idempotency result: <pass/fail>
- Failure-recovery result: <pass/fail>
- Notes: <details>
