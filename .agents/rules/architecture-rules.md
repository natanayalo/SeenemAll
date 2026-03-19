# Architecture and Coding Rules

Use these rules for all implementation tasks in SeenemAll.

## Architecture
- Keep domain logic in `api/core/`.
- Keep `api/routes/` thin: request validation, orchestration, and response shaping only.
- Treat recommendation ranking behavior as high-sensitivity logic. Any ranking-path change requires tests.
- Persist similarity-search state in PostgreSQL/pgvector, not ad-hoc in-memory stores.
- Do not hardcode evaluation-set titles, IDs, or exact benchmark phrasings into production recommendation logic.
- If evaluation-driven fixes are needed, express them as reusable semantic detectors or catalog-aware behaviors that should generalize to paraphrases and adjacent live queries.

## Coding Standards
- Prefer explicit type hints for all new functions.
- Keep I/O paths async where applicable (DB, network, provider calls).
- Prefer `pathlib` for filesystem path manipulation.
- Avoid silent behavior changes and dead flags.
- When adding query normalization or scoring heuristics, prefer semantic feature names over case-specific names.

## Testing Requirements
- Add or update tests for every behavior change.
- During implementation, run targeted tests for touched modules.
- Before final handoff or PR update, run the full test suite (`pytest`).
- For recommendation pipeline tests, isolate cache/stateful globals.
- Mock external LLM/network dependencies in unit tests.
- Maintain coverage at or above the project threshold.
- For evaluation-driven ranking fixes, add at least one paraphrase or adjacent-query test when practical so behavior is not pinned only to the benchmark wording.

## Workflow and CI
- Run `pre-commit run --all-files` before every commit.
- If pre-commit modifies files, re-stage and re-run pre-commit.
- Keep a clean working tree before finalizing.

## Approval and Git Operations
- Do not create a commit without explicit user approval.
- Do not push commits without explicit user approval.
- If a task is blocked by policy or permissions, report the blocker and ask for direction.
