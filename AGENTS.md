# SeenemAll Codex Instructions

Use this file as the project instruction root for Codex.

## Core References
- Architecture rules: [./.agents/rules/architecture-rules.md](./.agents/rules/architecture-rules.md)
- Reusable workflows: [./.agents/workflows/](./.agents/workflows/)
- Skills: [./.agents/skills/](./.agents/skills/)

## Repo Snapshot
- API service: `api/` (FastAPI routes + core recommendation logic)
- ETL pipelines: `etl/` (TMDB, embeddings, JustWatch)
- Frontend: `frontend/`
- Tests: `tests/`

## Quick Commands
- Start stack: `make up`
- Migrate DB: `make migrate`
- Sync TMDB: `make etl-tmdb`
- Compute embeddings: `make embed`
- Sync streaming offers: `make etl-justwatch`
- Health check: `make health`
- Metrics endpoint: `make metrics`
- Recommendation debug endpoint: `make debug-rec`

## Engineering Guardrails
- Keep business logic in `api/core/`; keep routes in `api/routes/` orchestration-only.
- Keep recommendation changes test-backed, especially cache, mixer, MMR, and reranker behavior.
- Avoid silent behavior changes in ranking paths; document intent in tests.
- Prefer additive, reversible changes over broad refactors when fixing bugs.

## Domain-Specific Guidance
- API work:
  - Keep route handlers thin and push core behavior into `api/core/`.
  - Keep request/response schemas explicit and stable.
  - If recommendation behavior changes, run `pytest tests/unit/test_recommend_route.py tests/unit/test_metrics.py`.
  - If route contracts change, run integration route tests.
- ETL work:
  - Keep sync jobs idempotent when rerun.
  - Normalize upstream payloads before persisting.
  - Preserve stable identifiers and avoid duplicate writes.
  - Log failures with enough context for replay/debugging.
  - Run ETL unit tests for touched modules and verify API compatibility.

## Quality Gate
- Run `pre-commit run --all-files` before committing.
- Run `pytest` before opening or updating a PR.
- Keep coverage above the configured threshold (target: 90%+, enforced threshold in CI/pytest config).
