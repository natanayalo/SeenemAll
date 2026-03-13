# Architecture & Coding Standards

This rule ensures that all AI agents maintain the "Seen’emAll" architectural integrity and coding quality.

## 🏗️ Architectural Philosophy
- **Agent-Oriented**: Partition logic into functional "agents" (ETL, Embedding, Candidate, Reranker, etc.).
- **Domain Layer**: Core logic must reside in `api/core/`. Routes in `api/routes/` should only orchestrate.
- **Persistence**: Use `pgvector` for similarity search. Avoid in-memory state that isn't backed by the DB or a dedicated cache.

## 🔄 Workflow & CI
- **Pre-commit**: You MUST run `pre-commit run --all-files` before every commit. This ensures linting, formatting (Black/Ruff), and type checking (Mypy) are valid before pushing.
- **Git State**: Maintain a clean working tree. If a pre-commit hook modifies files (like trailing-whitespace), stage those changes and re-run.

## 🐍 Coding Standards
- **PEP 8**: Follow standard Python formatting.
- **Async First**: Use `async/await` for all I/O bound operations (DB, LLM calls).
- **Type Hints**: All new functions must include type annotations (Pydantic models preferred for API data).
- **Path Handling**: Prefer `pathlib` for filesystem path manipulation to ensure cross-platform compatibility.

## 🧪 Testing Requirements
- **90% Coverage**: New code must include unit tests. Check coverage with `pytest --cov=api tests/`.
- **Isolation**: Use the `_clear_recommend_cache` fixture for any test involving the recommendation pipeline.
- **Mocking**: Mock external LLM providers (OpenAI/Gemini) in unit tests using the stubs in `conftest.py`.
