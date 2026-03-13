# Architecture & Coding Standards

This rule ensures that all AI agents maintain the "Seen’emAll" architectural integrity and coding quality.

## 🏗️ Architectural Philosophy
- **Agent-Oriented**: Partition logic into functional "agents" (ETL, Embedding, Candidate, Reranker, etc.).
- **Domain Layer**: Core logic must reside in `api/core/`. Routes in `api/routes/` should only orchestrate.
- **Persistence**: Use `pgvector` for similarity search. Avoid in-memory state that isn't backed by the DB or a dedicated cache.

## 🐍 Coding Standards
- **PEP 8**: Follow standard Python formatting.
- **Async First**: Use `async/await` for all I/O bound operations (DB, LLM calls).
- **Type Hints**: All new functions must include type annotations (Pydantic models preferred for API data).
- **Path Handling**: Always use `os.path.join` or `.replace("\\", "/")` for cross-platform compatibility (Windows vs Unix).

## 🧪 Testing Requirements
- **90% Coverage**: New code must include unit tests. Check coverage with `pytest --cov=api tests/`.
- **Isolation**: Use the `_clear_recommend_cache` fixture for any test involving the recommendation pipeline.
- **Mocking**: Mock external LLM providers (OpenAI/Gemini) in unit tests using the stubs in `conftest.py`.
