# 🤖 Gemini Agent Guidelines

> This file contains critical instructions and best practices for AI agents working on the Seen’emAll project. Read this before starting any task.

---

## 🏗️ Architectural Philosophy
- **Agent-Oriented**: The system is partitioned into functional "agents" (ETL, Embedding, Candidate, Reranker, etc.). Logic resides in `api/core/`, while `api/routes/` acts as the orchestration layer.
- **Data-Driven**: Recommendations rely on pgvector similarity blended with collaborative and trending signals.
- **Observable**: Every request is traceable via `request_id`. Diagnostics and metrics are first-class citizens.

---

## 🛠️ Environment & Workflow
- **Docker First**: The full stack runs via `docker compose up -d`. Use `--build` after frontend changes.
- **Local Dev**: Use a `.venv` for faster linting and testing. Reinstall dependencies from `requirements.txt` often.
- **Makefile**: Always check the `Makefile` for shortcuts (`make migrate`, `make health`, `make metrics`, `make debug-rec`).

---

## 🧪 Testing & Quality
- **90% Coverage**: We maintain high coverage. New features MUST include unit and integration tests.
- **Cache Isolation**: Use the `_clear_recommend_cache` fixture (defined in `conftest.py`) in tests to ensure results aren't polluted by previous runs.
- **Windows Sensitivity**: Always use `.replace("\\", "/")` or `os.path.join` for path handling to avoid cross-platform failures.
- **Torch Stubs**: In `conftest.py`, ensure the `torch` stub includes `torch.Tensor` and the `torch.cuda` submodule for scipy/sklearn compatibility.

---

## 📊 Diagnostics & Monitoring
- ** auditing**: Before debugging recommendation quality, check `GET /recommend/debug`. It exposes intent rewrites, candidate counts, and per-stage latencies.
- **Telemetry**: Use `GET /healthz/metrics` to verify that your changes aren't regressing pipeline performance.
- **Logging**: Use the structured `logger` in `api.core.logger` for all server-side events.

---

## 🚩 Common Pitfalls
- **Frontend URLs**: The React frontend requires `REACT_APP_API_URL` to be baked in at *build time*. If the API port changes, rebuild the frontend container.
- **Migration Lag**: If a migration fails with "locating revision", use `docker compose exec api alembic stamp <rev>` to align the DB with the codebase, or `docker compose down -v` to reset.
- **Model Loading**: First-request latencies (cold starts) can be high (>20s). This is normal behavior during embedding model initialization.

---

## 📂 Git Best Practices
- **Feature Branches**: Use `feature/` or `fix/` prefixes.
- **Commit Messages**: Follow conventional commits (`feat:`, `fix:`, `chore:`, `docs:`).
- **Track Documentation**: Always update `TASKS.md` and `README.md` when completing a major item.
