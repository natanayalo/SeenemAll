# Seen’emAll MVP Task Tracker

| # | Task | Status | Notes |
|---|------|---------|-------|
| **1** | Project scaffold (FastAPI + Docker + pgvector) | ✅ | Running `/healthz` |
| **2** | Database schema + migrations | ✅ | Alembic + pgvector ready |
| **3** | TMDB ETL ingestion | ✅ | `make etl-tmdb` |
| **4** | Embedding generator | ✅ | MiniLM-L6-v2 vectors |
| **5** | User history + profile vectors | ✅ | `/user/history` route works |
| **6** | Candidate ANN search | ✅ | pgvector cosine query functional |
| **7** | `/recommend` endpoint (core) | ✅ | Returns JSON titles/posters |
| **8** | Intent parser (LLM) | 🏗 | to parse mood/runtime/genre queries |
| **9** | Reranker + explanation | ⏳ | re-order & add natural language reasons |
| **10** | JustWatch ETL (streaming links) | ⏳ | availability per country/service |
| **11** | `/watch-link` resolver | ⏳ | redirect deep links |
| **12** | `/feedback` ingestion | ⏳ | collect click / impression data |
| **13** | Diversity & novelty logic | ⏳ | avoid repeats, inject serendipity |
| **14** | Minimal web UI | ⏳ | show cards + “watch” buttons |
| **15** | Prompt & eval fixtures | ⏳ | store prompt templates for testing |
| **16** | CI polish & deployment | ✅ | Docker Compose + Makefile done |
