# Seen'emAll Task Tracker

## Core Tasks

| # | Task | Status | Notes |
|---|------|---------|-------|
| **1** | Project scaffold (FastAPI + Docker + pgvector) | ✅ | Running `/healthz` |
| **2** | Database schema + migrations | ✅ | Alembic + pgvector ready |
| **3** | TMDB ETL ingestion | ✅ | `make etl-tmdb` |
| **4** | Embedding generator | ✅ | MiniLM-L6-v2 vectors |
| **5** | User history + profile vectors | ✅ | `/user/history` route works |
| **6** | Candidate ANN search | ✅ | pgvector cosine query functional |
| **7** | `/recommend` endpoint (core) | ✅ | Returns JSON titles/posters |
| **8** | Intent parser (LLM) | ✅ | parses mood/runtime/genre filters |
| **9** | Reranker + explanation | ✅ | LLM reranks candidates + inline rationales |
| **10** | JustWatch ETL (streaming links) | ✅ | `make etl-justwatch` fills availability |
| **11** | `/watch-link` resolver | ✅ | endpoint available |
| **12** | `/feedback` ingestion | ✅ | endpoint available |
| **13** | Diversity & novelty logic | ✅ | MMR diversification available via query param |
| **14** | Minimal web UI | ✅ | show cards + "watch" buttons |
| **15** | Prompt & eval fixtures | ✅ | Templates + eval suite for LLM interactions:
- Intent parser prompts
- Reranker prompts with examples
- Golden dataset of queries + expected results
- Evaluation metrics and thresholds
- Error case handling examples |
| **16** | CI polish & deployment | ✅ | Docker Compose + Makefile done |

## Quality Improvements

### Embedding Enhancement (Priority: High)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **17** | Add genres to embedding text | ⏳ | Prepend genre list to title+overview |
| **18** | Test embedding templates | ⏳ | Try different title/overview/genre formats |
| **19** | Evaluate domain models | ⏳ | Research MovieCLIP and alternatives |
| **20** | Add temporal context | ⏳ | Include year/decade in embeddings |
| **21** | A/B test framework | ⏳ | Compare embedding strategies |
| **22** | Version embeddings | ⏳ | Support migrations between versions |

### User Profile Enhancement (Priority: High)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **23** | Improve rating weights | ⏳ | Better weight for explicit feedback |
| **24** | Genre preferences | ⏳ | Add separate genre vector |
| **25** | Collaborative signals | ⏳ | Add similar users' signals |
| **26** | Configurable recency | ⏳ | Make time decay configurable |
| **27** | Negative feedback | ⏳ | Track and use "not interested" |
| **28** | Multi-profile support | ⏳ | Allow profile switching |

### Retrieval & Ranking (Priority: Medium)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **29** | Enhanced diversity | ⏳ | Improve MMR implementation |
| **30** | Hybrid retrieval | ⏳ | Add popularity/trending boost |
| **31** | Business rules | ⏳ | Config for boosting/filtering |
| **32** | Genre pre-filtering | ⏳ | Filter before ANN search |
| **33** | Cursor pagination | ⏳ | Replace offset pagination |
| **34** | Query cache | ⏳ | Cache frequent combinations |

### Testing & Evaluation (Priority: Medium)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **35** | Offline metrics | ⏳ | Add NDCG, diversity scores |
| **36** | Synthetic profiles | ⏳ | Generate test users |
| **37** | Quality monitoring | ⏳ | Track metrics over time |
| **38** | Test dataset | ⏳ | Build from user feedback |
| **39** | A/B framework | ⏳ | Test recommendation variants |
| **40** | Performance suite | ⏳ | Measure latency/throughput |
