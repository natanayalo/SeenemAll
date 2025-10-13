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
| **17** | Add genres to embedding text | ✅ | Genre list prepended to title+overview |
| **18** | Test embedding templates | ✅ | Implemented configurable templates with tests |
| **19** | Evaluate domain models | ✅ | E5-small-v2 recommended, MovieCLIP evaluated |
| **20** | Add temporal context | ✅ | Era + year/decade folded into embedding templates |
| **21** | A/B test framework | ✅ | Added offline embed A/B utility + unit coverage |
| **22** | Version embeddings | ✅ | Vectors keyed by version; pipeline accepts EMBED_VERSION |

### User Profile Enhancement (Priority: High)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **23** | Improve rating weights | ✅ | Event-type multipliers (watch/like/rate) applied |
| **24** | Genre preferences | ✅ | Time-decayed genre vector stored in `users.genre_prefs` |
| **25** | Collaborative signals | ✅ | Blend neighbor vectors from overlapping histories |
| **26** | Configurable recency | ✅ | Half-life via USER_PROFILE_DECAY_HALF_LIFE env |
| **27** | Negative feedback | ✅ | "not interested" events excluded & flagged for recs |
| **28** | Multi-profile support | ✅ | Profiles append via user_id::profile, API supports switching |

### Retrieval & Ranking (Priority: Medium)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **29** | Enhanced diversity | ✅ | MMR rewritten with normalized vectors + fallbacks |
| **30** | Hybrid retrieval | ✅ | ANN score blended with popularity/trending boosts |
| **31** | Business rules | ✅ | JSON-configurable filters/boosts wired into pipeline |
| **32** | Genre pre-filtering | ✅ | Intent-driven allowlist feeds ANN retrieval |
| **33** | Cursor pagination | ✅ | Stateless `next_cursor` replaces offset paging |
| **34** | Query cache | ✅ | In-memory cache with TTL + invalidation on profile updates |

### Testing & Evaluation (Priority: Medium)
| # | Task | Status | Notes |
|---|------|---------|-------|
| **35** | Offline metrics | ⏳ | Add NDCG, diversity scores |
| **36** | Synthetic profiles | ⏳ | Generate test users |
| **37** | Quality monitoring | ⏳ | Track metrics over time |
| **38** | Test dataset | ⏳ | Build from user feedback |
| **39** | A/B framework | ⏳ | Test recommendation variants |
| **40** | Performance suite | ⏳ | Measure latency/throughput |
