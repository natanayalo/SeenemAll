# Seen'emAll Task Tracker

### 1. Core Tasks

| Task | Status | Notes |
|---|---|---|
| **1.1** Project scaffold (FastAPI + Docker + pgvector) | ✅ | Running `/healthz` |
| **1.2** Database schema + migrations | ✅ | Alembic + pgvector ready |
| **1.3** TMDB ETL ingestion | ✅ | `make etl-tmdb` |
| **1.4** Embedding generator | ✅ | MiniLM-L6-v2 vectors |
| **1.5** User history + profile vectors | ✅ | `/user/history` route works |
| **1.6** Candidate ANN search | ✅ | pgvector cosine query functional |
| **1.7** `/recommend` endpoint (core) | ✅ | Returns JSON titles/posters |
| **1.8** Intent parser (LLM) | ✅ | Parses mood/runtime/genre filters |
| **1.9** Reranker + explanation | ✅ | LLM reranks candidates + inline rationales |
| **1.10** JustWatch ETL (streaming links) | ✅ | `make etl-justwatch` fills availability |
| **1.11** `/watch-link` resolver | ✅ | endpoint available |
| **1.12** `/feedback` ingestion | ✅ | endpoint available |
| **1.13** Diversity & novelty logic | ✅ | MMR diversification available via query param |
| **1.14** Minimal web UI | ✅ | show cards + "watch" buttons |
| **1.15** Prompt & eval fixtures | ✅ | Templates + eval suite for LLM interactions: Intent parser prompts, reranker prompts with examples, gold dataset, metrics, error handling |
| **1.16** CI polish & deployment | ✅ | Docker Compose + Makefile done |

## Quality Improvements

### 2. Embedding Enhancement (Priority: High) — Completed

| Task | Status | Notes |
|---|---|---|
| **2.1** Add genres to embedding text | ✅ | Genre list prepended to title+overview |
| **2.2** Test embedding templates | ✅ | Implemented configurable templates with tests |
| **2.3** Evaluate domain models | ✅ | E5-small-v2 recommended, MovieCLIP evaluated |
| **2.4** Add temporal context | ✅ | Era + year/decade folded into embedding templates |
| **2.5** A/B test framework | ✅ | Added offline embed A/B utility + unit coverage |
| **2.6** Version embeddings | ✅ | Vectors keyed by version; pipeline accepts EMBED_VERSION |

### 3. User Profile Enhancement (Priority: High) — Completed

| Task | Status | Notes |
|---|---|---|
| **3.1** Improve rating weights | ✅ | Event-type multipliers (watch/like/rate) applied |
| **3.2** Genre preferences | ✅ | Time-decayed genre vector stored in `users.genre_prefs` |
| **3.3** Collaborative signals | ✅ | Blend neighbor vectors from overlapping histories |
| **3.4** Configurable recency | ✅ | Half-life via USER_PROFILE_DECAY_HALF_LIFE env |
| **3.5** Negative feedback | ✅ | "not interested" events excluded & flagged for recs |
| **3.6** Multi-profile support | ✅ | Profiles append via user_id::profile, API supports switching |

### 4. Retrieval & Ranking (Priority: Medium) — Completed

| Task | Status | Notes |
|---|---|---|
| **4.1** Enhanced diversity | ✅ | MMR rewritten with normalized vectors + fallbacks |
| **4.2** Hybrid retrieval | ✅ | ANN score blended with popularity/trending boosts |
| **4.3** Business rules | ✅ | JSON-configurable filters/boosts wired into pipeline |
| **4.4** Genre pre-filtering | ✅ | Intent-driven allowlist feeds ANN retrieval |
| **4.5** Cursor pagination | ✅ | Stateless `next_cursor` replaces offset paging |
| **4.6** Query cache | ✅ | In-memory cache with TTL + invalidation on profile updates |

### 5. Intent Agent & Rewrite (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **5.1** Define intent schema | ✅ | Pydantic model: include/exclude genres, runtime, languages, year range, maturity, boosts |
| **5.2** Implement LLM parser | ✅ | Provider-agnostic `parse_intent` with few-shot prompt and user context |
| **5.3** Guardrails & fallback | ✅ | JSON validation, sane defaults, safe failure modes |
| **5.4** Intent cache & metrics | ✅ | TTL cache by (user, profile, query); hit/miss counters |
| **5.5** Intent fixtures | ✅ | Gold tests for “light sci-fi <2h”, “no gore”, etc. |
| **5.6** Define rewrite schema | ✅ | ≤8-word rewritten text, facet allow/block fields |
| **5.7** Implement rewrite LLM | ✅ | Bounded output with normalization & token cap |
| **5.8** Rewrite cache & metrics | ✅ | Cache by (intent_hash, query); expose hit/miss |
| **5.9** Rewrite tests | ✅ | Stability + length constraint coverage |

### 6. Entity Linking, Query Vector Blend & Mixer (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **6.1** Title/person linker | ✅ | Resolve entities via TMDB search API |
| **6.2** Linker integration | ✅ | Feed allowlist IDs into recall constraints |
| **6.3** Linker cache & limits | ✅ | Rate limiting + caching to protect TMDB |
| **6.4** Linker tests | ✅ | Queries like “like Interstellar”, “more from Villeneuve” |
| **6.5** Blend query vector | ✅ | `qvec = normalize(α·short_vec + (1-α)·emb(rewrite))` |
| **6.6** Cold-start fallbacks | ✅ | Handle missing vectors; log path taken |
| **6.7** Collaborative recall | ✅ | Neighbor-based recall from co-watch history |
| **6.8** Trending prior recall | ⏳ | Rolling CTR/popularity table |
| **6.9** Mixer scoring | ⏳ | Weighted blend (content/CF/popularity) + novelty bonus |
| **6.10** Mixer tests | ⏳ | Deterministic toy dataset validation |

### 7. Diversity, Franchise Caps & Heuristic Ranker (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **7.1** Advanced MMR | ⏳ | λ≈0.7 pass on top-K candidates |
| **7.2** Franchise cap | ⏳ | Limit 1–2 per franchise/series |
| **7.3** Serendipity slot | ⏳ | Force 10–20% long-tail inclusion |
| **7.4** Diversity tests | ⏳ | Verify caps and serendipity behaviour |
| **7.5** Heuristic ranker | ⏳ | Deterministic boosts + template system |
| **7.6** Explanation templates | ⏳ | Config-driven messaging with smoke tests |

### 8. Optional LLM Reranker (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **8.1** Small-model rerank | ⏳ | Top-40→Top-12 with cache, timeout, fallback |
| **8.2** Provider toggle | ⏳ | `RERANK_PROVIDER` + latency logging |
| **8.3** Strategy A/B | ⏳ | Enable via `strategy=v2_llm` for comparison |

### 9. API, Diagnostics & Metrics (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **9.1** `/recommend` strategy | ⏳ | Route `strategy=v2` through agent pipeline |
| **9.2** Cursor pagination reuse | ⏳ | Serve pages from cached candidate set |
| **9.3** `/debug/recommend` | ⏳ | Surface intent JSON, rewrite, sources, MMR picks |
| **9.4** Structured logs | ⏳ | Request-id logging with PII redaction |
| **9.5** Cache layer | ⏳ | Intent/rewrite/candidate/reranker caches (TTL) |
| **9.6** Cache invalidation | ⏳ | Clear on `/user/history` for same user/profile |
| **9.7** Metrics & telemetry | ⏳ | ANN latency, cache hit rates, reranker %, MMR impact; Prometheus/log counters |

### 10. Testing & Evaluation (Priority: Medium)

| Task | Status | Notes |
|---|---|---|
| **10.1** Offline metrics | ⏳ | Add NDCG, diversity scores |
| **10.2** Synthetic profiles | ⏳ | Generate test users |
| **10.3** Quality monitoring | ⏳ | Track metrics over time |
| **10.4** Test dataset | ⏳ | Build from user feedback |
| **10.5** A/B framework | ⏳ | Test recommendation variants |
| **10.6** Performance suite | ⏳ | Measure latency/throughput |
