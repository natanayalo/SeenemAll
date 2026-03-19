# Seen'emAll Task Tracker

## Current Freeze

- Recommendation baseline frozen and validated on `2026-03-19` after API restart plus warm full eval.
- Current accepted full-eval baseline:
  - `default nDCG@10 0.4957`
  - `MAP 0.3596`
  - `Hit@10 1.0000`
- Next planned work:
  - embedding representation `v2` experiment
  - specifically test whether TMDB `tagline` and filtered `keywords` improve semantic neighborhoods before any schema or ETL rollout

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
| **6.8** Trending prior recall | ✅ | Rolling CTR/popularity table |
| **6.9** Mixer scoring | ✅ | Weighted blend (content/CF/popularity) + novelty bonus |
| **6.10** Mixer tests | ✅ | Deterministic toy dataset validation |

### 7. Diversity, Franchise Caps & Heuristic Ranker (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **7.1** Advanced MMR | ✅ | λ≈0.7 pass on top-K candidates |
| **7.2** Franchise cap | ✅ | Limit 1–2 per franchise/series |
| **7.3** Serendipity slot | ✅ | Force 10–20% long-tail inclusion |
| **7.4** Diversity tests | ✅ | Verify caps and serendipity behaviour |
| **7.5** Heuristic ranker | ✅ | Deterministic boosts + template system |
| **7.6** Explanation templates | ✅ | Config-driven messaging with smoke tests |

### 8. Optional LLM Reranker (Priority: High) — Completed

| Task | Status | Notes |
|---|---|---|
| **8.1** Small-model rerank | ✅ | Top-40→Top-12 with cache, timeout, fallback |
| **8.2** Provider toggle | ✅ | `RERANK_PROVIDER` + latency logging |
| **8.3** Strategy A/B | ✅ | Handled via env vars & `RERANK_ENABLED` |

### 9. API, Diagnostics & Metrics (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **9.1** `/recommend` strategy | ✅ | `strategy` integrated into main pipeline natively |
| **9.2** Cursor pagination reuse | ✅ | Serve pages from cached candidate set |
| **9.3** `/debug/recommend` | ✅ | Surface intent JSON, rewrite, sources, MMR picks |
| **9.4** Structured logs | ✅ | Request-id logging with PII redaction |
| **9.5** Cache layer | ✅ | Intent/rewrite/candidate/reranker caches (TTL) |
| **9.6** Cache invalidation | ✅ | Clear on `/user/history` for same user/profile |
| **9.7** Metrics & telemetry | ✅ | ANN latency, cache hit rates, reranker %, MMR impact; Metric registry counters |

### 10. Testing & Evaluation (Priority: Medium)

| Task | Status | Notes |
|---|---|---|
| **10.1** Offline metrics | ⏳ | Add NDCG, diversity scores |
| **10.2** Synthetic profiles | ⏳ | Generate test users |
| **10.3** Quality monitoring | ⏳ | Track metrics over time |
| **10.4** Test dataset | ⏳ | Build from user feedback |
| **10.5** A/B framework | ⏳ | Test recommendation variants |
| **10.6** Performance suite | ⏳ | Measure latency/throughput |

### 11. Elasticsearch Retrieval Migration (Priority: High)

| Task | Status | Notes |
|---|---|---|
| **11.1** Provision local Elasticsearch | ⏳ | Add a Docker service (v8.x LTS with kNN/HNSW support) alongside the API/db stacks; configure JVM heap, persistence, and auth suitable for local dev |
| **11.2** Define rec index mapping | ⏳ | Create `items` index with structured fields (`genres`, `media_type`, `runtime`, `release_year`, `maturity`, `streaming_providers`) plus `dense_vector` for embeddings and full-text fields for title/overview |
| **11.3** Nightly data sync ETL | ⏳ | New ETL command to export items from Postgres, push into Elasticsearch (bulk API), and schedule nightly refresh; document reindex flow |
| **11.4** ANN/Hybrid query prototype | ⏳ | Implement ES kNN (HNSW) query, optionally blended with keyword filters; benchmark recall vs. pgvector for evaluation queries |
| **11.5** API integration | ⏳ | Add ES client config, replace `ann_candidates` with ES-based retrieval, keep collaborative/popularity blending, ensure filters (genres/year/runtime/provider) apply in ES |
| **11.6** Evaluation & rollout | ⏳ | Re-run `make eval`, capture metrics vs. current pipeline, monitor latency; update docs/env vars for staging/production deployment

### 12. Evaluation Set V2 (Priority: High)

Target outcome: evolve offline gold data from 18 cases to a v2 set of 72 high-signal cases with enforced bucket and slice quotas.

| Task | Status | Notes |
|---|---|---|
| **12.1** Freeze v2 target distribution | done | Added `evaluation/evaluation_set_v2_targets.json` with final 72-case quotas + milestone targets (`m1`, `m2`, `v2`) |
| **12.2** Add quota checker | done | Added `evaluation/check_distribution.py`; supports milestone checks and JSON report output |
| **12.3** Create v2 working set scaffold | done | Added `evaluation/evaluation_set_v2.json` cloned from current set as the growth baseline |
| **12.4** Wire workflow commands | done | Added `make eval-check-dist`; docs updated in `evaluation/README.md` |
| **12.5** Milestone A expansion (18 -> 30) | done | Completed with `evaluation_set_v2.json` at 30/30 and `distribution_m1_report.json` green |
| **12.6** Milestone B expansion (30 -> 50) | done | Completed with `evaluation_set_v2.json` at 50/50 and `distribution_m2_report.json` green |
| **12.7** Milestone C expansion (50 -> 72) | done | Completed at 72/72 with `distribution_v2_report.json` green |
| **12.8** Label-quality hardening | done | Verified constrained/edge cases satisfy minimums (`golden_set >= 5`, `negative_set >= 3`) |
| **12.9** Release-gate re-baseline on v2 | done | Regenerated `evaluation/artifacts/gate/{baseline,candidate}_*.{csv,json}` and `gate_report.json` on v2 set (gate PASS) |
| **12.10** Normalize metadata for stable diffs | done | Added `evaluation/normalize_evaluation_set.py`; backfills stable `case_id` and explicit `distribution_bucket` for v2 |
| **12.11** Add dataset quality audit | done | Added `evaluation/audit_evaluation_set.py` + `make eval-audit-set` to track weak labels, overlap, and negative coverage |
| **12.12** Publish v2 quality hardening backlog | done | Added `evaluation/v2_quality_backlog.md` with concrete remediation priorities and acceptance checks |
