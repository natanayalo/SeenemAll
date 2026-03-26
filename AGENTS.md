# Seen’emAll Agents Overview

> Multi-agent roles orchestrating data ingestion, embeddings, evaluation, and recommendation.

> **Quality bar:** Maintain automated test coverage ≥ 85% (pytest gate).

---

### 🧩 1. Catalog ETL Agent
**Goal:** Populate & enrich the catalog from TMDB
**Implements:** `etl/tmdb_sync.py`

- Imports movies/TV in batches (popular / trending / top rated)
- Upserts core metadata plus cast, directors, producers, writers
- Captures TMDB keywords and spoken languages for downstream filters
- Run with `make etl-tmdb` (supports `SINCE=...`, `BATCH=...`)

---

### 🔢 2. Embedding Agent
**Goal:** Generate semantic vectors for catalog items
**Implements:** `etl/compute_embeddings.py`

- Uses `sentence-transformers` MiniLM-L6-v2 (384-dim) by default
- Versioned vectors stored in `item_embeddings` (supports template overrides)
- Re-runs via `make embed` (accepts `EMBED_VERSION`, `EMBED_BATCH`, `DEVICE`)

---

### 🎬 3. Availability Agent
**Goal:** Resolve watch links & providers
**Implements:** `etl/justwatch_sync.py`

- Queries JustWatch to map TMDB IDs to per-country providers
- Normalises offers into the `availability` table (service, deeplink)
- Run with `make etl-justwatch`

---

### 🧠 4. Query Matcher Agent
**Goal:** Extract structured filters from natural language
**Implements:** `api/core/filter_matcher.py`

- spaCy PhraseMatcher for languages, canonical genres, keywords, media types
- Recognises cast/crew names and “like/similar to …” reference titles
- Feeds `SearchFilters` + residual text into the candidate generator

---

### 👤 5. User Profile Agent
**Goal:** Maintain user preference vectors & neighbors
**Implements:** `api/core/user_profile.py`

- Builds long/short vectors from watch history (time-decayed)
- Stores neighbor graph, genre prefs, and negative items
- Triggered automatically via `/user/history` ingestion

---

### 🧭 6. Candidate Agent
**Goal:** Produce intent-aware candidate IDs
**Implements:** `api/core/candidate_gen.py`

- Runs SQL prefilter (media type / genre / provider) to get allowlists & boosts
- Hybrid ANN retrieval:
  - **Elasticsearch** kNN + multi_match fusion (manual RRF)
  - **pgvector** cosine fallback
- Applies exclusion list, handles cold-start rewrite-only ANN, and catalogue fallback for cast/crew queries

---

### 🧠 7. Reranker Agent
**Goal:** Reorder candidates and generate explanations
**Implements:** `api/core/reranker.py`

- Supports OpenAI / Gemini providers for natural-language rationales
- Local “small” reranker (MiniLM-L6-v2) for offline or keyless deployments
- Gracefully falls back to heuristic ordering when disabled or failing

---

### 💬 8. Recommendation Agent
**Goal:** Serve `/recommend` API responses
**Implements:** `api/routes/recommend.py`

- Merges user vectors, rewrite vectors, collaborative neighbors, trending priors, and business rules
- Integrates entity linker + spaCy matcher outputs into `SearchFilters`
- Applies provider filtering, optional franchise cap/MMR diversification
- Calls reranker for final ordering & explanations; returns paginated JSON

---

### 📊 9. Evaluation Agent
**Goal:** Benchmark recommendation quality offline
**Implements:** `evaluation/evaluate.py`

- Resolves title-based gold sets via SQLAlchemy helper (`evaluation/resolver.py`)
- Computes Precision@K, Recall@K, MAP, nDCG@K; writes per-rank and summary CSVs
- Generates Evidently ranking reports (`evaluation/report.html`)
- Driven by CLI (`python -m evaluation.evaluate`) and make targets (`make eval`, `make eval-report`)

---

### (Coming Soon)
| Agent | Description |
|-------|-------------|
| ❤️ **Feedback Agent** | Collect explicit feedback, schedule retraining loops |
