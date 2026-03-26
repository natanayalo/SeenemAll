# 🎬 Seen’emAll

AI-powered movie & TV recommendations with on-demand evaluations, rich catalog metadata, and intent-aware hybrid search.

---

## 🧭 Overview

Seen’emAll is a multi-agent recommendation stack:

| Agent | Purpose | Key Module |
|-------|---------|------------|
| **ETL** | Pull TMDB / JustWatch metadata into Postgres (cast, crew, keywords, spoken languages, availability) | `etl/tmdb_sync.py`, `etl/justwatch_sync.py` |
| **Embedding** | Encode catalog titles with MiniLM-L6-v2, keep historical versions | `etl/compute_embeddings.py` |
| **User Profile** | Maintain long/short vectors, neighbor cache, negative feedback | `api/core/user_profile.py` |
| **Candidate Generator** | Blend ANN (Elasticsearch) + collaborative + trending + business rules | `api/core/candidate_gen.py`, `api/routes/recommend.py` |
| **Reranker** | Produce explanations (LLM) or lightweight MiniLM rerank | `api/core/reranker.py` |
| **Evaluation** | Offline evaluation, title resolution, Evidently report | `evaluation/evaluate.py` |

The system runs entirely in Docker Compose and exposes a single `/recommend` endpoint with rich query semantics.

---

## 🏗️ Architecture

```mermaid
graph TD
    Client["Client<br/>(Frontend / API Consumer)"]
    API["FastAPI Service<br/>/recommend"]
    Matcher["Query Filter Matcher<br/>(spaCy)"]
    Profiles["User Profile Agent<br/>(vectors, negatives, neighbors)"]
    Prefilter["SQL Prefilter<br/>(media/genre/provider)"]
    Candidates["Candidate Generator<br/>(ANN + fallbacks)"]
    Rules["Business Rules<br/>(boosts & filters)"]
    Reranker["Reranker<br/>(LLM or small model)"]
    History[(Postgres + pgvector)]
    ES[(Elasticsearch kNN index)]
    TMDB["TMDB ETL"]
    JustWatch["JustWatch ETL"]
    Embedder["Embedding Worker<br/>MiniLM-L6-v2"]

    Client --> API
    API --> Matcher
    API --> Profiles
    Profiles --> History
    Matcher --> Prefilter
    Prefilter --> Candidates
    Candidates --> ES
    Candidates --> History
    API --> Rules
    Rules --> Reranker
    Reranker --> API
    API --> Client

    TMDB --> History
    JustWatch --> History
    Embedder --> History
```

---

## ⚙️ Stack & Dependencies

- **FastAPI** + **Pydantic** backend
- **Postgres** with `pgvector` (user embeddings + metadata)
- **Elasticsearch 8** (HNSW kNN, manual RRF hybrid)
- **sentence-transformers MiniLM-L6-v2** embeddings
- **spaCy** (`en_core_web_sm`) matcher for languages/genres/people/keywords
- **Docker Compose** (API, DB, Elasticsearch, optional frontend)
- Optional: **OpenAI** or **Gemini** for LLM reranking
- **Evidently** for evaluation reporting (optional install)

---

## 🚀 Quick Start

```bash
git clone <repo>
cd SeenemAll

cp .env.example .env
# Set TMDB_API_KEY, optional RERANK_PROVIDER/API keys, tweak ANN/Rerank knobs.

docker compose up -d --build         # API, Postgres, Elasticsearch (and frontend if enabled)

make migrate                         # Alembic migrations (includes cast/crew/keywords schema)
make es-setup                        # Create Elasticsearch index (run with FORCE=1 on schema change)
make etl-tmdb                        # Pull TMDB catalog (supports SINCE=2024-01-01T00:00:00Z)
make embed                           # MiniLM embeddings (EMBED_VERSION, EMBED_BATCH, etc.)
make es-sync                         # Push catalog to Elasticsearch (BATCH=250, REFRESH=1 optional)
make etl-justwatch                   # Populate availability (optional)
```

Seed user history:

```bash
curl -X POST http://localhost:8000/user/history \
  -H "content-type: application/json" \
  -d '{"user_id":"u1","profile":"main","items":[603,155,1207],"event_type":"watched"}'
```

Request recommendations:

```bash
curl "http://localhost:8000/recommend?user_id=u1&profile=main&query=fantasy%20TV%20epics%20like%20The%20Witcher&limit=10&diversify=false"
```

---

## 🔁 Request Lifecycle Diagram

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI /recommend
    participant P as User Profile Loader
    participant M as Intent & Matcher
    participant F as SQL Prefilter
    participant G as ANN Candidates
    participant B as Business Rules
    participant D as Diversifier (MMR)
    participant R as Reranker

    C->>A: GET /recommend (user_id, query, params)
    A->>P: load_user_state(user_id::profile)
    P-->>A: vectors, negatives, providers
    A->>M: parse_intent + spaCy matcher + linked entities
    M-->>A: IntentFilters & SearchFilters
    A->>F: prefilter_allowed_ids(intent, limit)
    F-->>A: allowlist / boost IDs / enforce_genres flag
    A->>G: ann_candidates(vector, filters, allowlist)
    G-->>A: candidate ids
    A->>A: hydrate metadata (Postgres)
    A->>B: apply_business_rules(items)
    B-->>A: scored candidates
    A->>D: diversify_with_mmr(items, limit)
    D-->>A: diversified list
    A->>R: rerank_with_explanations(items, intent, user context)
    R-->>A: ordered items + rationales
    A-->>C: JSON response (items, next_cursor)
```

---

## 🔁 Request Flow (Detailed Steps)

1. **Profile Load** – `load_user_state` fetches short/long vectors, neighbors, negative items, provider preferences.
2. **Intent Parsing**
   - Entity linker captures explicit titles/people.
   - LLM parser (`use_llm_intent=true`) or legacy stub extracts genres/media types/runtime bounds.
   - spaCy matcher adds languages, normalized genres (maps “fantasy” → `Fantasy`, `Sci-Fi & Fantasy`), keywords, cast/crew, and “reference titles” (phrases like “like *The Witcher*”).
3. **Prefilter (SQL)** – `_prefilter_allowed_ids` runs a strict (media type + genres + providers) and relaxed pass to produce `allowed_ids`, `boost_ids`, and `enforce_genres`.
4. **Rewrite Vector** – `_build_rewrite_vector` blends ANN description, rewrites, and extracted titles. Even cold-start users get a query-driven vector for ANN.
5. **ANN Retrieval**
   - **Elasticsearch** (default): kNN + multi_match query; manual Reciprocal Rank Fusion merges ANN and keyword search. Filters include media types, genres, languages, keywords, cast/crew.
   - **pgvector** (fallback): cosine similarity in Postgres.
6. **Cold-start Safeguards** – If ANN returns nothing and the request has cast/crew filters, we retry without the allowlist, then fall back to a direct SQL lookup to ensure actor/creator queries always return something. Otherwise, `_cold_start_candidates` orders catalog titles by trending/popularity or top-rated heuristics.
7. **Collaborative & Trending** – Neighbor scores, business-rule boosts, and trending priors merge with the ANN results up to `candidate_limit`.
8. **Post-filtering** – We hydrate metadata, enforce media type/genre/runtime/maturity, and strictly reapply cast/crew filters even if we relaxed them earlier. Provider filtering trims watch options unless we need fallback options to fill `limit`.
9. **Diversification & Scoring** – Optional `diversify=true` runs franchise cap/MMR. Mixer weights are controlled via env vars or query overrides (`mixer_ann_weight`, etc.).
10. **Reranker** – `RERANK_ENABLED=1` uses the configured LLM provider for explanations; `RERANK_PROVIDER=small` runs a local MiniLM reranker; `RERANK_ENABLED=0` falls back to heuristic explanations.
11. **Response** – JSON payload with ranked `items`, each including metadata, watch options, source scores, and explanations (`explanation`, `reason` fields), plus optional cursor for pagination.

Logging highlights important fallback decisions (e.g., classic-top-rated heuristics, allowlist relaxation for people filters).

---

## 🔧 Request Parameters & Modes

- `query` – natural-language text (“gritty street-level superhero TV shows”)
- `profile` – pick a profile (`user_id::profile`)
- `limit`, `cursor` – pagination controls
- `diversify=true|false` – toggles franchise cap/MMR
- `use_llm_intent=true|false` – choose LLM parser vs legacy stub
- `rewrite_override` – custom ANN rewrite text
- `ann_backend_override=elasticsearch|pgvector`
- `mixer_*` weights – fine-tune ANN/collab/trending/popularity/vote signals
- `classic_top_rated=true|false` – force the top-rated blend
- Negative feedback (`event_type="not_interested"`) automatically excludes items per user/profile

---

## 📊 Evaluation Toolkit

Run the hybrid evaluation CLI and generate Evidently reports:

```bash
make eval                                 # ID-based gold set (evaluation/evaluation_set.json)
make eval-report                          # Generates evaluation/report.html (if Evidently installed)
python -m evaluation.evaluate --help      # CLI flags: --k, --resolve-titles, --dataset, --set, --titles-set
```

Features:
- Dual gold-set input: ID-based (`evaluation_set.json`) or title-based (`evaluation_set.titles.json`) resolved against Postgres.
- Optional `--dataset=movielens20m` stub ready for future public datasets.
- Per-rank CSV (`evaluation_results.csv`) with `query_id`, `rank`, `item_id`, `relevant`.
- Summary CSV (`evaluation_scores*.csv`) capturing Precision@K, Recall@K, MAP, nDCG@K per query and backend.
- Evidently ranking report stored at `evaluation/report.html`.
- Helper scripts:
  - `evaluation/build_golden.py` – generate candidate gold lists from TMDB for manual curation.
  - `evaluation/merge_candidates.py` – merge evaluation runs.
  - Sweep CSVs (`evaluation/sweep_*.csv`) demonstrate parameter search experiments.

---

## 🔍 Catalog & Metadata

The TMDB ETL fetches:
- Cast (top 5), directors (top 2), producers (top 2), writers (top 2)
- Keywords (mapped into Elasticsearch filters)
- Spoken languages (used by the matcher)
- Runtime, release year, popularity, vote counts/rankings

Availability (`etl/justwatch_sync.py`) keeps per-country provider data—usable in prefilters and watch option filtering.

Elasticsearch documents include the above fields so both ANN and the reranker can leverage them. Mapping updates are defined in `etl/elasticsearch_index.py`.

---

## 🧪 Testing & Tooling

- `./.venv/bin/python -m pytest` for unit/integration tests (coverage gate: 85%)
- `tests/unit/test_filter_matcher.py` verifies the spaCy matcher (languages, genres, people, reference titles)
- `tests/unit/test_elasticsearch_search.py` asserts dual-search fusion and filter placement
- `make lint` (if configured) for static checks

---

## 🛠️ Common Tasks

- **Recreate Elasticsearch index**: `make es-setup FORCE=1 && make es-sync`
- **Backfill embeddings**: `make embed EMBED_VERSION=v2`
- **Run cold-start milestone**: use `--resolve-titles` to test natural-language queries against sparse catalogs.
- **Switch ANN backend**: set `ANN_BACKEND=pgvector` in `.env` to force Postgres retrieval (useful for benchmarking).
- **Reranker small model**: `RERANK_PROVIDER=small` for local MiniLM reranker without external keys.

---

## 🧊 Cold-Start Strategy

- If the user has no vector, we still build a rewrite vector solely from the query (including reference titles like “The Witcher”) and run ANN.
- People filters trigger relaxed allowlists and a catalogue fallback to ensure actor-based queries never return empty lists.
- `_cold_start_candidates` orders catalog rows by trending / popularity when ANN fails (still respecting media types, genres, and provider constraints from the query).

---

## ⚠️ Notes & Troubleshooting

- If ANN returns nothing, check logs for “Relaxed ANN filters…” or “Using catalogue fallback…” messages. Tight genre/provider filters may leave no viable rows.
- Mapping (genre → catalog name) is controlled in `_GENRE_CANONICAL_TO_CATALOG` (e.g., “fantasy” maps to `["Fantasy", "Sci-Fi & Fantasy"]`).
- The entity linker uses TMDB data; restart the API after large ETL runs so caches refresh.
- LLM reranker grace-degrades to ANN ordering with explanations (`RERANK_ENABLED=0` or missing API key).
- Evaluation requires Postgres access (`EVAL_DB_DSN`; defaults to `postgresql+psycopg2://app:app@localhost:5432/reco`).

---

## 📂 Key Paths

- `/recommend` implementation: `api/routes/recommend.py`
- Intent parsing & matcher: `api/core/llm_parser.py`, `api/core/filter_matcher.py`
- Candidate retrieval: `api/core/candidate_gen.py`, `api/core/elasticsearch_search.py`
- Evaluation suite: `evaluation/`
- ETL jobs: `etl/`
- Business rules: `api/core/business_rules.py`

---

Happy recommending! 🎥✨
