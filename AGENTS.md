# Seen’emAll Agents Overview

> Multi-agent roles that orchestrate data ingestion, embeddings, and recommendations.

> **Quality bar:** Maintain automated test coverage above 90% (enforced via pytest’s coverage gate).

---

### 🧩 1. ETL Agent
**Goal:** Populate the catalog from TMDB
**Implements:** `etl/tmdb_sync.py`

- Fetches movies and TV shows (popular / top rated / trending)
- Upserts metadata into `items` table
- Runs on demand (`make etl-tmdb`)

---

### 🔢 2. Embedding Agent
**Goal:** Convert items into semantic vectors
**Implements:** `etl/compute_embeddings.py`

- Uses `sentence-transformers` MiniLM-L6-v2 (384-dim)
- Stores normalized vectors in `item_embeddings`
- Run with `make embed`

---

### 👤 3. User Profile Agent
**Goal:** Build and maintain user preference vectors
**Implements:** `api/core/user_profile.py`

- Averages recent watch history (time-decayed)
- Updates `users.long_vec` and `users.short_vec`
- Triggered automatically via `/user/history`

---

### 🧭 4. Candidate Agent
**Goal:** Retrieve similar items by cosine similarity
**Implements:** `api/core/candidate_gen.py`

- Uses pgvector `<->` distance
- Excludes items the user has already watched
- Returns ANN-ranked IDs for recommendation

---

### 🧠 5. Reranker Agent
**Goal:** Reorder candidates with natural-language explanations
**Implements:** `api/core/reranker.py`

- Calls OpenAI- or Gemini-compatible APIs when configured
- Generates concise reasons tied to the user’s query/filters
- Falls back gracefully to ANN order when disabled or failing

---

### 💬 6. Recommendation Agent
**Goal:** Serve recommendations via API
**Implements:** `api/routes/recommend.py`

- Combines user vector + ANN search
- Invokes the reranker for final ordering + explanations
- Endpoint: `GET /recommend?user_id=u1&limit=10`

---

### (Coming Soon)
| Agent | Description |
|-------|--------------|
| 🎯 **Intent Agent** | Parse free-text queries (“light sci-fi < 2h”) and filter catalog |
| 🎬 **Streaming Agent** | Resolve watch links from JustWatch |
| ❤️ **Feedback Agent** | Collect feedback signals and retrain user vector |
