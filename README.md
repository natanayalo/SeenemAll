# 🎬 Seen’emAll

AI-powered movie & TV recommender that learns your taste from what you’ve already seen.
Built with **FastAPI**, **Postgres + pgvector**, and **sentence-transformers**.

---

## ⚙️ Stack
- **FastAPI** backend
- **Postgres** (with `pgvector` extension)
- **sentence-transformers MiniLM-L6-v2** for embeddings
- **TMDB API** for metadata ingestion
- **Docker Compose** for one-command setup

---

## 🚀 Quick Start

```bash
# 1. Unpack & enter
unzip SeenemAll.zip -d .
cd SeenemAll

# 2. Configure
cp .env.example .env
# edit TMDB_API_KEY=your_tmdb_key
# optional: set RERANK_PROVIDER/RERANK_API_KEY for LLM reranking

# 3. Launch stack
docker compose up -d --build

# 4. Run migrations
make migrate

# 5. Populate catalog
make etl-tmdb

# 6. Generate embeddings
make embed

# 7. Create sample user history
curl -X POST http://localhost:8000/user/history \
  -H "content-type: application/json" \
  -d '{"user_id":"u1","items":[1,2,3,4,5]}'

# 8. Get recommendations
curl "http://localhost:8000/recommend?user_id=u1&limit=10"
````

---

## 🔁 LLM Reranker

- Set `RERANK_PROVIDER=openai` with `RERANK_API_KEY` (or rely on `OPENAI_API_KEY`) **or**
  `RERANK_PROVIDER=gemini` with a compatible Google Generative AI key.
- Defaults: `RERANK_MODEL=gpt-4o-mini` for OpenAI, `gemini-2.0-flash-exp` for Gemini.
- Disable temporarily with `RERANK_ENABLED=0`; without a key we automatically fall back
  to ANN ordering with heuristic explanations.

---

## 🧠 Agents

See [AGENTS.md](./AGENTS.md) for roles and flow.

---

## ✅ Progress

See [TASKS.md](./TASKS.md) for the up-to-date roadmap.

---

## ⚖️ Attribution

> This product uses the TMDB API but is not endorsed or certified by TMDB.
> Data provided by [The Movie Database (TMDB)](https://www.themoviedb.org).

---

## 📄 License

MIT © Natan Ayalo
