---
name: recommendation-logic
description: Deep domain knowledge of the Seen’emAll retrieval and ranking pipeline. Use when modifying core ranking logic, diversity (MMR), or reranking algorithms.
---
# Recommendation Logic Skill

This skill provides the deep context needed to modify or debug the core recommendation pipeline in `api/routes/recommend.py` and `api/core/`.

## 🧬 Pipeline Overview
1.  **Intent Parsing**: LLM-driven query understanding (genres, year, runtime, mood).
2.  **Recall (Candidate Generation)**:
    *   **ANN**: `pgvector` cosine similarity on user short-term vector + query rewrite.
    *   **Collaborative**: Neighbors' recently watched items.
    *   **Trending**: Popularity priors and trending racks.
3.  **Mixer Scoring**: Weighted blend of retrieval sources with novelty bonuses.
4.  **Business Rules**: JSON-configurable boosts and filters (`config/business_rules.json`).
5.  **Diversification**: Lambda-based MMR (Maximal Marginal Relevance) at Top-K.
6.  **Reranking**: Final LLM pass for explanation generation and precise ordering.

## 🛠️ Performance & Caching
- **Caching**: Per-request caching with TTL (300s default).
- **Invalidation**: Invalidate cache via `clear_cache_for_user()` when the user watch history changes.
- **Latency Targets**:
    - ANN < 100ms
    - MMR < 50ms
    - Reranker (small) < 500ms
    - Reranker (LLM) < 20s (async)

## 🔍 Debugging
- Always use `make debug-rec` or `GET /recommend/debug` to inspect the internal scores and source counts for specific items.
