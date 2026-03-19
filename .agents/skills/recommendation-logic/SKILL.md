---
name: recommendation-logic
description: Deep domain knowledge of the Seen'emAll retrieval and ranking pipeline. Use when modifying core ranking logic, diversity (MMR), or reranking algorithms.
---
# Recommendation Logic Skill

This skill provides the deep context needed to modify or debug the core recommendation pipeline in `api/routes/recommend.py` and `api/core/`.

## Pipeline Overview
1. **Intent Parsing**: LLM-driven query understanding and fallback heuristics for genres, year, runtime, mood, and media type.
2. **Recall (Candidate Generation)**:
   - **ANN**: `pgvector` cosine similarity on user short-term vector plus query rewrite.
   - **Collaborative**: Neighbors' recently watched items.
   - **Trending**: Popularity priors and trending racks.
   - **Constraint Prior**: Catalog-aware fallback ordered by semantic match and query-aware bonuses.
3. **Mixer Scoring**: Weighted blend of retrieval sources with novelty bonuses.
4. **Business Rules**: JSON-configurable boosts and filters (`config/business_rules.json`).
5. **Diversification**: Lambda-based MMR at top-K.
6. **Reranking**: Final LLM pass for explanation generation and precise ordering.

## Overfitting Guardrails
- Do not tune production logic against exact evaluation case titles, IDs, or exact benchmark phrasings.
- Prefer reusable semantic-family detectors such as `time-bending thriller`, `optimistic sci-fi TV`, or `family fantasy adventure`, not one-off eval phrases.
- If a change only improves the benchmark wording and not obvious paraphrases, treat it as suspect.
- Keep benchmark repair separate from ranking logic changes. Document whether gains came from eval-set cleanup or route behavior.
- Favor broader catalog-aware shaping over query-phrase micro-branches.

## Performance And Caching
- **Caching**: Per-request caching with TTL (`300s` default).
- **Invalidation**: Invalidate cache via `clear_cache_for_user()` when user watch history changes.
- **Latency Targets**:
  - ANN `< 100ms`
  - MMR `< 50ms`
  - Small reranker `< 500ms`
  - LLM reranker `< 20s` async

## Debugging
- Use `make debug-rec` or `GET /recommend/debug` to inspect internal scores, stage counts, and source counts for specific items.
- For suspected eval overfit, check whether the fix changed:
  - intent normalization
  - recall composition
  - constraint-prior ordering
  - or only the exact benchmark phrase path
