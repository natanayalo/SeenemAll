# Recommendation Pipeline Reference

This file is loaded on demand by the recommendation-logic skill.

## Pipeline Stages
1. Parse user intent and optional query enrichments.
2. Build retrieval candidates (ANN + optional collaborative + optional trending priors).
3. Apply mixer scoring and normalization.
4. Apply business rules and hard filters.
5. Optionally diversify with MMR.
6. Optionally rerank with explanation generation.
7. Emit response payload and diagnostics/metrics.

## Invariants
- Exclude already watched content from recommendations.
- Keep cache semantics correct across user/profile/query/override parameters.
- Keep deterministic ordering for identical inputs when reranker is disabled.
- Keep graceful fallback when external providers fail.

## High-Risk Change Areas
- Cache key composition
- Candidate blending weights
- MMR and post-filter ordering
- Reranker integration and fallback
- Query override and intent override behavior
- Query-family heuristics and rewrite normalization

## Anti-Overfit Notes
- Keep product logic independent from exact benchmark case wording.
- Prefer semantic detectors that should match paraphrases and neighboring queries.
- If a fix requires naming a benchmark phrase directly, treat it as temporary and refactor it before considering the work complete.

## Recommended Test Targets
- `tests/unit/test_recommend_route.py`
- `tests/unit/test_metrics.py`
- `tests/integration/test_routes.py`
