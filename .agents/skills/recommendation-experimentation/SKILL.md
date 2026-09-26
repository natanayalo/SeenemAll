---
name: recommendation-experimentation
description: Run safe recommendation tuning experiments for weights, flags, and ranking behavior. Use when adjusting mixer weights, ANN/collab/trending blends, MMR diversity settings, reranker strategy, or feature flags for recommendation quality.
---
# Recommendation Experimentation Skill

Use this skill when tuning recommendation behavior.

## Fast Path
1. Define hypothesis and success metrics before code changes.
2. Scope changes behind explicit flags or parameters when possible.
3. Keep baseline behavior reachable.
4. Benchmark candidate against stored baseline (`evaluation/baseline.json`).
5. Add tests that pin intentional ranking shifts (maintain >= 85% coverage).
6. Store updated baseline snapshot (`make eval-baseline CONFIG=<new_standard>`) when establishing a new performance benchmark.
7. Document expected impact and rollback plan.

## Guardrails
- Avoid unbounded tuning without measurable targets.
- Do not remove fallback behavior during experiments.
- Keep cache key logic aligned with new tunable inputs.
- Prefer small deltas and iterative validation.

Use [references/checklist.md](references/checklist.md) for command sequence.
