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
4. Add tests that pin intentional ranking shifts.
5. Document expected impact and rollback plan.

## Guardrails
- Avoid unbounded tuning without measurable targets.
- Do not remove fallback behavior during experiments.
- Keep cache key logic aligned with new tunable inputs.
- Prefer small deltas and iterative validation.
- Avoid benchmark-phrase tuning in production defaults. If a change starts from eval failures, rewrite it into a reusable semantic hypothesis before shipping it.
- Keep baseline behavior reachable and comparable so suspected overfit can be reverted quickly.
- Prefer experiments that can be validated on both benchmark cases and paraphrase/adjacent live-style queries.

Use [references/checklist.md](references/checklist.md) for command sequence.
