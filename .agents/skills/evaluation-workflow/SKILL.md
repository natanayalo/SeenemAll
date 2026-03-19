---
name: evaluation-workflow
description: Run and interpret SeenemAll offline recommendation evaluation, regression gates, and failure-pattern analysis. Use when comparing baseline vs candidate quality, checking slice regressions, or turning evaluation artifacts into prioritized fix recommendations.
---
# Evaluation Workflow Skill

Use this skill for repeatable evaluation runs and regression analysis.

## Fast Path
1. Validate and normalize the evaluation set before scoring.
2. Run evaluator with `evaluation/evaluation_config.json`.
3. If comparing changes, generate baseline and candidate gate artifacts.
4. Run release gate checks before recommending rollout.
5. Run failure-pattern analysis and prioritize fixes by concentration and recoverability.

## Guardrails
- Keep runs deterministic: same config, same evaluation set, same API mode.
- Treat transport failures separately from ranking failures.
- Do not claim quality improvements without gate outputs and per-slice checks.
- Prefer fixes that reduce concentrated zero-hit cohorts first.
- Treat eval-set cleanup and ranking improvements as separate classes of change.
- Do not accept a scoring gain at face value if it came mainly from benchmark tightening, provider drift, or unreachable-golden repair.
- When a ranking fix is driven by eval output, check whether it generalizes to paraphrases, nearby query families, and holdout before calling it product-safe.
- Flag likely phrase-level overfit when code changes mirror exact benchmark wording.

Use [references/checklist.md](references/checklist.md) for command sequence and triage rules.

