# Evaluation Set V2 - Milestone A Backlog

Milestone A target: 30 cases (`m1`)
Current count: 30
Cases to add: 0

Status: Completed (all m1 bucket and slice deficits closed).

## Deficits From `check_distribution.py --milestone m1`
- none

## Added Cases
- `m1_case_023` to `m1_case_030`

## Labeling Rules Per New Case
- `golden_set`: at least 5 items with `id`, `title`, `media_type`, `relevance`.
- `negative_set`: at least 3 items for constrained/profile/edge cases.
- Every item must include explicit `media_type` (`movie` or `tv`).
- Avoid duplicate `(id, media_type)` within a case.

## Completion Gate For Milestone A
1. `python evaluation/validate_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json`
2. `python evaluation/check_distribution.py --evaluation-set evaluation/evaluation_set_v2.json --milestone m1 --strict`
