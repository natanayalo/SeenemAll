# Evaluation Set V2 Quality Backlog

Generated from `evaluation/artifacts/snapshots/evaluation_set_audit.json` on 2026-03-14.

## Current Quality Snapshot
- Total cases: 72
- Weak cases (`golden_set < 4`): 9
- Cases with negatives: 31/72 (43.1%)
- High-overlap pairs (`jaccard >= 0.60`): 30 (many exact clones)
- Missing metadata (`case_id`, `distribution_bucket`): 0

## Priority 1: Raise Weak Cases to Minimum Signal
Target: every case has `golden_set >= 4` (prefer 5 for semantic buckets).

Cases to expand:
- `v2_case_003`
- `v2_case_007`
- `v2_case_011`
- `v2_case_013`
- `v2_case_014`
- `v2_case_015`
- `v2_case_016`
- `v2_case_017`
- `v2_case_018`

## Priority 2: Reduce Duplicate Label Clusters
Target: no exact-clone golden sets (`jaccard = 1.0`) across distinct cases.

Start with these clusters:
- `v2_case_001`, `m2_case_031`, `m3_case_059`
- `m1_case_019`, `m1_case_022`, `m1_case_027`, `m2_case_034`, `m2_case_043`, `m3_case_061`, `m3_case_063`
- `m1_case_020`, `m1_case_023`, `m1_case_025`, `m1_case_029`, `m2_case_033`, `m2_case_036`
- `m1_case_021`, `m1_case_028`, `m2_case_040`, `m2_case_049`, `m3_case_072`

For each cluster:
- keep one canonical case unchanged,
- diversify the others with at least 2 unique positives that fit their query/slice.

## Priority 3: Improve Negative Coverage Where It Is Too Low
Current low-coverage buckets:
- `semantic_query_movie`: 7.1%
- `semantic_query_tv`: 7.1%
- `no_query_warm_start`: 0%
- `no_query_cold_start`: 0%

Target:
- semantic buckets >= 50% cases include `negative_set`
- no-query buckets >= 33% cases include `negative_set`

## Acceptance Criteria
Run:

```powershell
.venv\Scripts\python.exe evaluation/validate_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/check_distribution.py --evaluation-set evaluation/evaluation_set_v2.json --targets evaluation/evaluation_set_v2_targets.json --milestone v2 --strict
.venv\Scripts\python.exe evaluation/audit_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json --strict --min-golden 4
```

Pass conditions:
- validation passes
- distribution remains green (72/72 + bucket/slice quotas)
- audit strict passes (`missing metadata = 0`, `weak cases = 0`)
