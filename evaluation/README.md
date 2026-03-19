# Evaluation Workspace

Use `evaluation/` for source files only. Generated files go under `evaluation/artifacts/`.

## Source Files
- `evaluation_set.json`: legacy golden queries and labels (kept for reference).
- `evaluation_set_v2.json`: active v2 evaluation set (72 cases).
- `evaluation_set_v2_targets.json`: distribution quotas and milestone targets.
- `v2_m1_backlog.md`: concrete backlog for Milestone A (18 -> 30).
- `v2_m2_backlog.md`: concrete backlog for Milestone B (30 -> 50).
- `v2_quality_backlog.md`: post-v2 quality hardening backlog from audit findings.
- `evaluation_config.json`: runtime config (API URL, metric cutoffs, param grid, judge provider).
- `evaluation_split_manifest_v1.json`: stable dev/holdout split assignments for v2 cases.
- `evaluate.py`: offline evaluator.
- `create_split_manifest.py`: deterministic dev/holdout manifest generator.
- `validate_evaluation_set.py`: schema + catalog consistency checks.
- `check_distribution.py`: milestone quota checker for v2 set.
- `normalize_evaluation_set.py`: backfills stable `case_id`/`distribution_bucket` metadata.
- `check_release_gate.py`: baseline vs candidate regression gate.
- `analyze_failure_patterns.py`: ranks failure buckets/slices/cases from gate outputs.
- `audit_evaluation_set.py`: dataset quality audit (golden density, overlap, negatives).
- `capture_baseline.py`: captures dated baseline report.

## Artifacts Layout
- `evaluation/artifacts/evaluation_results_latest.csv`
- `evaluation/artifacts/evaluation_summary_latest.json`
- `evaluation/artifacts/snapshots/` (dated evaluator outputs)
- `evaluation/artifacts/gate/` (baseline/candidate + gate report)
- `evaluation/artifacts/baselines/` (baseline markdown snapshots)

## Typical Commands
```powershell
.venv\Scripts\python.exe evaluation/validate_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/normalize_evaluation_set.py --input evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/create_split_manifest.py --evaluation-set evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/evaluate.py --config evaluation/evaluation_config.json
.venv\Scripts\python.exe evaluation/evaluate.py --config evaluation/evaluation_config.json --split holdout
.venv\Scripts\python.exe evaluation/check_distribution.py --evaluation-set evaluation/evaluation_set_v2.json --milestone v2 --strict
.venv\Scripts\python.exe evaluation/audit_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/analyze_failure_patterns.py --results-csv evaluation/artifacts/gate/candidate_results.csv --scenario default
.venv\Scripts\python.exe evaluation/cleanup_artifacts.py
```

Local app mode (uses in-process FastAPI app instead of HTTP host):
```powershell
.venv\Scripts\python.exe evaluation/evaluate.py --base-url local:///recommend
```
Note: local mode still uses your configured database URL, so DB connectivity must work from the current process.

Using `make` locally:
```powershell
make eval EVAL_CONFIG=evaluation/evaluation_config.json
make eval-normalize-set
make eval-baseline EVAL_CONFIG=evaluation/evaluation_config.json
make eval-validate-set
make eval-gate EVAL_CONFIG=evaluation/evaluation_config.json
make eval-analyze
make eval-audit-set
make eval-check-dist DIST_MILESTONE=m1
make eval-check-dist DIST_MILESTONE=m2
```

Windows (no `make` installed):
```powershell
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task validate
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task normalize
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task dist
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task dist -Milestone m2
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task audit
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task split
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task eval
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task baseline
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task gate
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task analyze
```

## Label Schema (Optional)
- `golden_set[]` supports `relevance` in range `0..3` for graded ranking quality.
- `negative_set[]` defines hard negatives that should not appear in top-K.
- If `relevance` is omitted, evaluator defaults to `1.0` (binary positive label).
- Matching uses `(tmdb_id, media_type)` to avoid movie/TV ID collisions.
- For `evaluation_set_v2.json`, keep `case_id` and `distribution_bucket` populated
  (`make eval-normalize-set` backfills missing values deterministically).

## Stable Holdout
- Use `evaluation/evaluation_split_manifest_v1.json` to keep a fixed dev/holdout partition.
- `evaluation/evaluate.py` supports `--split-manifest` and `--split` so you can run only `dev` or only `holdout`.
- The default manifest generator stratifies by `distribution_bucket`, uses a deterministic seed, and keeps at least one case on each side for buckets with two or more entries.

## Judge Provider
- Current implementation: `offline_golden_set` (deterministic label matching).
- `evaluation_config.json` includes a `judge` block for future providers (for example OpenAI), but non-offline providers are intentionally rejected for now.

## Release Gate (Example)
Generate baseline and candidate CSVs with the current evaluator version; legacy CSV schemas are not supported.

```powershell
.venv\Scripts\python.exe evaluation/evaluate.py `
  --output-csv evaluation/artifacts/gate/baseline_results.csv `
  --summary-json evaluation/artifacts/gate/baseline_summary.json

.venv\Scripts\python.exe evaluation/evaluate.py `
  --compare-csv evaluation/artifacts/gate/baseline_results.csv `
  --per-query-diff-json evaluation/artifacts/gate/candidate_per_query_diff.json `
  --output-csv evaluation/artifacts/gate/candidate_results.csv `
  --summary-json evaluation/artifacts/gate/candidate_summary.json

.venv\Scripts\python.exe evaluation/check_release_gate.py `
  --baseline-summary evaluation/artifacts/gate/baseline_summary.json `
  --candidate-summary evaluation/artifacts/gate/candidate_summary.json `
  --output-json evaluation/artifacts/gate/gate_report.json
```
