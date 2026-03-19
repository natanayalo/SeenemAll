# Evaluation Workflow Checklist

## Preconditions
- Use `.venv\Scripts\python.exe` for all evaluation commands.
- In this host environment, use HTTP mode: `--base-url http://localhost:8000/recommend` (default from config).
- Do not use `--base-url local:///recommend` from host shell when DB host is `db`; that hostname resolves only inside Docker network.
- Use local mode only when running from a context where DB connectivity is valid (for example inside the Docker network, or with a host-resolvable DB URL override).
- Keep baseline and candidate runs on the same evaluation set and config.

## Dataset Hygiene
```powershell
.venv\Scripts\python.exe evaluation/validate_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/normalize_evaluation_set.py --input evaluation/evaluation_set_v2.json
.venv\Scripts\python.exe evaluation/check_distribution.py --evaluation-set evaluation/evaluation_set_v2.json --milestone v2 --strict
.venv\Scripts\python.exe evaluation/audit_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json
```

## Run Evaluator
```powershell
.venv\Scripts\python.exe evaluation/evaluate.py --config evaluation/evaluation_config.json
```
Outputs:
- `evaluation/artifacts/evaluation_results_latest.csv`
- `evaluation/artifacts/evaluation_summary_latest.json`

## Baseline vs Candidate Gate
```powershell
.venv\Scripts\python.exe evaluation/evaluate.py `
  --config evaluation/evaluation_config.json `
  --output-csv evaluation/artifacts/gate/baseline_results.csv `
  --summary-json evaluation/artifacts/gate/baseline_summary.json

.venv\Scripts\python.exe evaluation/evaluate.py `
  --config evaluation/evaluation_config.json `
  --compare-csv evaluation/artifacts/gate/baseline_results.csv `
  --per-query-diff-json evaluation/artifacts/gate/candidate_per_query_diff.json `
  --output-csv evaluation/artifacts/gate/candidate_results.csv `
  --summary-json evaluation/artifacts/gate/candidate_summary.json

.venv\Scripts\python.exe evaluation/check_release_gate.py `
  --baseline-summary evaluation/artifacts/gate/baseline_summary.json `
  --candidate-summary evaluation/artifacts/gate/candidate_summary.json `
  --baseline-csv evaluation/artifacts/gate/baseline_results.csv `
  --candidate-csv evaluation/artifacts/gate/candidate_results.csv `
  --significance-mode strict `
  --output-json evaluation/artifacts/gate/gate_report.json
```

## Failure Pattern Analysis
```powershell
.venv\Scripts\python.exe evaluation/analyze_failure_patterns.py `
  --results-csv evaluation/artifacts/gate/candidate_results.csv `
  --evaluation-set evaluation/evaluation_set_v2.json `
  --scenario default
```
Outputs:
- `evaluation/artifacts/gate/failure_patterns_report.json`
- `evaluation/artifacts/gate/failure_patterns_report.md`

## Triage Rules
- Prioritize cohorts with highest `zero_hit_count` and high `zero_hit_rate`.
- Split recoverable vs non-recoverable misses:
  - Recoverable: scenario toggles already recover hits (weights/flags tuning).
  - Non-recoverable: require pipeline logic or data quality changes.
- Flag likely transport failures separately from ranking quality (example signals: `latency_ms == 0`, empty rec list on isolated case).
- Report improvements only with absolute deltas and sample counts per slice.
- Separate benchmark-quality fixes from ranking-quality fixes in all reporting.
- If code changes closely mirror a benchmark phrase, add a paraphrase check before considering the fix valid.
- Treat title-level or ID-level boosts keyed to benchmark items as disallowed production behavior.

## Reporting Template
- Run mode: `<http|local>`
- Baseline scenario: `<name>`
- Candidate scenario: `<name>`
- Overall deltas: `<P@10, R@10, MAP, nDCG@10, HitRate@10, MRR, latency>`
- Gate status: `<pass/fail>`
- Concentrated zero-hit cohorts: `<top buckets/slices with counts>`
- Next fixes: `<ranked list with expected impact>`

