# Offline Release Gate

Use this workflow to decide whether a recommendation change improved quality.

## 1) Freeze baseline (known-good code)
```powershell
.venv\Scripts\python.exe evaluation/evaluate.py `
  --config evaluation/evaluation_config.json `
  --output-csv evaluation/artifacts/gate/baseline_results.csv `
  --summary-json evaluation/artifacts/gate/baseline_summary.json
```

## 2) Run candidate (after code changes)
```powershell
.venv\Scripts\python.exe evaluation/evaluate.py `
  --config evaluation/evaluation_config.json `
  --compare-csv evaluation/artifacts/gate/baseline_results.csv `
  --per-query-diff-json evaluation/artifacts/gate/candidate_per_query_diff.json `
  --output-csv evaluation/artifacts/gate/candidate_results.csv `
  --summary-json evaluation/artifacts/gate/candidate_summary.json
```

## 3) Enforce gate
```powershell
.venv\Scripts\python.exe evaluation/check_release_gate.py `
  --baseline-summary evaluation/artifacts/gate/baseline_summary.json `
  --candidate-summary evaluation/artifacts/gate/candidate_summary.json `
  --baseline-csv evaluation/artifacts/gate/baseline_results.csv `
  --candidate-csv evaluation/artifacts/gate/candidate_results.csv `
  --significance-mode strict `
  --output-json evaluation/artifacts/gate/gate_report.json
```

The gate script exits non-zero if regression thresholds are violated.

Windows shortcut:
```powershell
powershell -ExecutionPolicy Bypass -File evaluation/run.ps1 -Task gate
```
