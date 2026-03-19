param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("normalize", "validate", "dist", "audit", "split", "eval", "baseline", "gate", "analyze", "clean")]
    [string]$Task,

    [string]$Config = "evaluation/evaluation_config.json",
    [string]$Milestone = "m1",
    [string]$SplitOutput = "evaluation/evaluation_split_manifest_v1.json",
    [double]$HoldoutRatio = 0.2,
    [int]$Seed = 42
)

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

switch ($Task) {
    "normalize" {
        & $python "evaluation/normalize_evaluation_set.py" "--input" "evaluation/evaluation_set_v2.json"
    }
    "validate" {
        & $python "evaluation/validate_evaluation_set.py" "--evaluation-set" "evaluation/evaluation_set_v2.json"
    }
    "dist" {
        $report = "evaluation/artifacts/snapshots/distribution_$Milestone`_report.json"
        & $python "evaluation/check_distribution.py" "--evaluation-set" "evaluation/evaluation_set_v2.json" "--targets" "evaluation/evaluation_set_v2_targets.json" "--milestone" $Milestone "--report-json" $report
    }
    "audit" {
        & $python "evaluation/audit_evaluation_set.py" "--evaluation-set" "evaluation/evaluation_set_v2.json" "--output-json" "evaluation/artifacts/snapshots/evaluation_set_audit.json" "--output-md" "evaluation/artifacts/snapshots/evaluation_set_audit.md"
    }
    "split" {
        & $python "evaluation/create_split_manifest.py" "--evaluation-set" "evaluation/evaluation_set_v2.json" "--output" $SplitOutput "--holdout-ratio" $HoldoutRatio "--seed" $Seed
    }
    "eval" {
        & $python "evaluation/evaluate.py" "--config" $Config
    }
    "baseline" {
        & $python "evaluation/capture_baseline.py" "--config" $Config
    }
    "gate" {
        & $python "evaluation/evaluate.py" "--config" $Config "--bootstrap-samples" "0" "--output-csv" "evaluation/artifacts/gate/baseline_results.csv" "--summary-json" "evaluation/artifacts/gate/baseline_summary.json"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python "evaluation/evaluate.py" "--config" $Config "--bootstrap-samples" "0" "--compare-csv" "evaluation/artifacts/gate/baseline_results.csv" "--per-query-diff-json" "evaluation/artifacts/gate/candidate_per_query_diff.json" "--output-csv" "evaluation/artifacts/gate/candidate_results.csv" "--summary-json" "evaluation/artifacts/gate/candidate_summary.json"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $python "evaluation/check_release_gate.py" "--baseline-summary" "evaluation/artifacts/gate/baseline_summary.json" "--candidate-summary" "evaluation/artifacts/gate/candidate_summary.json" "--baseline-csv" "evaluation/artifacts/gate/baseline_results.csv" "--candidate-csv" "evaluation/artifacts/gate/candidate_results.csv" "--significance-mode" "strict" "--output-json" "evaluation/artifacts/gate/gate_report.json"
    }
    "analyze" {
        & $python "evaluation/analyze_failure_patterns.py" "--results-csv" "evaluation/artifacts/gate/candidate_results.csv" "--evaluation-set" "evaluation/evaluation_set_v2.json" "--scenario" "default" "--output-json" "evaluation/artifacts/gate/failure_patterns_report.json" "--output-md" "evaluation/artifacts/gate/failure_patterns_report.md"
    }
    "clean" {
        & $python "evaluation/cleanup_artifacts.py"
    }
}

exit $LASTEXITCODE
