.PHONY: up down logs sh migrate rev head alembic-init etl-tmdb embed etl-justwatch eval eval-docker eval-normalize-set eval-validate-set eval-check-dist eval-audit-set eval-clean eval-baseline eval-gate eval-analyze health metrics debug-rec

PYTHON ?= .venv/Scripts/python.exe
EVAL_CONFIG ?= evaluation/evaluation_config.json
DIST_MILESTONE ?= m1

up:
	docker compose up -d --build
down:
	docker compose down -v
logs:
	docker compose logs -f --tail=200 api
sh:
	docker compose exec api bash

alembic-init:
	docker compose exec api alembic init -t async migrations || true

migrate:
	docker compose exec api alembic upgrade head

rev:
	docker compose exec api alembic revision --autogenerate -m "$(m)"

head:
	docker compose exec api alembic heads

etl-tmdb:
	docker compose exec api python scripts/run_tmdb_sync.py

embed:
	docker compose exec api python -m etl.compute_embeddings

etl-justwatch:
	docker compose exec api python scripts/run_justwatch_sync.py

eval:
	$(PYTHON) evaluation/evaluate.py --config $(EVAL_CONFIG)

eval-docker:
	docker compose exec api python evaluation/evaluate.py --config $(EVAL_CONFIG)

eval-normalize-set:
	$(PYTHON) evaluation/normalize_evaluation_set.py --input evaluation/evaluation_set_v2.json

eval-validate-set:
	$(PYTHON) evaluation/validate_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json

eval-check-dist:
	$(PYTHON) evaluation/check_distribution.py --evaluation-set evaluation/evaluation_set_v2.json --targets evaluation/evaluation_set_v2_targets.json --milestone $(DIST_MILESTONE) --report-json evaluation/artifacts/snapshots/distribution_$(DIST_MILESTONE)_report.json

eval-audit-set:
	$(PYTHON) evaluation/audit_evaluation_set.py --evaluation-set evaluation/evaluation_set_v2.json --output-json evaluation/artifacts/snapshots/evaluation_set_audit.json --output-md evaluation/artifacts/snapshots/evaluation_set_audit.md

eval-clean:
	$(PYTHON) evaluation/cleanup_artifacts.py

eval-baseline:
	$(PYTHON) evaluation/capture_baseline.py --config $(EVAL_CONFIG)

eval-gate:
	$(PYTHON) evaluation/evaluate.py --config $(EVAL_CONFIG) --bootstrap-samples 0 --output-csv evaluation/artifacts/gate/baseline_results.csv --summary-json evaluation/artifacts/gate/baseline_summary.json
	$(PYTHON) evaluation/evaluate.py --config $(EVAL_CONFIG) --bootstrap-samples 0 --compare-csv evaluation/artifacts/gate/baseline_results.csv --per-query-diff-json evaluation/artifacts/gate/candidate_per_query_diff.json --output-csv evaluation/artifacts/gate/candidate_results.csv --summary-json evaluation/artifacts/gate/candidate_summary.json
	$(PYTHON) evaluation/check_release_gate.py --baseline-summary evaluation/artifacts/gate/baseline_summary.json --candidate-summary evaluation/artifacts/gate/candidate_summary.json --baseline-csv evaluation/artifacts/gate/baseline_results.csv --candidate-csv evaluation/artifacts/gate/candidate_results.csv --significance-mode strict --output-json evaluation/artifacts/gate/gate_report.json

eval-analyze:
	$(PYTHON) evaluation/analyze_failure_patterns.py --results-csv evaluation/artifacts/gate/candidate_results.csv --evaluation-set evaluation/evaluation_set_v2.json --scenario default --output-json evaluation/artifacts/gate/failure_patterns_report.json --output-md evaluation/artifacts/gate/failure_patterns_report.md

health:
	curl -s http://localhost:8000/healthz

metrics:
	curl -s http://localhost:8000/healthz/metrics | { python3 -m json.tool || cat; }

debug-rec:
	curl -s "http://localhost:8000/recommend/debug?user_id=u1&limit=5" | { python3 -m json.tool || cat; }
