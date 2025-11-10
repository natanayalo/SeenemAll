.PHONY: up down logs sh migrate rev head alembic-init etl-tmdb embed etl-justwatch eval eval-report es-setup es-sync

EVAL_PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

up:
	docker compose up -d --build
down:
	docker compose down -v
logs:
	docker compose logs -f --tail=200 api
sh:
	docker compose exec api bash

test:
	docker compose exec api pytest

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
	$(EVAL_PYTHON) -m evaluation.evaluate --k=10 --set evaluation/evaluation_set.json

eval-report:
	@if [ -f evaluation/evaluation_set.titles.json ]; then \
		$(EVAL_PYTHON) -m evaluation.evaluate --k=10 --resolve-titles --titles-set evaluation/evaluation_set.titles.json; \
	else \
		$(EVAL_PYTHON) -m evaluation.evaluate --k=10 --set evaluation/evaluation_set.json; \
	fi
	@if [ -f evaluation/report.html ]; then \
		echo "Report ready at evaluation/report.html (open it with your browser)."; \
	else \
		echo "evaluation/report.html not generated (check Evidently installation)"; \
	fi

es-setup:
	docker compose exec api python scripts/setup_elasticsearch.py $(if $(FORCE),--force,)

es-sync:
	docker compose exec api python scripts/run_elasticsearch_sync.py \
		$(if $(BATCH),--batch-size $(BATCH),) \
		$(if $(MAX),--max-items $(MAX),) \
		$(if $(SINCE),--since $(SINCE),) \
		$(if $(EMBED_VERSION),--embed-version $(EMBED_VERSION),) \
		$(if $(REFRESH),--refresh,)
