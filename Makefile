.PHONY: up down logs sh migrate rev head alembic-init etl-tmdb embed etl-justwatch eval health metrics debug-rec

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
	docker compose exec api python evaluation/evaluate.py

health:
	curl -s http://localhost:8000/healthz

metrics:
	curl -s http://localhost:8000/healthz/metrics | { python3 -m json.tool || cat; }

debug-rec:
	curl -s "http://localhost:8000/recommend/debug?user_id=u1&limit=5" | { python3 -m json.tool || cat; }
