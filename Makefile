.PHONY: help up down logs test migrate shell

help:
	@echo "Available commands:"
	@echo "  make up        - Start all services in development mode"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - Tail logs of all services"
	@echo "  make test      - Run pytest inside backend container"
	@echo "  make migrate   - Run Alembic migrations"
	@echo "  make shell     - Open a shell in the backend container"

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	docker-compose exec backend pytest

migrate:
	docker-compose exec backend alembic upgrade head

shell:
	docker-compose exec backend bash