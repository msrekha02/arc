# ARC - Makefile
# Recipes are deliberately one command per line so they run unchanged under
# both /bin/sh (CI) and cmd.exe (Windows GNU make).

UV ?= uv
DATABASE_URL ?= postgresql://arc:arc@localhost:5432/arc
export DATABASE_URL

.DEFAULT_GOAL := help
.PHONY: help up down test lint fmt migrate validate

help:
	@echo ARC targets: up down migrate test lint fmt validate

up:
	docker compose up -d --wait
	docker compose ps

down:
	docker compose down

migrate:
	$(UV) run python scripts/migrate.py

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

validate:
	$(UV) run python -m arc.simulator.validate
