# ARC - Makefile
# Recipes are deliberately one command per line so they run unchanged under
# both /bin/sh (CI) and cmd.exe (Windows GNU make).

UV ?= uv
DATABASE_URL ?= postgresql://arc:arc@localhost:5432/arc
export DATABASE_URL

.DEFAULT_GOAL := help
.PHONY: help up down test lint fmt migrate validate console demo demo-live demo-adversarial

help:
	@echo ARC targets: up down migrate test lint fmt validate console demo demo-live demo-adversarial

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

# ---------------------------------------------------------------------------
# M14 console and M17 demo.
#
# SEED defaults to 3, the judged seed. `demo` is the deterministic replay: it
# reads no clock and prints no wall time, so three consecutive runs produce
# byte-identical output. `demo-live` is the same script with pauses to narrate
# into and is NOT byte-stable, which is why it is a separate target rather
# than a flag on the first.
# ---------------------------------------------------------------------------
SEED ?= 3
SIZE ?= 1200
CYCLES ?= 4

console:
	$(UV) run python -m arc.console.build --seed $(SEED) --size $(SIZE) --cycles $(CYCLES) --out console

demo:
	$(UV) run python -m arc.demo.run --seed $(SEED) --size $(SIZE) --cycles $(CYCLES)

demo-live:
	$(UV) run python -m arc.demo.run --seed $(SEED) --size $(SIZE) --cycles $(CYCLES) --live

demo-adversarial:
	$(UV) run python -m arc.demo.run --adversarial

demo-digest:
	$(UV) run python -m arc.demo.run --seed $(SEED) --size $(SIZE) --cycles $(CYCLES) --digest-only
