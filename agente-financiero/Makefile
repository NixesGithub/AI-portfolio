.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Crea el entorno virtual e instala dependencias de desarrollo
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements-dev.txt

test: ## Ejecuta los tests (no necesitan red ni claves)
	$(VENV)/bin/pytest -q

run: ## Levanta la API en local con recarga automática
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

up: ## Levanta la API en Docker
	docker compose up --build

down: ## Para y limpia el entorno de Docker
	docker compose down

demo: ## Conversación de ejemplo contra la API ya levantada
	./scripts/demo.sh

.PHONY: help install test run up down demo
