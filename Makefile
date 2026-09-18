.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lock lint format typecheck test validate prompts check bench clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create venv, install project + dev tools, install git hooks
	$(UV) sync
	$(UV) run pre-commit install

lock: ## Refresh uv.lock after dependency changes
	$(UV) lock

lint: ## Lint and check formatting (no changes)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-fix lint issues and format code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Static type check
	$(UV) run mypy

test: ## Run unit tests with coverage
	$(UV) run pytest --cov=infer_bench --cov-report=term-missing

validate: ## Validate engine configs, suite, and the committed prompt set
	$(UV) run infer-bench validate

prompts: ## Regenerate the tokenized prompt set (downloads tokenizer + WikiText-2 train)
	$(UV) run --group prompts python scripts/build_prompts.py

check: lint typecheck test validate ## Everything CI runs

bench: ## Run the full benchmark suite
	@echo "bench: not implemented yet (harness lands Day 4)" >&2
	@exit 1

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
