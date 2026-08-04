.PHONY: help install dev test lint format typecheck check clean build docs hooks

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode (core deps only)
	pip install -e .

dev:  ## Install package with all development dependencies + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

all:  ## Install package with all optional dependencies
	pip install -e ".[all]"

hooks:  ## Install pre-commit hooks (run once after cloning)
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-run:  ## Run all pre-commit hooks on all files
	pre-commit run --all-files

test:  ## Run test suite
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=quantspt --cov-report=term-missing --cov-report=html

lint:  ## Run linter (ruff check)
	ruff check quantspt/ tests/

lint-fix:  ## Run linter and auto-fix issues
	ruff check --fix quantspt/ tests/

format:  ## Auto-format code (ruff format)
	ruff format quantspt/ tests/

format-check:  ## Check formatting without modifying files
	ruff format --check quantspt/ tests/

typecheck:  ## Run mypy strict type checking
	mypy quantspt/ --strict

check: lint format-check typecheck test  ## Run all checks (lint + format + types + tests)

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

build: clean  ## Build distribution packages
	python -m build

docs:  ## Build documentation
	sphinx-build docs/ docs/_build/ -W --keep-going
