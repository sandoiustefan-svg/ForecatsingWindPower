.PHONY: help setup lint format fmt test testv precommit clean ci

help:
	@echo "Targets:"
	@echo "  setup      Install deps + pre-commit"
	@echo "  lint       Ruff lint (no fixes)"
	@echo "  format     Ruff format + auto-fix lint"
	@echo "  test       Run pytest"
	@echo "  testv      Run pytest verbose"
	@echo "  precommit  Run pre-commit on all files"
	@echo "  clean      Remove caches/build artifacts"
	@echo "  ci         Run lint + tests (like CI)"

setup:
	cd ml && python -m pip install --upgrade pip
	cd ml && pip install -r requirements.txt
	cd ml && pip install ruff pytest pre-commit
	pre-commit install

lint:
	cd ml && ruff check .

format:
	cd ml && ruff check . --fix
	cd ml && ruff format .

fmt: format

test:
	cd ml && pytest -q

testv:
	cd ml && pytest -vv

precommit:
	pre-commit run --all-files

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +

ci: lint test
