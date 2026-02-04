# ForecastingWindPower 🌬️⚡

This project provides an end-to-end pipeline for **regional wind power forecasting** in Norway, using:

- MET weather **forecast + nowcast** data  
- Wind park metadata (capacity + bidding zones)  
- Statnett regional wind power production (ground truth)  
- Deep learning models (RNN/GRU-based forecasting)

The goal is to build a fully aligned dataset at the bidding-zone level (NO1–NO4) and train sequence models for short-term wind power prediction.

---

## Repository Structure

```bash
ForecastingWindPower/
│
├── backend/          # API backend (FastAPI)
├── frontend/         # Web dashboard (Vue)
├── ml/               # Machine learning pipeline
│   ├── src/
│   │   ├── features/     # Data alignment + preprocessing
│   │   ├── models/       # Training code
│   │   ├── tests/        # Unit tests
│   │   └── utils/        # Helper functions
│   │
│   ├── raw_data/         # Original datasets (parquet/csv)
│   ├── processed_data/   # Generated aligned datasets
│   └── notebooks/        # Development notebooks
│
├── main.py           # Project CLI entrypoint
├── Makefile          # Developer commands
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Setup
```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## CLI Usage
The repository includes a unified CLI entrypoint:
python3 main.py model <command>

# Data Alignment
```bash
python3 main.py model align
```

# Data Preprocessing
```bash
python3 main.py model preprocess
```

# Train Model
```bash
python3 main.py model train
```

# Run Tests
```bash
python3 main.py model test
```

## Makefile

For convenience, the project includes a `Makefile` with common development commands:

```makefile
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
```