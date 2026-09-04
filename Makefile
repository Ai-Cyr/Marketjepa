.PHONY: install test lint format train-synthetic

install:
	uv pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

train-synthetic:
	marketjepa-train --config configs/default.yaml --synthetic-steps 20000 --epochs 2
