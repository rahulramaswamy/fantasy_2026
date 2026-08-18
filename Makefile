.PHONY: install test lint fmt typecheck data clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

# Pull all upstream data into the local cache
data:
	ff data sync

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
