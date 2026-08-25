# Local dev shortcuts for the Vibe CAD backend.
# Usage: make <target>   (run from vibecad_mvp/backend)
PY := .venv/bin/python

.PHONY: help install lint fmt test cov check

help:
	@echo "install  - create .venv and install dev dependencies"
	@echo "lint     - ruff static checks"
	@echo "fmt      - ruff auto-fix (imports, simple lints)"
	@echo "test     - run the pytest suite"
	@echo "cov      - run tests with coverage (fails under 95%)"
	@echo "check    - lint + cov (what CI runs)"

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

lint:
	$(PY) -m ruff check app tests

fmt:
	$(PY) -m ruff check --fix app tests

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov --cov-config=.coveragerc

check: lint cov
