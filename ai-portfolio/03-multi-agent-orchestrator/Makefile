# Multi-Agent Orchestrator — developer entrypoints.
# Everything runs offline and deterministically.

PY ?= python3
export PYTHONPATH := src
export MPLBACKEND := Agg

.PHONY: help setup data run test bench screenshots all clean

help:
	@echo "make setup        - install dependencies (requirements.txt)"
	@echo "make run          - solve the task suite, print a summary"
	@echo "make test         - run the pytest suite"
	@echo "make bench        - run benchmarks, write results.csv/.md + trace"
	@echo "make screenshots  - render assets/*.png from a real run"
	@echo "make all          - test + bench + screenshots"
	@echo "make clean        - remove caches and generated data"

setup:
	$(PY) -m pip install -r requirements.txt

# 'data' is a no-op alias: this project generates its knowledge base in-code.
data: bench

run:
	$(PY) benchmarks/tasks.py
	$(PY) scripts/run_benchmark.py

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/run_benchmark.py

screenshots:
	$(PY) scripts/make_screenshots.py

all: test bench screenshots

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__ tests/__pycache__
	rm -f data/trace_showcase.json
