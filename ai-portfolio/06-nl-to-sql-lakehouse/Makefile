# NL-to-SQL Lakehouse
# Offline, deterministic. No paid APIs. MPLBACKEND=Agg for headless charts.

PY ?= python
export PYTHONPATH := src
export MPLBACKEND := Agg

ROWS ?= 50000000

.PHONY: help setup data run test bench screenshots all clean

help:
	@echo "make setup       - install requirements"
	@echo "make data        - build the star-schema lakehouse ($(ROWS) rows) to data/lakehouse"
	@echo "make run         - run the NL-to-SQL suite end-to-end vs DuckDB"
	@echo "make test        - run pytest"
	@echo "make bench       - query-latency benchmark across scales"
	@echo "make screenshots - render the 4 PNGs into assets/"
	@echo "make all         - data + run + bench + screenshots"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/generate_data.py --rows $(ROWS)

run:
	$(PY) scripts/run_suite.py

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/run_benchmark.py

screenshots:
	$(PY) scripts/make_screenshots.py

all: data run bench screenshots

clean:
	rm -rf data/lakehouse data/_bench_* __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
