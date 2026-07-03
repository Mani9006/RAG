.PHONY: setup data run test bench screenshots all clean

PY ?= python3
export MPLBACKEND=Agg

setup:  ## Install dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## Generate synthetic baseline + drifted datasets (parquet)
	$(PY) scripts/generate_data.py --rows 50000

run:  ## Run the end-to-end lifecycle simulation
	$(PY) scripts/run_lifecycle.py

test:  ## Run the pytest suite
	$(PY) -m pytest tests/ -q

bench:  ## Run the scoring/monitoring scaling benchmark (1M/10M sweep)
	$(PY) benchmarks/bench_scoring.py --sweep --batch 50000

screenshots:  ## Render the 4 dashboard PNGs from the last run
	$(PY) scripts/make_screenshots.py

all: run screenshots test  ## Full pipeline: simulate -> render -> test

clean:  ## Remove generated artifacts (keeps committed assets)
	rm -f data/*.duckdb data/*.json data/*.csv data/*.parquet
	rm -rf data/artifacts
