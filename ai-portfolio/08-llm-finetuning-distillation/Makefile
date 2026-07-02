.PHONY: setup data run test bench screenshots all clean

PY ?= python3
export MPLBACKEND = Agg

setup:  ## Install dependencies (all are in the shared portfolio env already)
	$(PY) -m pip install -r requirements.txt

data:  ## Materialize synthetic instruction splits to data/
	$(PY) scripts/generate_data.py --rows 20000 --domain source
	$(PY) scripts/generate_data.py --rows 20000 --domain target

run:  ## Train base/LoRA/teacher/student + gate, write benchmarks/results.json
	$(PY) scripts/run_pipeline.py

bench:  ## Bounded-memory data-generator scaling benchmark
	$(PY) scripts/benchmark_generator.py --rows 1000000

test:  ## Run the pytest suite (real assertions)
	$(PY) -m pytest -q

screenshots:  ## Render the 4 PNGs into assets/ (runs pipeline if needed)
	$(PY) scripts/make_screenshots.py

all: run screenshots test  ## Full pipeline + charts + tests

clean:
	rm -rf data/*.parquet benchmarks/results.json __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
