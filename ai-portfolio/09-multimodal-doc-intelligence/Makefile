# Multimodal Document Intelligence
PY ?= python
export PYTHONPATH := src
export MPLBACKEND := Agg

DOCS        ?= 100000
TRAIN_DOCS  ?= 8000
EVAL_DOCS   ?= 2000

.PHONY: setup data run test bench screenshots all clean

setup:
	$(PY) -m pip install -r requirements.txt

## generate a large synthetic corpus of token layouts -> data/tokens.parquet
data:
	$(PY) scripts/generate_data.py --docs $(DOCS) --out data/tokens.parquet

## train + evaluate the full pipeline; writes data/metrics.json + samples.json
run:
	$(PY) scripts/run_pipeline.py --train-docs $(TRAIN_DOCS) --eval-docs $(EVAL_DOCS)

## unit + behavioural tests
test:
	$(PY) -m pytest tests/ -q

## streaming-scale + inference throughput benchmark -> benchmarks/results.*
bench:
	$(PY) scripts/benchmark.py --scales 2000 10000 50000 100000

## render the four PNG screenshots into assets/
screenshots:
	$(PY) scripts/make_screenshots.py

## full reproduction: pipeline -> benchmark -> screenshots
all: run bench screenshots

clean:
	rm -f data/*.parquet data/*.json
	rm -f benchmarks/results.csv benchmarks/results.md
	rm -f assets/*.png
