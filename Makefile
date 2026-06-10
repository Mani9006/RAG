.PHONY: install test lint demo eval reset serve docker data

install:        ## install package + dev tooling
	pip install -e ".[dev]"

test:           ## run the test suite
	python -m pytest

lint:           ## lint with ruff
	ruff check src tests scripts

demo: reset     ## run the full pipeline offline (zero cost) and show results
	sentinel run
	@echo "\n=== INCIDENTS ==="
	sentinel incidents
	@echo "\n=== APPROVAL QUEUE ==="
	sentinel approvals

eval:           ## run the agent evaluation harness (fails on gate regression)
	sentinel eval

reset:          ## rebuild the local store from the repo dataset
	sentinel reset

serve:          ## start the control-plane API on :8000
	sentinel serve

docker:         ## build and start the containerized service
	docker compose up --build

data:           ## regenerate the synthetic dataset (deterministic)
	python scripts/generate_data.py
