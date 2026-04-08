# ─── 5G QoS Predictor Makefile ────────────────────────────────────────────────

.PHONY: help setup data features train evaluate dashboard api docker clean

# Default target
help:
	@echo "5G QoS Predictor - Available Commands:"
	@echo ""
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  data       - Download 5G-NIDD and generate synthetic data"
	@echo "  features   - Run feature engineering pipeline"
	@echo "  train      - Train all models"
	@echo "  evaluate   - Run full evaluation suite"
	@echo "  dashboard  - Launch Streamlit dashboard"
	@echo "  api        - Launch FastAPI server"
	@echo "  docker     - Build and run Docker containers"
	@echo "  clean      - Remove generated files and caches"
	@echo "  test       - Run test suite"
	@echo "  lint       - Run code linting"
	@echo "  notebook   - Launch Jupyter Lab"
	@echo ""

# ─── SETUP ────────────────────────────────────────────────────────────────────

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && python -m ipykernel install --user --name=5g-qos-predictor
	@echo "Setup complete. Activate with: source venv/bin/activate"

# ─── DATA PIPELINE ────────────────────────────────────────────────────────────

data: data-download data-generate

data-download:
	@echo "Downloading 5G-NIDD dataset..."
	python -m src.data.downloader

data-generate:
	@echo "Generating synthetic slice data..."
	python -m src.data.generator

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

features:
	@echo "Running feature engineering pipeline..."
	python -m src.features.builder

# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

train:
	@echo "Training models..."
	python -m src.models.classifier
	python -m src.models.forecaster

# ─── EVALUATION ───────────────────────────────────────────────────────────────

evaluate:
	@echo "Running evaluation..."
	python -m src.evaluation.analysis

# ─── SERVING ──────────────────────────────────────────────────────────────────

dashboard:
	@echo "Launching Streamlit dashboard..."
	streamlit run dashboard/app.py

api:
	@echo "Launching FastAPI server..."
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ─── DOCKER ───────────────────────────────────────────────────────────────────

docker:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

# ─── DEVELOPMENT ──────────────────────────────────────────────────────────────

notebook:
	jupyter lab --notebook-dir=notebooks

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	black src/ tests/ api/ dashboard/
	isort src/ tests/ api/ dashboard/
	flake8 src/ tests/ api/ dashboard/

# ─── CLEANUP ──────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned generated files and caches"

clean-data:
	rm -rf data/raw/generated/*
	rm -rf data/processed/*
	rm -rf data/splits/*
	@echo "Cleaned generated data files"

clean-models:
	rm -rf models/*
	@echo "Cleaned model artifacts"

clean-all: clean clean-data clean-models
	@echo "Full cleanup complete"