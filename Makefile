# ============================================================
# AgentDisruptBench — Makefile
# ============================================================

.PHONY: help install install-dev test lint format typecheck clean eval eval-quick docker

PYTHON ?= python3
PIP ?= pip

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ----- Installation -----

install: ## Install core package
	$(PIP) install -e .

install-dev: ## Install with all development dependencies
	$(PIP) install -e ".[dev,all,cli]"

# ----- Quality -----

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=python/agentdisruptbench --cov-report=term-missing --cov-report=html

lint: ## Run ruff linter
	ruff check python/ evaluation/ tests/

format: ## Format code with ruff
	ruff format python/ evaluation/ tests/
	ruff check --fix python/ evaluation/ tests/

typecheck: ## Run mypy type checker
	mypy python/agentdisruptbench/

quality: lint typecheck test ## Run all quality checks

# ----- Evaluation -----

eval: ## Run full benchmark (simple baseline, all profiles)
	$(PYTHON) -m evaluation.run_benchmark --runner simple --profiles clean mild_production moderate_production hostile_environment --seeds 42 123 456

eval-quick: ## Run quick smoke test (simple baseline, clean only)
	$(PYTHON) -m evaluation.run_benchmark --runner simple --profiles clean --max-difficulty 3

eval-openai: ## Run benchmark with OpenAI GPT-4o
	$(PYTHON) -m evaluation.run_benchmark --runner openai --model gpt-4o --profiles clean moderate_production hostile_environment --seeds 42 123 456

eval-langchain: ## Run benchmark with LangChain
	$(PYTHON) -m evaluation.run_benchmark --runner langchain --profiles clean moderate_production hostile_environment --seeds 42 123 456

# ----- Maintenance -----

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-runs: ## Remove all evaluation run outputs
	rm -rf runs/

# ----- Croissant -----

validate-croissant: ## Validate Croissant metadata
	$(PYTHON) -c "from mlcroissant import Dataset; Dataset(jsonld='croissant.json'); print('✅ Croissant metadata is valid')"

# ----- Docker -----

docker-build: ## Build reproducibility Docker image
	docker build -t agentdisruptbench:latest .

docker-test: ## Run tests inside Docker
	docker run --rm agentdisruptbench:latest make test
