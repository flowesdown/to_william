.PHONY: help install install-dev test test-unit test-safety test-cov \
        lint format validate backtest calibrate paper clean

PYTHON  := python3
PYTEST  := $(PYTHON) -m pytest
ENV     := TOPARB_FORCE_CPU=1 TOPARB_PAPER_TRADING=1

help:
	@echo ""
	@echo "TopArb — GPU-Accelerated Topological StatArb"
	@echo "============================================="
	@echo "  make install        Install production deps"
	@echo "  make install-dev    Install all deps incl. dev tools"
	@echo "  make test           Run ALL tests"
	@echo "  make test-unit      Unit tests only (fast, no network)"
	@echo "  make test-safety    Safety-critical tests only"
	@echo "  make test-cov       Tests with HTML coverage report"
	@echo "  make lint           Ruff linter"
	@echo "  make format         Black + isort auto-format"
	@echo "  make validate       Validate configuration"
	@echo "  make backtest       Run historical backtest"
	@echo "  make calibrate      Parameter grid search"
	@echo "  make paper          Start paper trading"
	@echo "  make clean          Remove build artifacts"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	$(ENV) $(PYTEST) tests/ -v --tb=short

test-unit:
	$(ENV) $(PYTEST) tests/ -v --tb=short \
		--ignore=tests/test_integration.py \
		--ignore=tests/test_live_safety.py \
		-m "not integration"

test-safety:
	@echo "=== SAFETY-CRITICAL TESTS ==="
	$(ENV) $(PYTEST) tests/test_risk_manager.py tests/test_live_safety.py \
		-v --tb=long -x
	@echo "Safety tests complete."

test-cov:
	$(ENV) $(PYTEST) tests/ \
		--cov=src --cov=config \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=75 -q

lint:
	$(PYTHON) -m ruff check src/ tests/ config/ --fix

format:
	$(PYTHON) -m black src/ tests/ config/ main.py calibrate.py
	$(PYTHON) -m isort src/ tests/ config/ main.py calibrate.py

validate:
	$(ENV) $(PYTHON) main.py --mode validate

backtest:
	$(ENV) $(PYTHON) main.py --mode backtest \
		--start $(or $(START),2020-01-01) \
		--end   $(or $(END),2023-12-31) \
		--train-window 60

calibrate:
	$(ENV) $(PYTHON) calibrate.py --tickers 20 --years 2 --folds 3

paper:
	$(ENV) $(PYTHON) main.py --mode live

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache .ruff_cache 2>/dev/null || true
	@echo "Clean."
