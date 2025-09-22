.PHONY: install test lint format clean dev-setup dev

UV ?= UV_PROJECT_ENVIRONMENT=.venv UV_NO_PROJECT=1 uv

# Development setup
dev-setup:
	$(UV) pip install -r requirements-dev.txt
	$(UV) run pre-commit install

# Install dependencies
install:
	$(UV) pip install -r requirements.txt

# Run tests
test:
	$(UV) run pytest -v

# Run linting
lint:
	$(UV) run black --check memoir_ai tests
	$(UV) run mypy memoir_ai
	$(UV) run flake8 memoir_ai tests

# Format code
format:
	$(UV) run black memoir_ai tests
	$(UV) run isort memoir_ai tests

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Run development server (for testing)
dev:
	$(UV) run python -c "from memoir_ai import MemoirAI; print('MemoirAI imported successfully')"
