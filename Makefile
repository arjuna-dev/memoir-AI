.PHONY: install test lint format clean dev-setup

# Development setup
dev-setup:
	poetry install
	poetry run pre-commit install

# Install dependencies
install:
	poetry install --no-dev

# Run tests
test:
	poetry run pytest -v --cov=memoir_ai --cov-report=html --cov-report=term

# Run linting
lint:
	poetry run black --check memoir_ai tests
	poetry run mypy memoir_ai
	poetry run flake8 memoir_ai tests

# Format code
format:
	poetry run black memoir_ai tests
	poetry run isort memoir_ai tests

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
	poetry run python -c "from memoir_ai import MemoirAI; print('MemoirAI imported successfully')"