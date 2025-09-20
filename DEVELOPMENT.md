# MemoirAI Development Guide

## Project Structure

```
memoir-ai/
├── memoir_ai/              # Main package
│   ├── __init__.py         # Package exports
│   ├── core.py             # Main MemoirAI class
│   ├── models.py           # Data models and types
│   ├── exceptions.py       # Custom exceptions
│   └── config.py           # Configuration management
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_config.py      # Configuration tests
├── examples/               # Usage examples
│   └── basic_usage.py      # Basic usage demonstration
├── pyproject.toml          # Project metadata
├── README.md               # Project documentation
├── Makefile                # Development commands
└── .pre-commit-config.yaml # Code quality hooks
```

## Completed Features ✅

### Task 1: Project Structure and Core Dependencies

- ✅ Python package structure with uv-based dependency workflow
- ✅ Core dependencies: SQLAlchemy, Pydantic AI, liteLLM, asyncio support
- ✅ Development dependencies: pytest, black, mypy, pre-commit hooks
- ✅ Configuration management for database and LLM settings
- ✅ Comprehensive configuration validation with clear error messages
- ✅ Environment variable support for all major settings
- ✅ Basic test suite with 9 passing tests

### Task 2.1: SQLAlchemy Database Models

- ✅ Category model with hierarchy constraints and validation
- ✅ Chunk model with token counting and source tracking
- ✅ ContextualHelper model for source context storage
- ✅ CategoryLimits model for configurable category limits per level
- ✅ All required indexes and constraints as specified in design
- ✅ Comprehensive model validation with clear error messages
- ✅ Database manager with connection pooling and retry logic
- ✅ Multi-database support (SQLite, PostgreSQL, MySQL configuration)
- ✅ Transaction management and error recovery
- ✅ 28 passing tests covering all database functionality

### Task 2.2: Database Migration and Initialization System

- ✅ Alembic migration setup for schema versioning
- ✅ Database initialization logic that auto-creates tables and indexes
- ✅ Database connection management with retry logic and error handling
- ✅ Migration tracking and validation system
- ✅ Database reset and schema validation functionality
- ✅ Integration with MemoirAI core class for automatic initialization
- ✅ Comprehensive migration manager with error recovery
- ✅ 37+ passing tests covering all migration functionality

### Task 3.1: Token-Based Text Chunker

- ✅ TextChunker class using liteLLM's token_counter function
- ✅ Configurable delimiters and token size thresholds (300-500 tokens default)
- ✅ Paragraph boundary preservation logic with configurable options
- ✅ Chunk merging for undersized chunks and splitting for oversized chunks
- ✅ Comprehensive validation and error handling with clear messages
- ✅ Multi-model token counting support (GPT, Claude, etc.)
- ✅ Unicode and special character support
- ✅ Integration with MemoirAI core class for seamless usage
- ✅ 25+ passing tests covering all chunking functionality and edge cases

### Configuration Validation Features

- ✅ Token budget vs chunk size constraint validation
- ✅ Hierarchy depth validation (1-100 levels)
- ✅ Batch size validation (1-50)
- ✅ Category limits validation (global and per-level)
- ✅ Database URL format validation
- ✅ LLM provider and model validation
- ✅ Clear error messages with suggested fixes

### Code Quality Setup

- ✅ Black code formatting
- ✅ MyPy type checking
- ✅ Pre-commit hooks configuration
- ✅ Comprehensive test coverage
- ✅ Development Makefile with common tasks

## Next Steps 🚀

The foundation is now solid and ready for implementing core features:

1. **Database Layer** (Task 2.1-2.2)

   - SQLAlchemy models for categories, chunks, contextual helpers
   - Database migration and initialization system
   - Multi-database backend support

2. **Text Processing** (Task 3.1-3.2)

   - Token-based text chunker using liteLLM
   - Contextual helper generation system
   - Paragraph boundary preservation

3. **LLM Integration** (Task 4.1-4.2)
   - Pydantic AI schemas and agents
   - Batch classification system
   - Native structured output support

## Development Commands

```bash
# Create virtual environment (one-time)
uv venv .venv

# Install dev dependencies and pre-commit hooks
make dev-setup

# Run tests
make test

# Run example script
UV_PROJECT_ENVIRONMENT=.venv UV_NO_PROJECT=1 uv run python examples/basic_usage.py

# Format code
make format

# Run linting
make lint
```

## Configuration Examples

```python
# Basic configuration
memoir = MemoirAI(
    database_url="sqlite:///memoir.db",
    llm_provider="openai",
    model_name="gpt-4"
)

# Advanced configuration
memoir = MemoirAI(
    database_url="postgresql://user:pass@localhost/memoir",
    hierarchy_depth=5,
    chunk_min_tokens=200,
    chunk_max_tokens=800,
    batch_size=10,
    max_categories_per_level={1: 20, 2: 50, 3: 100},
    max_token_budget=8000
)
```

## Environment Variables

```bash
export MEMOIR_DATABASE_URL="sqlite:///memoir.db"
export MEMOIR_LLM_PROVIDER="openai"
export MEMOIR_MODEL_NAME="gpt-4"
export MEMOIR_HIERARCHY_DEPTH="3"
export MEMOIR_BATCH_SIZE="5"
export MEMOIR_MAX_TOKEN_BUDGET="4000"
```

The project foundation is complete and ready for core feature implementation! 🎉
