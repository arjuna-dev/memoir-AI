# MemoirAI

A Python library for LLM-powered hierarchical text storage and retrieval using relational databases.

## Features

- Token-based text chunking with configurable size constraints
- Batch classification using LLM providers (OpenAI, Anthropic, etc.)
- Configurable category hierarchy (1-100 levels, default 3)
- Four query strategies: one shot, wide branch, zoom in, branch out
- Advanced result aggregation with pruning and summarization
- Multi-database support (SQLite, PostgreSQL, MySQL)
- Transaction-based error recovery and configuration validation

## Installation

```bash
pip install memoir-ai
```

## Quick Start

```python
from memoir_ai import MemoirAI

# Initialize with database
memoir = MemoirAI(
    database_url="sqlite:///memoir.db",
    llm_provider="openai",
    model_name="gpt-5-nano"
)

# Ingest text
result = await memoir.ingest_text(
    content="Your large text document...",
    source_id="document_1"
)

# Query content
query_result = await memoir.query(
    "What did the document say about AI?"
)

print(query_result.answer)
```

## Documentation

Coming soon...

## License

MIT License
