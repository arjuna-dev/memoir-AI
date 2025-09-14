"""
Query module for MemoirAI.

This module provides query processing capabilities including strategy-based
category traversal and content retrieval.
"""

from .query_strategy_engine import (
    QueryStrategyEngine,
    QueryStrategy,
    CategoryPath,
    QueryClassificationResult,
    LLMCallResponse,
    QueryExecutionResult,
    create_query_strategy_engine,
    validate_strategy_params,
)
from .chunk_retrieval import (
    ChunkRetriever,
    ResultConstructor,
    ChunkResult,
    PathRetrievalResult,
    QueryResult,
    create_chunk_retriever,
    create_result_constructor,
)
from .query_processor import (
    QueryProcessor,
    create_query_processor,
    process_natural_language_query,
)

__all__ = [
    # Query strategy components
    "QueryStrategyEngine",
    "QueryStrategy",
    "CategoryPath",
    "QueryClassificationResult",
    "LLMCallResponse",
    "QueryExecutionResult",
    "create_query_strategy_engine",
    "validate_strategy_params",
    # Chunk retrieval components
    "ChunkRetriever",
    "ResultConstructor",
    "ChunkResult",
    "PathRetrievalResult",
    "QueryResult",
    "create_chunk_retriever",
    "create_result_constructor",
    # Query processing integration
    "QueryProcessor",
    "create_query_processor",
    "process_natural_language_query",
]
