"""
MemoirAI - LLM-powered hierarchical text storage and retrieval library.
"""

try:  # Core may import optional modules not yet implemented (e.g., iterative classifier)
    from .core import MemoirAI  # type: ignore
except Exception:  # pragma: no cover
    MemoirAI = None  # type: ignore
from .database import Category, CategoryLimits, Chunk, ContextualHelper, DatabaseManager
from .exceptions import (
    ClassificationError,
    ConfigurationError,
    DatabaseError,
    MemoirAIError,
    ValidationError,
)
from .models import (
    CategoryTree,
    IngestionResult,
    PromptLimitingStrategy,
    QueryResult,
    QueryStrategy,
)
from .text_processing import TextChunk, TextChunker

# Query imports temporarily disabled to avoid circular imports
# from .query import (
#     QueryStrategyEngine,
#     CategoryPath,
#     QueryClassificationResult,
#     LLMCallResponse,
#     QueryExecutionResult,
#     create_query_strategy_engine,
#     validate_strategy_params,
# )

__version__ = "0.1.0"
__all__ = [
    "MemoirAI",
    "IngestionResult",
    "QueryResult",
    "CategoryTree",
    "QueryStrategy",
    "PromptLimitingStrategy",
    "MemoirAIError",
    "ConfigurationError",
    "ClassificationError",
    "DatabaseError",
    "ValidationError",
    "DatabaseManager",
    "Category",
    "Chunk",
    "ContextualHelper",
    "CategoryLimits",
    "TextChunker",
    "TextChunk",
    # Query exports temporarily disabled
    # "QueryStrategyEngine",
    # "CategoryPath",
    # "QueryClassificationResult",
    # "LLMCallResponse",
    # "QueryExecutionResult",
    # "create_query_strategy_engine",
    # "validate_strategy_params",
]
