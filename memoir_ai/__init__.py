"""
MemoirAI - LLM-powered hierarchical text storage and retrieval library.
"""

from .core import MemoirAI
from .models import (
    IngestionResult,
    QueryResult,
    CategoryTree,
    QueryStrategy,
    PromptLimitingStrategy,
)
from .exceptions import (
    MemoirAIError,
    ConfigurationError,
    ClassificationError,
    DatabaseError,
    ValidationError,
)
from .database import (
    DatabaseManager,
    Category,
    Chunk,
    ContextualHelper,
    CategoryLimits,
)
from .text_processing import (
    TextChunker,
    TextChunk,
)

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
