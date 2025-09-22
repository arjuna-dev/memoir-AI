"""
Result aggregation and token budget management for MemoirAI.

This module provides token budget estimation, pruning, and summarization
capabilities for managing LLM prompt limits.
"""

from .budget_manager import (
    BudgetConfig,
    BudgetManager,
    BudgetValidationResult,
    create_budget_manager,
)
from .pruning_engine import (
    PruningEngine,
    PruningResult,
    PruningStrategy,
    create_pruning_engine,
)
from .result_aggregator import (
    AggregationResult,
    ResultAggregator,
    TokenEstimate,
    create_result_aggregator,
)

__all__ = [
    # Result aggregation
    "ResultAggregator",
    "TokenEstimate",
    "AggregationResult",
    "create_result_aggregator",
    # Pruning system
    "PruningEngine",
    "PruningResult",
    "PruningStrategy",
    "create_pruning_engine",
    # Budget management
    "BudgetManager",
    "BudgetConfig",
    "BudgetValidationResult",
    "create_budget_manager",
]
