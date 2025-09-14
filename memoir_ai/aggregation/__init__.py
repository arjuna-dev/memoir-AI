"""
Result aggregation and token budget management for MemoirAI.

This module provides token budget estimation, pruning, and summarization
capabilities for managing LLM prompt limits.
"""

from .result_aggregator import (
    ResultAggregator,
    TokenEstimate,
    AggregationResult,
    create_result_aggregator,
)
from .pruning_engine import (
    PruningEngine,
    PruningResult,
    PruningStrategy,
    create_pruning_engine,
)
from .budget_manager import (
    BudgetManager,
    BudgetConfig,
    BudgetValidationResult,
    create_budget_manager,
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
