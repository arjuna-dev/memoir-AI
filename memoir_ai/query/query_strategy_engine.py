"""
Query strategy engine for MemoirAI.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from ..database.models import Category
from ..exceptions import ValidationError


class QueryStrategy(Enum):
    """Available query traversal strategies."""

    ONE_SHOT = "one_shot"
    WIDE_BRANCH = "wide_branch"
    ZOOM_IN = "zoom_in"
    BRANCH_OUT = "branch_out"


@dataclass
class CategoryPath:
    """Represents a complete path from root to leaf category."""

    path: List[Category]
    ranked_relevance: int

    @property
    def leaf_category(self) -> Category:
        return self.path[-1] if self.path else None

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def path_string(self) -> str:
        return " > ".join(cat.name for cat in self.path)


class QueryClassificationResult(BaseModel):
    """Pydantic model for LLM query classification responses."""

    category: str
    ranked_relevance: int


@dataclass
class LLMCallResponse:
    """Response object for individual LLM calls."""

    llm_output: Optional[QueryClassificationResult]
    timestamp: datetime
    latency_ms: int


@dataclass
class QueryExecutionResult:
    """Result of executing a query strategy."""

    category_paths: List[CategoryPath]
    llm_responses: List[LLMCallResponse]
    total_latency_ms: int
    strategy_used: QueryStrategy
    strategy_params: Dict[str, Any]


class QueryStrategyEngine:
    """Engine for executing different query traversal strategies."""

    def __init__(self, category_manager, model_name: str = "openai:gpt-4"):
        self.category_manager = category_manager
        self.model_name = model_name

    async def execute_strategy(
        self, query_text: str, strategy: QueryStrategy, **kwargs
    ):
        """Execute a query strategy."""
        # Simplified implementation
        return QueryExecutionResult(
            category_paths=[],
            llm_responses=[],
            total_latency_ms=0,
            strategy_used=strategy,
            strategy_params=kwargs,
        )

    def get_strategy_info(self, strategy: QueryStrategy) -> Dict[str, Any]:
        """Get information about a strategy."""
        return {"name": strategy.value, "description": f"Strategy: {strategy.value}"}


def create_query_strategy_engine(**kwargs) -> QueryStrategyEngine:
    """Create a query strategy engine."""
    from ..classification.category_manager import create_category_manager
    from ..database.connection import get_session

    session = get_session()
    category_manager = create_category_manager(db_session=session, hierarchy_depth=3)
    return QueryStrategyEngine(category_manager=category_manager)


def validate_strategy_params(
    strategy: QueryStrategy, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate strategy parameters."""
    if strategy == QueryStrategy.WIDE_BRANCH:
        n = params.get("n", 3)
        if not isinstance(n, int) or n < 1:
            raise ValidationError(
                "Parameter 'n' must be a positive integer", field="n", value=n
            )
        return {"n": n}
    return {}
