"""Query strategy engine for MemoirAI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from pydantic import BaseModel
from pydantic_ai import Agent

from ..database.models import Category
from ..exceptions import ClassificationError, ValidationError
from ..llm.agents import (
    create_query_classification_agent,
    create_query_multi_selection_agent,
)
from ..llm.interactions import select_categories_for_query, select_category_for_query

if TYPE_CHECKING:
    from ..classification.category_manager import CategoryManager

logger = logging.getLogger(__name__)


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
    def leaf_category(self) -> Optional[Category]:
        return self.path[-1] if self.path else None

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def path_string(self) -> str:
        return " > ".join(cat.name for cat in self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoryPath):
            return False
        return [cat.id for cat in self.path] == [cat.id for cat in other.path]

    def __hash__(self) -> int:
        return hash(tuple(cat.id for cat in self.path))


class QueryClassificationResult(BaseModel):
    """Structured LLM output for category selection."""

    category: str
    ranked_relevance: int


@dataclass
class LLMCallResponse:
    """Metadata for an individual LLM call during traversal."""

    llm_output: Optional[QueryClassificationResult]
    timestamp: datetime
    latency_ms: int


@dataclass
class QueryExecutionResult:
    """Result of executing a query traversal strategy."""

    category_paths: List[CategoryPath]
    llm_responses: List[LLMCallResponse]
    total_latency_ms: int
    strategy_used: QueryStrategy
    strategy_params: Dict[str, Any]


class QueryStrategyEngine:
    """Engine for executing category traversal strategies."""

    def __init__(
        self,
        category_manager: "CategoryManager",
        model_name: str = "openai:gpt-5-nano",
    ) -> None:
        self.category_manager = category_manager
        self.model_name = model_name
        self.session = getattr(category_manager, "db_session", None)
        self.hierarchy_depth = getattr(category_manager, "hierarchy_depth", 3)
        self._query_agent: Agent | None = None
        self._query_multi_agent: Agent | None = None

    async def execute_strategy(
        self,
        query_text: str,
        strategy: QueryStrategy,
        contextual_helper: str,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> QueryExecutionResult:
        """Execute the requested traversal strategy."""

        if not isinstance(strategy, QueryStrategy):
            raise ValidationError(
                "Unknown strategy provided",
                field="strategy",
                value=strategy,
            )

        params = validate_strategy_params(strategy, strategy_params or {})

        if strategy == QueryStrategy.ONE_SHOT:
            paths, responses = await self._execute_one_shot(
                query_text=query_text,
                contextual_helper=contextual_helper,
            )
        else:
            paths, responses = await self._execute_branching_strategy(
                query_text=query_text,
                contextual_helper=contextual_helper,
                strategy=strategy,
                params=params,
            )

        deduped_paths = self._validate_paths(self._deduplicate_paths(paths))
        total_latency = sum(response.latency_ms for response in responses)

        return QueryExecutionResult(
            category_paths=deduped_paths,
            llm_responses=responses,
            total_latency_ms=total_latency,
            strategy_used=strategy,
            strategy_params=params,
        )

    async def _execute_one_shot(
        self, query_text: str, contextual_helper: str
    ) -> Tuple[List[CategoryPath], List[LLMCallResponse]]:
        """Traverse the hierarchy selecting a single path."""

        responses: List[LLMCallResponse] = []
        selected_categories: List[Category] = []
        parent: Optional[Category] = None

        # Anchor traversal at the contextual helper (level 1) when available
        try:
            level_one_categories = self.category_manager.get_existing_categories(1)
            ctx_norm = contextual_helper.strip().lower()
            for category in level_one_categories:
                meta = getattr(category, "metadata_json", {}) or {}
                helper_text = str(meta.get("helper_text", "")).strip().lower()
                if category.name.strip().lower() == ctx_norm or (
                    helper_text and helper_text == ctx_norm
                ):
                    parent = category
                    break
        except Exception:
            parent = None

        # We start at level 2 since level 1 is handled separately and reserved for contextual helpers
        start_level = 2

        for level in range(start_level, self.hierarchy_depth + 1):
            categories = self.category_manager.get_existing_categories(
                level, parent.id if parent else None
            )
            if not categories:
                break

            selection = await self._call_selection_agent(
                query_text=query_text,
                contextual_helper=contextual_helper,
                level=level,
                available_categories=categories,
            )

            responses.append(selection["response"])
            if not selection["results"]:
                raise ClassificationError(
                    "LLM did not return any category selection",
                    model=self.model_name,
                )

            chosen_category = self._match_category_by_name(
                categories, selection["results"][0].category
            )
            if not chosen_category:
                raise ClassificationError(
                    f"LLM selected unknown category '{selection['results'][0].category}'",
                    model=self.model_name,
                )

            selected_categories.append(chosen_category)
            parent = chosen_category

        if not selected_categories:
            return [], responses

        final_response = responses[-1] if responses else None
        final_rank = (
            final_response.llm_output.ranked_relevance
            if final_response and final_response.llm_output
            else 0
        )
        path = CategoryPath(path=selected_categories, ranked_relevance=final_rank)
        return [path], responses

    async def _execute_branching_strategy(
        self,
        query_text: str,
        contextual_helper: str,
        strategy: QueryStrategy,
        params: Dict[str, Any],
    ) -> Tuple[List[CategoryPath], List[LLMCallResponse]]:
        """Execute a branching strategy using simple deterministic expansion."""

        # Strategy parameters
        n = cast(int, params.get("n", 1))
        n2 = cast(int, params.get("n2", 1))

        # Determine how many categories to select per level
        def selections_for_level(level_index: int) -> int:
            if strategy == QueryStrategy.WIDE_BRANCH:
                return n
            if strategy == QueryStrategy.ZOOM_IN:
                return max(1, n - level_index * n2)
            if strategy == QueryStrategy.BRANCH_OUT:
                return n + level_index * n2
            return 1

        responses: List[LLMCallResponse] = []
        path_records: List[
            Tuple[List[Category], Optional[QueryClassificationResult]]
        ] = [([], None)]

        # Find the contextual helper (level 1) to use as parent for level 2
        contextual_helper_category: Optional[Category] = None
        if contextual_helper:
            level_one_categories = self.category_manager.get_existing_categories(
                level=1
            )
            for category in level_one_categories:
                if category.name.strip().lower() == contextual_helper.strip().lower():
                    contextual_helper_category = category
                    break

        # We start at level 2 since level 1 is handled separately and reserved for contextual helpers
        start_level = 2

        for level in range(start_level, self.hierarchy_depth + 1):
            new_records: List[
                Tuple[List[Category], Optional[QueryClassificationResult]]
            ] = []

            for path, _ in path_records:
                # Get last category in the current path to use as parent
                if path:
                    parent_category = path[-1]
                elif level == 2 and contextual_helper_category:
                    # For level 2, use the contextual helper as parent
                    parent_category = contextual_helper_category
                else:
                    parent_category = None

                categories = self.category_manager.get_existing_categories(
                    level, parent_category.id if parent_category else None
                )
                # If no categories are found, raise an error
                if not categories:
                    continue
                # Get how many categories to select at this level
                top_k = selections_for_level(level - 1)
                # Ensure top_k is not greater than the number of available categories
                top_k = min(len(categories), top_k)

                selection = await self._call_selection_agent(
                    query_text=query_text,
                    contextual_helper=contextual_helper,
                    level=level,
                    available_categories=categories,
                    top_k=top_k,
                )
                responses.append(selection["response"])

                for result in selection["results"]:
                    chosen_category = self._match_category_by_name(
                        categories, result.category
                    )
                    if not chosen_category:
                        logger.warning(
                            "LLM selected unknown category '%s' at level %s",
                            result.category,
                            level,
                        )
                        continue

                    new_records.append((path + [chosen_category], result))

            if not new_records:
                break
            path_records = new_records

        category_paths: List[CategoryPath] = []
        for path, result in path_records:
            if not path:
                continue
            rank = result.ranked_relevance if result else 0
            category_paths.append(CategoryPath(path=path, ranked_relevance=rank))

        return category_paths, responses

    async def _call_selection_agent(
        self,
        query_text: str,
        contextual_helper: str,
        level: int,
        available_categories: Sequence[Category],
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Call the LLM agent to select one or more categories."""

        top_k = max(1, top_k)

        if top_k == 1:
            if self._query_agent is None:
                self._query_agent = create_query_classification_agent(self.model_name)

            selection, metadata = await select_category_for_query(
                query_text=query_text,
                level=level,
                available_categories=available_categories,
                contextual_helper=contextual_helper,
                agent=self._query_agent,
                model_name=self.model_name,
            )

            result = QueryClassificationResult(
                category=selection.category, ranked_relevance=selection.ranked_relevance
            )

            return {
                "results": [result],
                "response": LLMCallResponse(
                    llm_output=result,
                    timestamp=metadata.get("timestamp", datetime.now()),
                    latency_ms=metadata.get("latency_ms", 0),
                ),
            }

        if self._query_multi_agent is None:
            self._query_multi_agent = create_query_multi_selection_agent(
                self.model_name
            )

        selections, metadata = await select_categories_for_query(
            query_text=query_text,
            level=level,
            available_categories=available_categories,
            contextual_helper=contextual_helper or "",
            selection_count=top_k,
            agent=self._query_multi_agent,
            model_name=self.model_name,
        )

        results = [
            QueryClassificationResult(
                category=selection.category, ranked_relevance=selection.ranked_relevance
            )
            for selection in selections
        ]

        primary_result = results[0] if results else None

        return {
            "results": results,
            "response": LLMCallResponse(
                llm_output=primary_result,
                timestamp=metadata.get("timestamp", datetime.now()),
                latency_ms=metadata.get("latency_ms", 0),
            ),
        }

    def _match_category_by_name(
        self, categories: Iterable[Category], name: str
    ) -> Optional[Category]:
        for category in categories:
            if category.name.lower() == name.lower():
                return category
        return None

    def _deduplicate_paths(self, paths: List[CategoryPath]) -> List[CategoryPath]:
        """Remove duplicate paths preserving first occurrence."""

        seen: set[Tuple[int, ...]] = set()
        unique_paths: List[CategoryPath] = []
        for path in paths:
            key = tuple(cat.id for cat in path.path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)
        return unique_paths

    def _validate_paths(self, paths: List[CategoryPath]) -> List[CategoryPath]:
        """Filter out empty paths."""

        return [path for path in paths if path.path]

    def get_strategy_info(self, strategy: QueryStrategy) -> Dict[str, Any]:
        """Provide human-friendly information about a strategy."""

        descriptions = {
            QueryStrategy.ONE_SHOT: (
                "One Shot",
                "Select a single best category at each level until reaching a leaf.",
            ),
            QueryStrategy.WIDE_BRANCH: (
                "Wide Branch",
                "Explore the top-N categories at every level for broader coverage.",
            ),
            QueryStrategy.ZOOM_IN: (
                "Zoom In",
                "Start wide then narrow selections deeper in the hierarchy.",
            ),
            QueryStrategy.BRANCH_OUT: (
                "Branch Out",
                "Begin focused and expand selections as depth increases.",
            ),
        }

        name, description = descriptions[strategy]

        parameters = {
            QueryStrategy.ONE_SHOT: {},
            QueryStrategy.WIDE_BRANCH: {"n": "Number of categories per level"},
            QueryStrategy.ZOOM_IN: {
                "n": "Initial categories",
                "n2": "Reduction per level",
            },
            QueryStrategy.BRANCH_OUT: {
                "n": "Initial categories",
                "n2": "Expansion per level",
            },
        }

        return {
            "name": name,
            "description": description,
            "parameters": parameters[strategy],
        }


def create_query_strategy_engine(**kwargs: Any) -> QueryStrategyEngine:
    """Factory helper for query strategy engine."""

    from ..classification.category_manager import create_category_manager
    from ..database.connection import get_session

    session = get_session()
    category_manager = create_category_manager(db_session=session, hierarchy_depth=3)
    return QueryStrategyEngine(category_manager=category_manager, **kwargs)


def validate_strategy_params(
    strategy: QueryStrategy, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate and normalise strategy parameters."""

    def _positive_int(value: Any, field: str, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(
                f"Parameter '{field}' must be a positive integer",
                field=field,
                value=value,
            )
        return value

    if strategy == QueryStrategy.ONE_SHOT:
        return {}

    if strategy == QueryStrategy.WIDE_BRANCH:
        n = _positive_int(params.get("n"), "n", 3)
        return {"n": n}

    if strategy == QueryStrategy.ZOOM_IN:
        n = _positive_int(params.get("n"), "n", 3)
        n2 = _positive_int(params.get("n2"), "n2", 1)
        if n2 >= n:
            raise ValidationError(
                "Parameter 'n2' must be less than 'n' for zoom_in strategy",
                field="n2",
                value=n2,
            )
        return {"n": n, "n2": n2}

    if strategy == QueryStrategy.BRANCH_OUT:
        n = _positive_int(params.get("n"), "n", 1)
        n2 = _positive_int(params.get("n2"), "n2", 1)
        return {"n": n, "n2": n2}

    raise ValidationError("Unknown strategy provided", field="strategy", value=strategy)
