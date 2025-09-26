"""
Tests for query strategy engine.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from memoir_ai.database.models import Category
from memoir_ai.exceptions import ClassificationError, ValidationError
from memoir_ai.llm.schemas import QueryCategorySelection
from memoir_ai.query.query_strategy_engine import (
    CategoryPath,
    LLMCallResponse,
    QueryClassificationResult,
    QueryExecutionResult,
    QueryStrategy,
    QueryStrategyEngine,
    create_query_strategy_engine,
    validate_strategy_params,
)


class TestCategoryPath:
    """Test CategoryPath functionality."""

    def test_category_path_creation(self) -> None:
        """Test CategoryPath creation and properties."""
        categories = [
            Category(id=1, name="Technology", level=1),
            Category(id=2, name="AI", level=2, parent_id=1),
            Category(id=3, name="ML", level=3, parent_id=2),
        ]

        path = CategoryPath(path=categories, ranked_relevance=5)

        assert path.path == categories
        assert path.ranked_relevance == 5
        assert path.leaf_category == categories[2]
        assert path.depth == 3
        assert path.path_string == "Technology > AI > ML"

    def test_category_path_empty(self) -> None:
        """Test CategoryPath with empty path."""
        path = CategoryPath(path=[], ranked_relevance=1)

        assert path.path == []
        assert path.leaf_category is None
        assert path.depth == 0
        assert path.path_string == ""

    def test_category_path_equality(self) -> None:
        """Test CategoryPath equality and hashing."""
        categories1 = [
            Category(id=1, name="Technology", level=1),
            Category(id=2, name="AI", level=2, parent_id=1),
        ]
        categories2 = [
            Category(id=1, name="Technology", level=1),
            Category(id=2, name="AI", level=2, parent_id=1),
        ]
        categories3 = [
            Category(id=1, name="Technology", level=1),
            Category(id=3, name="ML", level=2, parent_id=1),
        ]

        path1 = CategoryPath(path=categories1, ranked_relevance=5)
        path2 = CategoryPath(
            path=categories2, ranked_relevance=3
        )  # Different relevance
        path3 = CategoryPath(path=categories3, ranked_relevance=5)

        # Equality based on category IDs, not relevance
        assert path1 == path2
        assert path1 != path3

        # Hashing for deduplication
        assert hash(path1) == hash(path2)
        assert hash(path1) != hash(path3)


class TestQueryClassificationResult:
    """Test QueryClassificationResult Pydantic model."""

    def test_query_classification_result_creation(self) -> None:
        """Test QueryClassificationResult creation."""
        result = QueryClassificationResult(category="Technology", ranked_relevance=5)

        assert result.category == "Technology"
        assert result.ranked_relevance == 5


class TestQueryStrategyEngine:
    """Test QueryStrategyEngine functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock category manager
        self.mock_category_manager = Mock()
        self.mock_category_manager.hierarchy_depth = 3
        self.mock_category_manager.db_session = Mock()

        # Mock categories
        self.level1_categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="Science", level=1, parent_id=None),
        ]

        self.level2_tech_categories = [
            Category(id=3, name="AI", level=2, parent_id=1),
            Category(id=4, name="Web", level=2, parent_id=1),
        ]

        self.level3_ai_categories = [
            Category(id=5, name="ML", level=3, parent_id=3),
            Category(id=6, name="NLP", level=3, parent_id=3),
        ]

    def test_initialization(self) -> None:
        """Test QueryStrategyEngine initialization."""
        with patch(
            "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
        ):
            engine = QueryStrategyEngine(
                category_manager=self.mock_category_manager,
                model_name="openai:gpt-4o-mini",
            )

            assert engine.category_manager == self.mock_category_manager
            assert engine.model_name == "openai:gpt-4o-mini"
            assert engine.session == self.mock_category_manager.db_session

    @pytest.mark.asyncio
    async def test_execute_strategy_invalid(self) -> None:
        """Test execute_strategy with invalid strategy."""
        with patch(
            "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
        ):
            engine = QueryStrategyEngine(category_manager=self.mock_category_manager)

            with pytest.raises(ValidationError) as exc_info:
                await engine.execute_strategy(
                    query_text="test query", strategy="invalid_strategy"
                )
            assert "Unknown strategy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_one_shot_strategy(self) -> None:
        """Test one-shot strategy execution."""
        with (
            patch(
                "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
            ) as mock_agent_factory,
            patch(
                "memoir_ai.query.query_strategy_engine.select_category_for_query",
                new_callable=AsyncMock,
            ) as mock_select,
        ):
            mock_agent_factory.return_value = Mock()

            mock_select.side_effect = [
                (
                    QueryCategorySelection(category="Technology", ranked_relevance=5),
                    {
                        "latency_ms": 42,
                        "timestamp": datetime.now(),
                        "prompt": "",
                    },
                ),
                (
                    QueryCategorySelection(category="AI", ranked_relevance=5),
                    {
                        "latency_ms": 41,
                        "timestamp": datetime.now(),
                        "prompt": "",
                    },
                ),
                (
                    QueryCategorySelection(category="ML", ranked_relevance=5),
                    {
                        "latency_ms": 40,
                        "timestamp": datetime.now(),
                        "prompt": "",
                    },
                ),
            ]

            # Mock category manager responses
            self.mock_category_manager.get_existing_categories.side_effect = [
                self.level1_categories,  # Level 1
                self.level2_tech_categories,  # Level 2 under Technology
                self.level3_ai_categories,  # Level 3 under AI
                [],  # Level 4 (no more categories)
            ]

            engine = QueryStrategyEngine(category_manager=self.mock_category_manager)

            result = await engine.execute_strategy(
                query_text="machine learning algorithms",
                strategy=QueryStrategy.ONE_SHOT,
            )

            assert isinstance(result, QueryExecutionResult)
            assert result.strategy_used == QueryStrategy.ONE_SHOT
            assert len(result.category_paths) > 0
            assert len(result.llm_responses) > 0

            # Check that we got a complete path
            path = result.category_paths[0]
            assert len(path.path) == 3  # Technology > AI > ML
            assert path.path[0].name == "Technology"
            assert path.path[1].name == "AI"
            assert path.path[2].name == "ML"

    def test_deduplicate_paths(self) -> None:
        """Test path deduplication."""
        with patch(
            "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
        ):
            engine = QueryStrategyEngine(category_manager=self.mock_category_manager)

            # Create duplicate paths
            categories1 = [
                Category(id=1, name="Technology", level=1),
                Category(id=2, name="AI", level=2, parent_id=1),
            ]
            categories2 = [
                Category(id=1, name="Technology", level=1),
                Category(id=2, name="AI", level=2, parent_id=1),
            ]
            categories3 = [
                Category(id=1, name="Technology", level=1),
                Category(id=3, name="Web", level=2, parent_id=1),
            ]

            paths = [
                CategoryPath(path=categories1, ranked_relevance=5),
                CategoryPath(path=categories2, ranked_relevance=3),  # Duplicate
                CategoryPath(path=categories3, ranked_relevance=4),
            ]

            unique_paths = engine._deduplicate_paths(paths)

            assert len(unique_paths) == 2
            # Should keep first occurrence of duplicate
            assert unique_paths[0].ranked_relevance == 5
            assert unique_paths[1].path[1].name == "Web"

    def test_validate_paths(self) -> None:
        """Test path validation."""
        with patch(
            "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
        ):
            engine = QueryStrategyEngine(category_manager=self.mock_category_manager)

            categories = [
                Category(id=1, name="Technology", level=1),
                Category(id=2, name="AI", level=2, parent_id=1),
            ]

            paths = [
                CategoryPath(path=categories, ranked_relevance=5),
                CategoryPath(path=[], ranked_relevance=1),  # Empty path
            ]

            valid_paths = engine._validate_paths(paths)

            # Should filter out empty paths but keep valid ones
            assert len(valid_paths) == 1
            assert valid_paths[0].path == categories

    def test_get_strategy_info(self) -> None:
        """Test getting strategy information."""
        with patch(
            "memoir_ai.query.query_strategy_engine.create_query_classification_agent"
        ):
            engine = QueryStrategyEngine(category_manager=self.mock_category_manager)

            # Test known strategy
            info = engine.get_strategy_info(QueryStrategy.ONE_SHOT)
            assert info["name"] == "One Shot"
            assert "single best category" in info["description"]

            # Test wide branch strategy
            info = engine.get_strategy_info(QueryStrategy.WIDE_BRANCH)
            assert info["name"] == "Wide Branch"
            assert len(info["parameters"]) > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_query_strategy_engine(self) -> None:
        """Test create_query_strategy_engine function."""
        # For now, just test that the function exists and can be called
        # In a real implementation, this would test the actual creation logic
        engine_func = create_query_strategy_engine
        assert callable(engine_func)

    def test_validate_strategy_params_one_shot(self) -> None:
        """Test parameter validation for one-shot strategy."""
        params = validate_strategy_params(QueryStrategy.ONE_SHOT, {})
        assert params == {}

        # Extra params should be ignored
        params = validate_strategy_params(QueryStrategy.ONE_SHOT, {"n": 5})
        assert params == {}

    def test_validate_strategy_params_wide_branch(self) -> None:
        """Test parameter validation for wide branch strategy."""
        # Valid params
        params = validate_strategy_params(QueryStrategy.WIDE_BRANCH, {"n": 3})
        assert params == {"n": 3}

        # Default value
        params = validate_strategy_params(QueryStrategy.WIDE_BRANCH, {})
        assert params == {"n": 3}

        # Invalid params
        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_params(QueryStrategy.WIDE_BRANCH, {"n": 0})
        assert "must be a positive integer" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_params(QueryStrategy.WIDE_BRANCH, {"n": "invalid"})
        assert "must be a positive integer" in str(exc_info.value)

    def test_validate_strategy_params_zoom_in(self) -> None:
        """Test parameter validation for zoom in strategy."""
        # Valid params
        params = validate_strategy_params(QueryStrategy.ZOOM_IN, {"n": 5, "n2": 2})
        assert params == {"n": 5, "n2": 2}

        # Default values
        params = validate_strategy_params(QueryStrategy.ZOOM_IN, {})
        assert params == {"n": 3, "n2": 1}

        # Invalid n
        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_params(QueryStrategy.ZOOM_IN, {"n": -1})
        assert "must be a positive integer" in str(exc_info.value)

        # Invalid n2
        with pytest.raises(ValidationError) as exc_info:
            validate_strategy_params(QueryStrategy.ZOOM_IN, {"n": 3, "n2": 0})
        assert "must be a positive integer" in str(exc_info.value)

    def test_validate_strategy_params_branch_out(self) -> None:
        """Test parameter validation for branch out strategy."""
        # Valid params
        params = validate_strategy_params(QueryStrategy.BRANCH_OUT, {"n": 2, "n2": 3})
        assert params == {"n": 2, "n2": 3}

        # Default values
        params = validate_strategy_params(QueryStrategy.BRANCH_OUT, {})
        assert params == {"n": 1, "n2": 1}

        # Invalid params should raise ValidationError
        with pytest.raises(ValidationError):
            validate_strategy_params(QueryStrategy.BRANCH_OUT, {"n": "invalid"})
