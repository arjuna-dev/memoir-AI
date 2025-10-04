"""
Tests for query processor integration.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from memoir_ai.database.models import Category
from memoir_ai.exceptions import ValidationError
from memoir_ai.query.chunk_retrieval import (
    ChunkResult,
    PathRetrievalResult,
    QueryResult,
)
from memoir_ai.query.query_processor import (
    QueryProcessor,
    create_query_processor,
    process_natural_language_query,
)
from memoir_ai.query.query_strategy_engine import (
    CategoryPath,
    LLMCallResponse,
    QueryClassificationResult,
    QueryExecutionResult,
    QueryStrategy,
)


class TestQueryProcessor:
    """Test QueryProcessor functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_category_manager = Mock()
        self.mock_category_manager.hierarchy_depth = 3

        self.mock_session = Mock()

        self.processor = QueryProcessor(
            category_manager=self.mock_category_manager,
            session=self.mock_session,
            model_name="openai:gpt-4o-mini",
            default_chunk_limit=100,
        )

        # Mock contextual helpers (level 1) and actual categories (level 2+)
        self.contextual_helpers = [
            Category(id=1, name="Tech Article", level=1, parent_id=None),
        ]

        # Actual category hierarchy starts at level 2
        self.categories = [
            Category(id=2, name="Technology", level=2, parent_id=None),
            Category(id=3, name="AI", level=3, parent_id=2),
            Category(id=4, name="ML", level=4, parent_id=3),
        ]

        self.category_path = CategoryPath(path=self.categories, ranked_relevance=5)

    def test_initialization(self) -> None:
        """Test QueryProcessor initialization."""
        assert self.processor.category_manager == self.mock_category_manager
        assert self.processor.session == self.mock_session
        assert self.processor.model_name == "openai:gpt-4o-mini"
        assert self.processor.default_chunk_limit == 100

        # Check that components are initialized
        assert self.processor.strategy_engine is not None
        assert self.processor.chunk_retriever is not None
        assert self.processor.result_constructor is not None

    @pytest.mark.asyncio
    async def test_process_query_success(self) -> None:
        """Test successful query processing."""
        # Mock strategy execution result
        strategy_result = QueryExecutionResult(
            category_paths=[self.category_path],
            llm_responses=[
                LLMCallResponse(
                    llm_output=QueryClassificationResult(
                        category="ML", ranked_relevance=5
                    ),
                    timestamp=datetime.now(),
                    latency_ms=100,
                )
            ],
            total_latency_ms=100,
            strategy_used=QueryStrategy.ONE_SHOT,
            strategy_params={},
        )

        # Mock chunk retrieval result
        chunk_result = ChunkResult(
            chunk_id=1,
            text_content="Machine learning content",
            category_path="Technology > AI > ML",
            category_id_path="2/3/4",  # Updated to reflect level 2+ IDs
            ranked_relevance=5,
            created_at=datetime.now(),
        )

        path_result = PathRetrievalResult(
            category_path=self.category_path,
            chunks=[chunk_result],
            chunk_count=1,
            retrieval_latency_ms=50,
            success=True,
        )

        # Mock the components
        with (
            patch.object(
                self.processor.strategy_engine,
                "execute_strategy",
                new_callable=AsyncMock,
            ) as mock_strategy,
            patch.object(
                self.processor.chunk_retriever, "retrieve_chunks_for_paths"
            ) as mock_retriever,
            patch.object(
                self.processor.result_constructor, "construct_query_result"
            ) as mock_constructor,
            patch.object(
                self.processor.result_constructor, "validate_query_result"
            ) as mock_validator,
        ):
            mock_strategy.return_value = strategy_result
            mock_retriever.return_value = [path_result]

            expected_result = QueryResult(
                chunks=[chunk_result],
                responses=strategy_result.llm_responses,
                total_latency_ms=150,
                total_chunks=1,
                successful_paths=1,
                failed_paths=0,
                query_text="machine learning algorithms",
                strategy_used="one_shot",
            )

            mock_constructor.return_value = expected_result
            mock_validator.return_value = []  # No validation errors

            # Mock category manager to return contextual helpers at level 1
            self.mock_category_manager.get_existing_categories.return_value = (
                self.contextual_helpers
            )

            # Execute query
            result = await self.processor.process_query(
                query_text="machine learning algorithms",
                strategy=QueryStrategy.ONE_SHOT,
            )

            # Verify calls
            mock_strategy.assert_called_once_with(
                query_text="machine learning algorithms",
                strategy=QueryStrategy.ONE_SHOT,
                contextual_helper="Tech Article",  # From level-1 contextual helper
                strategy_params={},
            )

            mock_retriever.assert_called_once_with(
                category_paths=[self.category_path], limit_per_path=100, offset=0
            )

            mock_constructor.assert_called_once()
            mock_validator.assert_called_once()

            # Verify result
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_process_query_with_parameters(self) -> None:
        """Test query processing with custom parameters."""
        with (
            patch.object(
                self.processor.strategy_engine,
                "execute_strategy",
                new_callable=AsyncMock,
            ) as mock_strategy,
            patch.object(
                self.processor.chunk_retriever, "retrieve_chunks_for_paths"
            ) as mock_retriever,
            patch.object(
                self.processor.result_constructor, "construct_query_result"
            ) as mock_constructor,
            patch.object(
                self.processor.result_constructor, "validate_query_result"
            ) as mock_validator,
        ):
            # Setup mocks
            mock_strategy.return_value = QueryExecutionResult(
                category_paths=[],
                llm_responses=[],
                total_latency_ms=0,
                strategy_used=QueryStrategy.WIDE_BRANCH,
                strategy_params={"n": 3},
            )
            mock_retriever.return_value = []
            mock_constructor.return_value = QueryResult(
                chunks=[],
                responses=[],
                total_latency_ms=0,
                total_chunks=0,
                successful_paths=0,
                failed_paths=0,
            )
            mock_validator.return_value = []

            # Mock category manager to return contextual helpers at level 1
            self.mock_category_manager.get_existing_categories.return_value = (
                self.contextual_helpers
            )

            # Execute with custom parameters
            await self.processor.process_query(
                query_text="test query",
                strategy=QueryStrategy.WIDE_BRANCH,
                strategy_params={"n": 3},
                chunk_limit_per_path=50,
                offset=10,
            )

            # Verify strategy was called with correct parameters
            mock_strategy.assert_called_once_with(
                query_text="test query",
                strategy=QueryStrategy.WIDE_BRANCH,
                contextual_helper="Tech Article",  # From level-1 contextual helper
                strategy_params={"n": 3},
            )

            # Verify retriever was called with correct parameters
            mock_retriever.assert_called_once_with(
                category_paths=[], limit_per_path=50, offset=10
            )

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self) -> None:
        """Test error handling in query processing."""
        # Mock strategy engine to raise an error
        with patch.object(
            self.processor.strategy_engine, "execute_strategy", new_callable=AsyncMock
        ) as mock_strategy:
            mock_strategy.side_effect = Exception("Strategy execution failed")

            result = await self.processor.process_query(
                query_text="test query",
                strategy=QueryStrategy.ONE_SHOT,
            )

            # Verify error result
            assert len(result.chunks) == 0
            assert len(result.responses) == 0
            assert result.successful_paths == 0
            assert result.failed_paths == 1
            assert result.query_text == "test query"
            assert result.strategy_used == "one_shot"
            assert result.dropped_paths is not None
            assert "Strategy execution failed" in result.dropped_paths[0]

    def test_get_query_statistics(self) -> None:
        """Test getting query statistics."""
        stats = self.processor.get_query_statistics()

        assert stats["model_name"] == "openai:gpt-4o-mini"
        assert stats["default_chunk_limit"] == 100
        assert stats["hierarchy_depth"] == 3
        assert "available_strategies" in stats
        assert len(stats["available_strategies"]) == 4  # Four strategies

    def test_validate_query_parameters_valid(self) -> None:
        """Test validation of valid query parameters."""
        errors = self.processor.validate_query_parameters(
            query_text="valid query",
            strategy=QueryStrategy.ONE_SHOT,
            strategy_params={},
        )

        assert errors == []

    def test_validate_query_parameters_invalid_query(self) -> None:
        """Test validation of invalid query text."""
        # Empty query
        errors = self.processor.validate_query_parameters(
            query_text="", strategy=QueryStrategy.ONE_SHOT
        )
        assert any("cannot be empty" in error for error in errors)

        # Too long query
        long_query = "x" * 1001
        errors = self.processor.validate_query_parameters(
            query_text=long_query, strategy=QueryStrategy.ONE_SHOT
        )
        assert any("too long" in error for error in errors)

    def test_validate_query_parameters_invalid_strategy(self) -> None:
        """Test validation of invalid strategy."""
        errors = self.processor.validate_query_parameters(
            query_text="valid query",
            strategy="invalid_strategy",  # Not a QueryStrategy enum
        )

        assert any("Invalid strategy type" in error for error in errors)

    def test_validate_query_parameters_invalid_strategy_params(self) -> None:
        """Test validation of invalid strategy parameters."""
        with patch(
            "memoir_ai.query.query_processor.validate_strategy_params"
        ) as mock_validate:
            mock_validate.side_effect = ValidationError(
                "Invalid parameter", field="n", value=-1
            )

            errors = self.processor.validate_query_parameters(
                query_text="valid query",
                strategy=QueryStrategy.WIDE_BRANCH,
                strategy_params={"n": -1},
            )

            assert any("Invalid strategy parameters" in error for error in errors)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_query_processor(self) -> None:
        """Test create_query_processor function."""
        mock_category_manager = Mock()
        mock_session = Mock()

        processor = create_query_processor(
            category_manager=mock_category_manager,
            session=mock_session,
            model_name="openai:gpt-4o-mini",
            default_chunk_limit=50,
        )

        assert isinstance(processor, QueryProcessor)
        assert processor.category_manager == mock_category_manager
        assert processor.session == mock_session
        assert processor.model_name == "openai:gpt-4o-mini"
        assert processor.default_chunk_limit == 50

    @pytest.mark.asyncio
    async def test_process_natural_language_query(self) -> None:
        """Test process_natural_language_query convenience function."""
        mock_category_manager = Mock()
        mock_session = Mock()

        with (
            patch(
                "memoir_ai.query.query_processor.create_query_processor"
            ) as mock_create,
            patch.object(
                QueryProcessor, "process_query", new_callable=AsyncMock
            ) as mock_process,
        ):
            mock_processor = Mock()
            mock_create.return_value = mock_processor

            expected_result = QueryResult(
                chunks=[],
                responses=[],
                total_latency_ms=0,
                total_chunks=0,
                successful_paths=0,
                failed_paths=0,
            )
            mock_process.return_value = expected_result

            result = await process_natural_language_query(
                query_text="test query",
                category_manager=mock_category_manager,
                session=mock_session,
                strategy=QueryStrategy.ZOOM_IN,
                contextual_helper="test context",
            )

            # Verify processor was created correctly
            mock_create.assert_called_once_with(
                category_manager=mock_category_manager, session=mock_session
            )

            # Verify query was processed correctly
            mock_process.assert_called_once_with(
                query_text="test query",
                strategy=QueryStrategy.ZOOM_IN,
                contextual_helper="test context",
            )

            assert result == expected_result


@pytest.mark.integration
class TestQueryProcessorIntegration:
    """Integration tests for query processor."""

    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self) -> None:
        """Test complete end-to-end query processing."""
        # This would be a comprehensive integration test
        # that tests the entire pipeline with real-like data
        pass

    @pytest.mark.asyncio
    async def test_query_processing_with_multiple_strategies(self) -> None:
        """Test query processing with different strategies."""
        # Test all four strategies with the same query
        pass

    @pytest.mark.asyncio
    async def test_query_processing_performance(self) -> None:
        """Test query processing performance with large datasets."""
        # Test performance characteristics
        pass
