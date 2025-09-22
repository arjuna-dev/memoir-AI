"""
Tests for chunk retrieval and result construction.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from memoir_ai.database.models import Category, Chunk
from memoir_ai.exceptions import DatabaseError
from memoir_ai.query.chunk_retrieval import (
    ChunkResult,
    ChunkRetriever,
    PathRetrievalResult,
    QueryResult,
    ResultConstructor,
    create_chunk_retriever,
    create_result_constructor,
)
from memoir_ai.query.query_strategy_engine import (
    CategoryPath,
    LLMCallResponse,
    QueryClassificationResult,
)


class TestChunkResult:
    """Test ChunkResult functionality."""

    def test_chunk_result_creation(self) -> None:
        """Test ChunkResult creation."""
        now = datetime.now()

        chunk_result = ChunkResult(
            chunk_id=1,
            text_content="This is test content",
            category_path="Technology > AI > ML",
            category_id_path="1/2/3",
            ranked_relevance=5,
            created_at=now,
            source_id="test_source",
            token_count=100,
        )

        assert chunk_result.chunk_id == 1
        assert chunk_result.text_content == "This is test content"
        assert chunk_result.category_path == "Technology > AI > ML"
        assert chunk_result.ranked_relevance == 5
        assert chunk_result.created_at == now
        assert chunk_result.source_id == "test_source"
        assert chunk_result.token_count == 100


class TestPathRetrievalResult:
    """Test PathRetrievalResult functionality."""

    def test_path_retrieval_result_success(self) -> None:
        """Test successful path retrieval result."""
        categories = [
            Category(id=1, name="Technology", level=1),
            Category(id=2, name="AI", level=2, parent_id=1),
        ]
        category_path = CategoryPath(path=categories, ranked_relevance=5)

        chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="Test content",
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=datetime.now(),
            )
        ]

        result = PathRetrievalResult(
            category_path=category_path,
            chunks=chunks,
            chunk_count=1,
            retrieval_latency_ms=150,
            success=True,
        )

        assert result.category_path == category_path
        assert len(result.chunks) == 1
        assert result.chunk_count == 1
        assert result.retrieval_latency_ms == 150
        assert result.success is True
        assert result.error is None

    def test_path_retrieval_result_failure(self) -> None:
        """Test failed path retrieval result."""
        categories = [Category(id=1, name="Technology", level=1)]
        category_path = CategoryPath(path=categories, ranked_relevance=3)

        result = PathRetrievalResult(
            category_path=category_path,
            chunks=[],
            chunk_count=0,
            retrieval_latency_ms=50,
            success=False,
            error="No chunks found in leaf category",
        )

        assert result.success is False
        assert result.error == "No chunks found in leaf category"
        assert len(result.chunks) == 0


class TestQueryResult:
    """Test QueryResult functionality."""

    def test_query_result_creation(self) -> None:
        """Test QueryResult creation with all fields."""
        chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="Content 1",
                category_path="Tech > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=datetime.now(),
            )
        ]

        responses = [
            LLMCallResponse(
                llm_output=QueryClassificationResult(category="AI", ranked_relevance=5),
                timestamp=datetime.now(),
                latency_ms=100,
            )
        ]

        result = QueryResult(
            chunks=chunks,
            responses=responses,
            total_latency_ms=250,
            total_chunks=1,
            successful_paths=1,
            failed_paths=0,
            query_text="test query",
            strategy_used="one_shot",
        )

        assert len(result.chunks) == 1
        assert len(result.responses) == 1
        assert result.total_latency_ms == 250
        assert result.total_chunks == 1
        assert result.successful_paths == 1
        assert result.failed_paths == 0
        assert result.query_text == "test query"
        assert result.strategy_used == "one_shot"


class TestChunkRetriever:
    """Test ChunkRetriever functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.retriever = ChunkRetriever(session=self.mock_session, default_limit=100)

        # Mock categories
        self.categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="AI", level=2, parent_id=1),
            Category(id=3, name="ML", level=3, parent_id=2),
        ]

    def test_initialization(self) -> None:
        """Test ChunkRetriever initialization."""
        assert self.retriever.session == self.mock_session
        assert self.retriever.default_limit == 100
        assert self.retriever.enable_pagination is True

    def test_retrieve_chunks_for_paths_empty(self) -> None:
        """Test retrieving chunks for empty path list."""
        results = self.retriever.retrieve_chunks_for_paths([])
        assert results == []

    def test_retrieve_chunks_for_paths_success(self) -> None:
        """Test successful chunk retrieval for paths."""
        # Create category path
        category_path = CategoryPath(path=self.categories, ranked_relevance=5)

        # Mock SQL execution result
        mock_row = Mock()
        mock_row.chunk_id = 1
        mock_row.text_content = "Test content"
        mock_row.token_count = 50
        mock_row.source_id = "test_source"
        mock_row.created_at = datetime.now()
        mock_row.category_path = "Technology → AI → ML"
        mock_row.category_id_path = "1/2/3"
        mock_row.category_id = 3

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_row]))

        self.mock_session.execute.return_value = mock_result

        # Execute retrieval
        results = self.retriever.retrieve_chunks_for_paths([category_path])

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.category_path == category_path
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == 1
        assert result.chunks[0].text_content == "Test content"
        assert result.chunks[0].category_path == "Technology → AI → ML"

    def test_retrieve_chunks_for_paths_no_chunks(self) -> None:
        """Test retrieval when no chunks found."""
        category_path = CategoryPath(path=self.categories, ranked_relevance=3)

        # Mock empty result
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        self.mock_session.execute.return_value = mock_result

        results = self.retriever.retrieve_chunks_for_paths([category_path])

        assert len(results) == 1
        result = results[0]
        assert result.success is True  # Success but no chunks
        assert len(result.chunks) == 0
        assert result.error == "No chunks found in leaf category"

    def test_retrieve_chunks_for_paths_database_error(self) -> None:
        """Test handling of database errors during retrieval."""
        category_path = CategoryPath(path=self.categories, ranked_relevance=2)

        # Mock database error
        self.mock_session.execute.side_effect = Exception("Database connection failed")

        results = self.retriever.retrieve_chunks_for_paths([category_path])

        assert len(results) == 1
        result = results[0]
        assert result.success is False
        assert "Database connection failed" in result.error
        assert len(result.chunks) == 0

    def test_retrieve_chunks_for_single_path_empty_path(self) -> None:
        """Test retrieving chunks for empty category path."""
        empty_path = CategoryPath(path=[], ranked_relevance=1)

        chunks = self.retriever._retrieve_chunks_for_single_path(empty_path)
        assert chunks == []

    def test_get_chunk_count_for_path(self) -> None:
        """Test getting chunk count for a path."""
        category_path = CategoryPath(path=self.categories, ranked_relevance=4)

        # Mock query result
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 5

        self.mock_session.query.return_value = mock_query

        count = self.retriever.get_chunk_count_for_path(category_path)

        assert count == 5
        self.mock_session.query.assert_called_once_with(Chunk)

    def test_get_chunk_count_for_path_error(self) -> None:
        """Test error handling in chunk count."""
        category_path = CategoryPath(path=self.categories, ranked_relevance=2)

        # Mock error
        self.mock_session.query.side_effect = Exception("Query failed")

        count = self.retriever.get_chunk_count_for_path(category_path)
        assert count == 0


class TestResultConstructor:
    """Test ResultConstructor functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.constructor = ResultConstructor()

        # Create test data
        self.categories = [
            Category(id=1, name="Technology", level=1),
            Category(id=2, name="AI", level=2, parent_id=1),
        ]

        self.category_path = CategoryPath(path=self.categories, ranked_relevance=5)

        self.chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="First chunk",
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=datetime.now() - timedelta(minutes=1),
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Second chunk",
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=datetime.now(),
            ),
        ]

    def test_initialization(self) -> None:
        """Test ResultConstructor initialization."""
        constructor = ResultConstructor()
        assert constructor is not None

    def test_construct_query_result_basic(self) -> None:
        """Test basic query result construction."""
        path_results = [
            PathRetrievalResult(
                category_path=self.category_path,
                chunks=self.chunks,
                chunk_count=2,
                retrieval_latency_ms=100,
                success=True,
            )
        ]

        llm_responses = [
            LLMCallResponse(
                llm_output=QueryClassificationResult(category="AI", ranked_relevance=5),
                timestamp=datetime.now(),
                latency_ms=50,
            )
        ]

        result = self.constructor.construct_query_result(
            path_results=path_results,
            llm_responses=llm_responses,
            query_text="test query",
            strategy_used="one_shot",
        )

        assert len(result.chunks) == 2
        assert len(result.responses) == 1
        assert result.total_chunks == 2
        assert result.successful_paths == 1
        assert result.failed_paths == 0
        assert result.total_latency_ms == 150  # 100 + 50
        assert result.query_text == "test query"
        assert result.strategy_used == "one_shot"

    def test_construct_query_result_with_failures(self) -> None:
        """Test query result construction with failed paths."""
        successful_result = PathRetrievalResult(
            category_path=self.category_path,
            chunks=self.chunks[:1],
            chunk_count=1,
            retrieval_latency_ms=80,
            success=True,
        )

        failed_result = PathRetrievalResult(
            category_path=CategoryPath(
                path=[Category(id=3, name="Science", level=1)], ranked_relevance=3
            ),
            chunks=[],
            chunk_count=0,
            retrieval_latency_ms=20,
            success=False,
            error="No chunks found",
        )

        path_results = [successful_result, failed_result]
        llm_responses = []

        result = self.constructor.construct_query_result(
            path_results=path_results, llm_responses=llm_responses
        )

        assert len(result.chunks) == 1
        assert result.successful_paths == 1
        assert result.failed_paths == 1
        assert result.total_latency_ms == 100  # 80 + 20

    def test_construct_query_result_chunk_ordering(self) -> None:
        """Test that chunks are properly ordered in result."""
        # Create chunks with different timestamps
        chunk1 = ChunkResult(
            chunk_id=2,
            text_content="Later chunk",
            category_path="Tech > AI",
            category_id_path="1/2",
            ranked_relevance=5,
            created_at=datetime.now(),
        )

        chunk2 = ChunkResult(
            chunk_id=1,
            text_content="Earlier chunk",
            category_path="Tech > AI",
            category_id_path="1/2",
            ranked_relevance=5,
            created_at=datetime.now() - timedelta(minutes=1),
        )

        path_results = [
            PathRetrievalResult(
                category_path=self.category_path,
                chunks=[chunk1, chunk2],  # Intentionally out of order
                chunk_count=2,
                retrieval_latency_ms=100,
                success=True,
            )
        ]

        result = self.constructor.construct_query_result(
            path_results=path_results, llm_responses=[]
        )

        # Verify chunks are ordered by created_at, then chunk_id
        assert result.chunks[0].chunk_id == 1  # Earlier timestamp
        assert result.chunks[1].chunk_id == 2  # Later timestamp

    def test_validate_query_result_valid(self) -> None:
        """Test validation of a valid query result."""
        result = QueryResult(
            chunks=self.chunks,
            responses=[],
            total_latency_ms=100,
            total_chunks=2,
            successful_paths=1,
            failed_paths=0,
            path_results=[
                PathRetrievalResult(
                    category_path=self.category_path,
                    chunks=self.chunks,
                    chunk_count=2,
                    retrieval_latency_ms=100,
                    success=True,
                )
            ],
        )

        errors = self.constructor.validate_query_result(result)
        assert errors == []

    def test_validate_query_result_invalid(self) -> None:
        """Test validation of an invalid query result."""
        result = QueryResult(
            chunks=self.chunks,
            responses=[],
            total_latency_ms=100,
            total_chunks=5,  # Mismatch with actual chunk count
            successful_paths=2,  # Mismatch with path results
            failed_paths=1,
            path_results=[
                PathRetrievalResult(
                    category_path=self.category_path,
                    chunks=self.chunks,
                    chunk_count=2,
                    retrieval_latency_ms=100,
                    success=True,
                )
            ],
        )

        errors = self.constructor.validate_query_result(result)
        assert len(errors) >= 2  # Should have multiple validation errors
        assert any("Total chunks mismatch" in error for error in errors)
        assert any("Path count mismatch" in error for error in errors)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_chunk_retriever(self) -> None:
        """Test create_chunk_retriever function."""
        mock_session = Mock()

        retriever = create_chunk_retriever(session=mock_session, default_limit=50)

        assert isinstance(retriever, ChunkRetriever)
        assert retriever.session == mock_session
        assert retriever.default_limit == 50

    def test_create_result_constructor(self) -> None:
        """Test create_result_constructor function."""
        constructor = create_result_constructor()

        assert isinstance(constructor, ResultConstructor)


@pytest.mark.integration
class TestChunkRetrievalIntegration:
    """Integration tests for chunk retrieval system."""

    def test_end_to_end_retrieval_workflow(self) -> None:
        """Test complete end-to-end retrieval workflow."""
        # This would be a more comprehensive integration test
        # that tests the entire flow with real-like data
        pass

    def test_performance_with_large_datasets(self) -> None:
        """Test performance with large numbers of chunks."""
        # Test system behavior with many chunks and paths
        pass

    def test_concurrent_retrieval_requests(self) -> None:
        """Test handling of concurrent retrieval requests."""
        # Test thread safety and concurrent access
        pass
