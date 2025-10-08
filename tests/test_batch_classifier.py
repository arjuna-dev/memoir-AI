"""
Tests for batch classification system.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from memoir_ai.classification.batch_classifier import (
    BatchCategoryClassifier,
    BatchClassificationMetrics,
    ClassificationResult,
    create_batch_classifier,
    validate_batch_size,
)
from memoir_ai.database.models import Category
from memoir_ai.exceptions import ClassificationError, ValidationError
from memoir_ai.llm.llm_models import Model, Models
from memoir_ai.llm.schemas import (
    BatchClassificationResponse,
    CategorySelection,
    ChunkClassificationRequest,
)
from memoir_ai.text_processing.chunker import TextChunk


class TestBatchCategoryClassifier:
    """Test BatchCategoryClassifier functionality."""

    def test_initialization_defaults(self) -> None:
        """Test classifier initialization with defaults."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            assert classifier.model_name == "openai:gpt-4.1-mini"
            assert classifier.batch_size == 5
            assert classifier.max_retries == 3
            assert classifier.hierarchy_depth == 3
            assert classifier.max_categories_per_level == 128
            assert classifier.temperature == 0.0

    def test_initialization_custom(self) -> None:
        """Test classifier initialization with custom parameters."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier(
                model=Models.anthropic_claude_3_5_haiku_20241022,
                batch_size=10,
                max_retries=5,
                hierarchy_depth=5,
                max_categories_per_level={1: 50, 2: 100},
                temperature=0.7,
            )

            assert classifier.model_name == "anthropic:claude-3-5-haiku-20241022"
            assert classifier.batch_size == 10
            assert classifier.max_retries == 5
            assert classifier.hierarchy_depth == 5
            assert classifier.max_categories_per_level == {1: 50, 2: 100}
            assert classifier.temperature == 0.7

    def test_initialization_validation(self) -> None:
        """Test classifier initialization validation."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            # Test invalid batch size
            with pytest.raises(ValidationError) as exc_info:
                BatchCategoryClassifier(batch_size=0)
            assert "batch_size must be positive" in str(exc_info.value)

            # Test negative retries
            with pytest.raises(ValidationError) as exc_info:
                BatchCategoryClassifier(max_retries=-1)
            assert "max_retries cannot be negative" in str(exc_info.value)

            # Test invalid hierarchy depth
            with pytest.raises(ValidationError) as exc_info:
                BatchCategoryClassifier(hierarchy_depth=0)
            assert "hierarchy_depth must be between 1 and 100" in str(exc_info.value)

    def test_create_batches(self) -> None:
        """Test batch creation from chunks."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier(batch_size=3)

            # Create test chunks
            chunks = [
                TextChunk(
                    content=f"Content {i}",
                    token_count=10,
                    start_position=i * 10,
                    end_position=(i + 1) * 10,
                )
                for i in range(7)
            ]

            batches = classifier._create_batches(chunks)

            assert len(batches) == 3  # 7 chunks / 3 per batch = 3 batches
            assert len(batches[0]) == 3
            assert len(batches[1]) == 3
            assert len(batches[2]) == 1

    def test_create_batch_prompt_level_by_level(self) -> None:
        """Test batch prompt creation for level by level processing."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            # Create test data
            chunks = [
                TextChunk(
                    content="AI research content",
                    token_count=10,
                    start_position=0,
                    end_position=20,
                ),
                TextChunk(
                    content="Machine learning algorithms",
                    token_count=12,
                    start_position=21,
                    end_position=50,
                ),
            ]

            categories = [
                Category(id=1, name="Technology", level=1),
                Category(id=2, name="Science", level=1),
            ]

            contextual_helper = "Research paper about artificial intelligence"

            prompt = classifier._create_batch_prompt_level_by_level(
                chunks, categories, contextual_helper, 1
            )

            # Check prompt structure
            assert (
                "Document Context: Research paper about artificial intelligence"
                in prompt
            )
            assert "Classification Level: 1" in prompt
            assert "Existing Categories: Technology, Science" in prompt
            assert "Chunk 1:" in prompt
            assert "AI research content" in prompt
            assert "Chunk 2:" in prompt
            assert "Machine learning algorithms" in prompt
            assert '"""' in prompt  # Check for proper chunk delimiters

    def test_create_batch_prompt_level_by_level_no_existing_categories(self) -> None:
        """Test batch prompt creation with no existing categories."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunks = [
                TextChunk(
                    content="Test content",
                    token_count=5,
                    start_position=0,
                    end_position=12,
                ),
            ]

            prompt = classifier._create_batch_prompt_level_by_level(
                chunks, [], "Test context", 1
            )

            assert "No existing categories at this level" in prompt
            assert "You may create new categories" in prompt

    def test_create_batch_prompt_level_by_level_category_limit_reached(self) -> None:
        """Test batch prompt creation when category limit is reached."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier(max_categories_per_level=2)

            chunks = [
                TextChunk(
                    content="Test content",
                    token_count=5,
                    start_position=0,
                    end_position=12,
                ),
            ]

            # Create categories that reach the limit
            categories = [
                Category(id=1, name="Category1", level=1),
                Category(id=2, name="Category2", level=1),
            ]

            prompt = classifier._create_batch_prompt_level_by_level(
                chunks, categories, "Test context", 1
            )

            assert "Category limit (2) reached" in prompt
            assert "MUST select from existing categories only" in prompt

    def test_validate_batch_response_valid(self) -> None:
        """Test validation of valid batch response."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            response = BatchClassificationResponse(
                chunks=[
                    ChunkClassificationRequest(chunk_id=1, category="Technology"),
                    ChunkClassificationRequest(chunk_id=2, category="Science"),
                ]
            )

            assert classifier._validate_batch_response(response, 2) is True

    def test_validate_batch_response_invalid(self) -> None:
        """Test validation of invalid batch responses."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            # Test empty response
            assert classifier._validate_batch_response(None, 2) is False

            # Test wrong count
            response = BatchClassificationResponse(
                chunks=[ChunkClassificationRequest(chunk_id=1, category="Technology")]
            )
            assert classifier._validate_batch_response(response, 2) is False

            # Test duplicate IDs
            response = BatchClassificationResponse(
                chunks=[
                    ChunkClassificationRequest(chunk_id=1, category="Technology"),
                    ChunkClassificationRequest(chunk_id=1, category="Science"),
                ]
            )
            assert classifier._validate_batch_response(response, 2) is False

            # Test invalid ID range
            response = BatchClassificationResponse(
                chunks=[
                    ChunkClassificationRequest(chunk_id=0, category="Technology"),
                    ChunkClassificationRequest(chunk_id=2, category="Science"),
                ]
            )
            assert classifier._validate_batch_response(response, 2) is False

            # Test empty category
            response = BatchClassificationResponse(
                chunks=[
                    ChunkClassificationRequest(chunk_id=1, category=""),
                    ChunkClassificationRequest(chunk_id=2, category="Science"),
                ]
            )
            assert classifier._validate_batch_response(response, 2) is False

    def test_create_single_chunk_prompt(self) -> None:
        """Test single chunk prompt creation for retries."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunk = TextChunk(
                content="AI research content",
                token_count=10,
                start_position=0,
                end_position=20,
            )
            categories = [Category(id=1, name="Technology", level=1)]

            prompt = classifier._create_single_chunk_prompt(
                chunk, categories, "Test context", 1
            )

            assert "Document Context: Test context" in prompt
            assert "Classification Level: 1" in prompt
            assert "Existing Categories: Technology" in prompt
            assert "AI research content" in prompt
            assert "Provide the category name only" in prompt

    @pytest.mark.asyncio
    async def test_classify_chunks_batch_empty(self) -> None:
        """Test batch classification with empty chunks list."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            results = await classifier.classify_chunks_batch(
                chunks=[],
                level=1,
                parent_category=None,
                existing_categories=[],
                contextual_helper="Test context",
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_success(self) -> None:
        """Test successful batch processing."""
        # Mock agents
        mock_batch_agent = AsyncMock()
        mock_single_agent = AsyncMock()

        # Mock successful response
        mock_response = Mock()
        mock_response.data = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category="Technology"),
                ChunkClassificationRequest(chunk_id=2, category="Science"),
            ]
        )
        mock_batch_agent.run_async.return_value = mock_response

        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent",
                return_value=mock_batch_agent,
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent",
                return_value=mock_single_agent,
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunks = [
                TextChunk(
                    content="AI content",
                    token_count=10,
                    start_position=0,
                    end_position=10,
                ),
                TextChunk(
                    content="Science content",
                    token_count=12,
                    start_position=11,
                    end_position=25,
                ),
            ]

            results = await classifier._process_batch(
                chunks=chunks,
                level=1,
                parent_category=None,
                existing_categories=[],
                contextual_helper="Test context",
                batch_id="test_batch",
            )

            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].category == "Technology"
            assert results[1].category == "Science"
            assert results[0].chunk_id == 1
            assert results[1].chunk_id == 2

    @pytest.mark.asyncio
    async def test_process_batch_with_retries(self) -> None:
        """Test batch processing with failed chunks requiring retries."""
        # Mock agents
        mock_batch_agent = AsyncMock()
        mock_single_agent = AsyncMock()

        # Mock batch response with complete response but we'll simulate partial failure
        mock_batch_response = Mock()
        mock_batch_response.data = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category="Technology"),
                ChunkClassificationRequest(
                    chunk_id=2, category=""
                ),  # Empty category triggers retry
            ]
        )
        mock_batch_agent.run_async.return_value = mock_batch_response

        # Mock successful retry response
        mock_retry_response = Mock()
        mock_retry_response.data = CategorySelection(
            category="Science", ranked_relevance=1
        )
        mock_single_agent.run_async.return_value = mock_retry_response

        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent",
                return_value=mock_batch_agent,
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent",
                return_value=mock_single_agent,
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunks = [
                TextChunk(
                    content="AI content",
                    token_count=10,
                    start_position=0,
                    end_position=10,
                ),
                TextChunk(
                    content="Science content",
                    token_count=12,
                    start_position=11,
                    end_position=25,
                ),
            ]

            results = await classifier._process_batch(
                chunks=chunks,
                level=1,
                parent_category=None,
                existing_categories=[],
                contextual_helper="Test context",
                batch_id="test_batch",
            )

            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].category == "Technology"
            assert results[1].category == "Science"
            assert results[0].retry_count == 0  # No retry needed
            assert results[1].retry_count == 1  # One retry

    @pytest.mark.asyncio
    async def test_retry_failed_chunks_success(self) -> None:
        """Test successful retry of failed chunks."""
        mock_batch_agent = AsyncMock()
        mock_single_agent = AsyncMock()

        # Mock successful retry response
        mock_response = Mock()
        mock_response.data = CategorySelection(
            category="Retried Category", ranked_relevance=1
        )
        mock_single_agent.run_async.return_value = mock_response

        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent",
                return_value=mock_batch_agent,
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent",
                return_value=mock_single_agent,
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunk = TextChunk(
                content="Failed content",
                token_count=10,
                start_position=0,
                end_position=13,
            )
            failed_chunks = [(chunk, 1)]

            results = await classifier._retry_failed_chunks(
                failed_chunks=failed_chunks,
                level=1,
                existing_categories=[],
                contextual_helper="Test context",
            )

            assert len(results) == 1
            assert results[0].success is True
            assert results[0].category == "Retried Category"
            assert results[0].retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_failed_chunks_all_retries_fail(self) -> None:
        """Test retry when all attempts fail."""
        mock_batch_agent = AsyncMock()
        mock_single_agent = AsyncMock()

        # Mock failed retry responses
        mock_single_agent.run_async.side_effect = Exception("Retry failed")

        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent",
                return_value=mock_batch_agent,
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent",
                return_value=mock_single_agent,
            ),
        ):
            classifier = BatchCategoryClassifier(max_retries=2)

            chunk = TextChunk(
                content="Failed content",
                token_count=10,
                start_position=0,
                end_position=13,
            )
            failed_chunks = [(chunk, 1)]

            results = await classifier._retry_failed_chunks(
                failed_chunks=failed_chunks,
                level=1,
                existing_categories=[],
                contextual_helper="Test context",
            )

            assert len(results) == 1
            assert results[0].success is False
            assert results[0].category == ""
            assert "Failed after 2 retries" in results[0].error
            assert results[0].retry_count == 2

    def test_record_batch_metrics(self) -> None:
        """Test batch metrics recording."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            chunks = [Mock(), Mock(), Mock()]
            results = [
                ClassificationResult(
                    chunk_id=1, chunk=Mock(), category="Cat1", success=True
                ),
                ClassificationResult(
                    chunk_id=2,
                    chunk=Mock(),
                    category="Cat2",
                    success=True,
                    retry_count=1,
                ),
                ClassificationResult(
                    chunk_id=3, chunk=Mock(), category="", success=False
                ),
            ]

            classifier._record_batch_metrics("test_batch", chunks, results, 1500, 2)

            assert len(classifier.metrics_history) == 1
            metrics = classifier.metrics_history[0]

            assert metrics.batch_id == "test_batch"
            assert metrics.chunks_sent == 3
            assert metrics.chunks_successful == 2
            assert metrics.chunks_failed == 1
            assert metrics.chunks_retried == 1
            assert metrics.total_latency_ms == 1500
            assert metrics.llm_calls == 2

    def test_get_metrics_summary(self) -> None:
        """Test metrics summary generation."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            # Add some test metrics
            classifier.metrics_history = [
                BatchClassificationMetrics(
                    batch_id="batch1",
                    chunks_sent=5,
                    chunks_successful=4,
                    chunks_failed=1,
                    chunks_retried=1,
                    total_latency_ms=1000,
                    llm_calls=2,
                    timestamp=datetime.now(),
                    model_name="test-model",
                ),
                BatchClassificationMetrics(
                    batch_id="batch2",
                    chunks_sent=3,
                    chunks_successful=3,
                    chunks_failed=0,
                    chunks_retried=0,
                    total_latency_ms=800,
                    llm_calls=1,
                    timestamp=datetime.now(),
                    model_name="test-model",
                ),
            ]

            summary = classifier.get_metrics_summary()

            assert summary["total_batches"] == 2
            assert summary["total_chunks"] == 8
            assert summary["total_successful"] == 7
            assert summary["total_failed"] == 1
            assert summary["success_rate"] == 7 / 8
            assert summary["average_latency_ms"] == 900.0  # (1000 + 800) / 2
            assert summary["total_llm_calls"] == 3
            assert summary["average_chunks_per_batch"] == 4.0

    def test_get_metrics_summary_empty(self) -> None:
        """Test metrics summary with no data."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            summary = classifier.get_metrics_summary()

            assert summary["total_batches"] == 0
            assert summary["total_chunks"] == 0
            assert summary["success_rate"] == 0.0
            assert summary["average_latency_ms"] == 0.0
            assert summary["total_llm_calls"] == 0

    def test_clear_metrics(self) -> None:
        """Test clearing metrics history."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = BatchCategoryClassifier()

            # Add some metrics
            classifier.metrics_history = [Mock(), Mock()]
            assert len(classifier.metrics_history) == 2

            classifier.clear_metrics()
            assert len(classifier.metrics_history) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_batch_classifier(self) -> None:
        """Test batch classifier creation utility."""
        with (
            patch(
                "memoir_ai.classification.batch_classifier.create_batch_classification_agent"
            ),
            patch(
                "memoir_ai.classification.batch_classifier.create_classification_agent"
            ),
        ):
            classifier = create_batch_classifier(
                model=Models.openai_gpt_4o_mini, batch_size=10, max_retries=5
            )

            assert isinstance(classifier, BatchCategoryClassifier)
            assert classifier.model_name == "openai:gpt-5-nano"
            assert classifier.batch_size == 10
            assert classifier.max_retries == 5

    def test_validate_batch_size(self) -> None:
        """Test batch size validation utility."""
        assert validate_batch_size(1) is True
        assert validate_batch_size(5) is True
        assert validate_batch_size(50) is True

        assert validate_batch_size(0) is False
        assert validate_batch_size(-1) is False


class TestClassificationResult:
    """Test ClassificationResult data class."""

    def test_classification_result_creation(self) -> None:
        """Test creating classification result."""
        chunk = TextChunk(
            content="test", token_count=5, start_position=0, end_position=4
        )

        result = ClassificationResult(
            chunk_id=1,
            chunk=chunk,
            category="Technology",
            success=True,
            retry_count=2,
            latency_ms=500,
        )

        assert result.chunk_id == 1
        assert result.chunk == chunk
        assert result.category == "Technology"
        assert result.success is True
        assert result.error is None
        assert result.retry_count == 2
        assert result.latency_ms == 500

    def test_classification_result_failed(self) -> None:
        """Test creating failed classification result."""
        chunk = TextChunk(
            content="test", token_count=5, start_position=0, end_position=4
        )

        result = ClassificationResult(
            chunk_id=1,
            chunk=chunk,
            category="",
            success=False,
            error="Classification failed",
        )

        assert result.chunk_id == 1
        assert result.success is False
        assert result.error == "Classification failed"
        assert result.retry_count == 0


class TestBatchClassificationMetrics:
    """Test BatchClassificationMetrics data class."""

    def test_metrics_creation(self) -> None:
        """Test creating batch metrics."""
        timestamp = datetime.now()

        metrics = BatchClassificationMetrics(
            batch_id="test_batch",
            chunks_sent=5,
            chunks_successful=4,
            chunks_failed=1,
            chunks_retried=1,
            total_latency_ms=1500,
            llm_calls=2,
            timestamp=timestamp,
            model_name="test-model",
        )

        assert metrics.batch_id == "test_batch"
        assert metrics.chunks_sent == 5
        assert metrics.chunks_successful == 4
        assert metrics.chunks_failed == 1
        assert metrics.chunks_retried == 1
        assert metrics.total_latency_ms == 1500
        assert metrics.llm_calls == 2
        assert metrics.timestamp == timestamp
        assert metrics.model_name == "test-model"
