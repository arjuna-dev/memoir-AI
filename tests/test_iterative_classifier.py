"""
Tests for iterative classification workflow.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from memoir_ai.classification.category_manager import CategoryManager
from memoir_ai.classification.iterative_classifier import (
    ClassificationWorkflowMetrics,
    IterativeClassificationResult,
    IterativeClassificationWorkflow,
    create_iterative_classifier,
)
from memoir_ai.database.models import Category, Chunk
from memoir_ai.exceptions import ClassificationError, ValidationError
from memoir_ai.llm.schemas import CategorySelection
from memoir_ai.text_processing.chunker import TextChunk


class TestIterativeClassificationResult:
    """Test IterativeClassificationResult data class."""

    def test_iterative_classification_result_success(self) -> None:
        """Test creating successful classification result."""
        chunk = TextChunk(
            content="test", token_count=5, start_position=0, end_position=4
        )
        categories = [
            Category(id=1, name="Tech", level=1, parent_id=None),
            Category(id=2, name="AI", level=2, parent_id=1),
        ]

        result = IterativeClassificationResult(
            chunk=chunk,
            category_path=categories,
            final_category=categories[-1],
            success=True,
            total_latency_ms=1500,
            llm_calls=2,
            levels_processed=2,
        )

        assert result.chunk == chunk
        assert result.category_path == categories
        assert result.final_category == categories[-1]
        assert result.success is True
        assert result.total_latency_ms == 1500
        assert result.llm_calls == 2
        assert result.levels_processed == 2
        assert result.error is None

    def test_iterative_classification_result_failure(self) -> None:
        """Test creating failed classification result."""
        chunk = TextChunk(
            content="test", token_count=5, start_position=0, end_position=4
        )

        result = IterativeClassificationResult(
            chunk=chunk,
            category_path=[],
            final_category=None,
            success=False,
            total_latency_ms=500,
            llm_calls=1,
            levels_processed=0,
            error="Classification failed",
        )

        assert result.chunk == chunk
        assert result.success is False
        assert result.error == "Classification failed"


class TestClassificationWorkflowMetrics:
    """Test ClassificationWorkflowMetrics data class."""

    def test_workflow_metrics_creation(self) -> None:
        """Test creating workflow metrics."""
        timestamp = datetime.now()

        metrics = ClassificationWorkflowMetrics(
            workflow_id="test_workflow",
            chunks_processed=10,
            chunks_successful=8,
            chunks_failed=2,
            total_latency_ms=5000,
            total_llm_calls=24,
            average_levels_per_chunk=2.4,
            timestamp=timestamp,
            model_name="openai:gpt-4o-mini",
        )

        assert metrics.workflow_id == "test_workflow"
        assert metrics.chunks_processed == 10
        assert metrics.chunks_successful == 8
        assert metrics.chunks_failed == 2
        assert metrics.total_latency_ms == 5000
        assert metrics.total_llm_calls == 24
        assert metrics.average_levels_per_chunk == 2.4
        assert metrics.timestamp == timestamp
        assert metrics.model_name == "openai:gpt-4o-mini"


class TestIterativeClassificationWorkflow:
    """Test IterativeClassificationWorkflow functionality."""

    def create_mock_session(self) -> None:
        """Create a mock database session."""
        session = Mock()
        session.add = Mock()
        session.flush = Mock()
        session.rollback = Mock()
        return session

    def create_mock_category_manager(self) -> None:
        """Create a mock category manager."""
        manager = Mock(spec=CategoryManager)
        manager.hierarchy_depth = 3
        manager.limits = Mock()
        manager.limits.global_limit = 128
        return manager

    def test_workflow_initialization_defaults(self) -> None:
        """Test workflow initialization with defaults."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            assert workflow.db_session == session
            assert workflow.category_manager == category_manager
            assert workflow.model_name == "openai:gpt-4o-mini"
            assert workflow.use_batch_processing is True
            assert workflow.batch_size == 5
            assert workflow.max_retries == 3
            assert workflow.temperature == 0.0

    def test_workflow_initialization_custom(self) -> None:
        """Test workflow initialization with custom parameters."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(
                session,
                category_manager,
                model_name="anthropic:claude-3",
                use_batch_processing=False,
                batch_size=10,
                max_retries=5,
                temperature=0.7,
            )

            assert workflow.model_name == "anthropic:claude-3"
            assert workflow.use_batch_processing is False
            assert workflow.batch_size == 10
            assert workflow.max_retries == 5
            assert workflow.temperature == 0.7

    @pytest.mark.asyncio
    async def test_classify_chunks_empty(self) -> None:
        """Test classifying empty chunks list."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            results = await workflow.classify_chunks([], "test context")

            assert results == []

    @pytest.mark.asyncio
    async def test_classify_single_chunk_success(self) -> None:
        """Test successful single chunk classification."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        # Mock category manager methods
        category_manager.get_categories_for_llm_prompt.return_value = ([], True)
        category_manager.is_leaf_category.return_value = True

        # Mock LLM agent
        mock_agent = AsyncMock()
        mock_response = Mock()
        mock_response.data = CategorySelection(
            category="Technology", ranked_relevance=1
        )
        mock_agent.run_async.return_value = mock_response

        # Mock category creation/retrieval
        mock_categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="AI", level=2, parent_id=1),
            Category(id=3, name="ML", level=3, parent_id=2),
        ]

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent",
                return_value=mock_agent,
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            # Mock _get_or_create_category to return sequential categories
            async def mock_get_or_create_category(
                name, level, parent_id, can_create
            ) -> None:
                return mock_categories[level - 1]

            workflow._get_or_create_category = mock_get_or_create_category

            chunk = TextChunk(
                content="AI research content",
                token_count=10,
                start_position=0,
                end_position=20,
            )

            result = await workflow._classify_single_chunk(
                chunk, "test context", "test_workflow"
            )

            assert result.success is True
            assert len(result.category_path) == 3
            assert result.final_category == mock_categories[2]
            assert result.llm_calls == 3
            assert result.levels_processed == 3

    @pytest.mark.asyncio
    async def test_classify_at_level_success(self) -> None:
        """Test successful classification at a specific level."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        # Mock LLM agent
        mock_agent = AsyncMock()
        mock_response = Mock()
        mock_response.data = CategorySelection(
            category="Technology", ranked_relevance=1
        )
        mock_agent.run_async.return_value = mock_response

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent",
                return_value=mock_agent,
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="AI research", token_count=5, start_position=0, end_position=11
            )
            existing_categories = [
                Category(id=1, name="Technology", level=1, parent_id=None)
            ]

            result = await workflow._classify_at_level(
                chunk, 1, existing_categories, "test context", True
            )

            assert result == "Technology"

    @pytest.mark.asyncio
    async def test_classify_at_level_empty_response(self) -> None:
        """Test classification at level with empty LLM response."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        # Mock LLM agent with empty response
        mock_agent = AsyncMock()
        mock_response = Mock()
        mock_response.data = None
        mock_agent.run_async.return_value = mock_response

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent",
                return_value=mock_agent,
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )

            with pytest.raises(ClassificationError) as exc_info:
                await workflow._classify_at_level(chunk, 1, [], "test context", True)

            assert "Empty category response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_classify_at_level_cannot_create_new(self) -> None:
        """Test classification when cannot create new categories."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        # Mock LLM agent returning non-existing category
        mock_agent = AsyncMock()
        mock_response = Mock()
        mock_response.data = CategorySelection(
            category="NonExisting", ranked_relevance=1
        )
        mock_agent.run_async.return_value = mock_response

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent",
                return_value=mock_agent,
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )
            existing_categories = [
                Category(id=1, name="Technology", level=1, parent_id=None)
            ]

            result = await workflow._classify_at_level(
                chunk,
                1,
                existing_categories,
                "test context",
                False,  # Cannot create new
            )

            # Should fallback to first existing category
            assert result == "Technology"

    def test_create_level_prompt_with_existing_categories(self) -> None:
        """Test creating level prompt with existing categories."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="AI research", token_count=5, start_position=0, end_position=11
            )
            existing_categories = [
                Category(id=1, name="Technology", level=1, parent_id=None),
                Category(id=2, name="Science", level=1, parent_id=None),
            ]

            prompt = workflow._create_level_prompt(
                chunk, 1, existing_categories, "test context", True
            )

            assert "Document Context (root category): test context" in prompt
            assert "Classification Level: 1" in prompt
            assert "Existing Categories: Technology, Science" in prompt
            assert "AI research" in prompt
            assert "select from existing categories when possible" in prompt

    def test_create_level_prompt_with_parent_context(self) -> None:
        """Ensure parent category context is included for deeper levels."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="Quantum computing breakthroughs",
                token_count=15,
                start_position=0,
                end_position=30,
            )

            existing_categories = [
                Category(id=2, name="Quantum Mechanics", level=2, parent_id=1),
                Category(id=3, name="Particle Physics", level=2, parent_id=1),
            ]
            parent_categories = [
                Category(id=1, name="Physics", level=1, parent_id=None)
            ]

            prompt = workflow._create_level_prompt(
                chunk,
                2,
                existing_categories,
                "science context",
                True,
                parent_categories=parent_categories,
            )

            assert "Parent Category Path: Physics" in prompt
            assert "more specific" in prompt
            assert "Avoid repeating the exact parent category name" in prompt

    def test_create_level_prompt_no_existing_categories(self) -> None:
        """Test creating level prompt with no existing categories."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )

            prompt = workflow._create_level_prompt(chunk, 1, [], "test context", True)

            assert "No existing categories at this level" in prompt
            assert "You may create a new category" in prompt

    def test_create_level_prompt_cannot_create_new(self) -> None:
        """Test creating level prompt when cannot create new categories."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()
        category_manager.get_category_limit.return_value = 10

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )
            existing_categories = [
                Category(id=1, name="Technology", level=1, parent_id=None)
            ]

            prompt = workflow._create_level_prompt(
                chunk, 1, existing_categories, "test context", False
            )

            assert "Category limit (10) reached" in prompt
            assert "MUST select from existing categories only" in prompt

    @pytest.mark.asyncio
    async def test_get_or_create_category_existing(self) -> None:
        """Test getting existing category."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        existing_category = Category(id=1, name="Technology", level=1, parent_id=None)
        category_manager.get_existing_categories.return_value = [existing_category]

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            result = await workflow._get_or_create_category("Technology", 1, None, True)

            assert result == existing_category

    @pytest.mark.asyncio
    async def test_get_or_create_category_create_new(self) -> None:
        """Test creating new category."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        new_category = Category(id=2, name="NewCategory", level=1, parent_id=None)
        category_manager.get_existing_categories.return_value = []
        category_manager.create_category.return_value = new_category

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            result = await workflow._get_or_create_category(
                "NewCategory", 1, None, True
            )

            assert result == new_category
            category_manager.create_category.assert_called_once_with(
                name="NewCategory", level=1, parent_id=None
            )

    @pytest.mark.asyncio
    async def test_get_or_create_category_cannot_create_fallback(self) -> None:
        """Test fallback to existing category when cannot create new."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        existing_category = Category(id=1, name="Technology", level=1, parent_id=None)
        category_manager.get_existing_categories.return_value = [existing_category]

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            result = await workflow._get_or_create_category(
                "NonExisting", 1, None, False
            )

            assert result == existing_category

    @pytest.mark.asyncio
    async def test_get_or_create_category_cannot_create_no_existing(self) -> None:
        """Test error when cannot create new and no existing categories."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        category_manager.get_existing_categories.return_value = []

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            with pytest.raises(ClassificationError) as exc_info:
                await workflow._get_or_create_category("Test", 1, None, False)

            assert "No categories available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_chunk_classification_success(self) -> None:
        """Test successful chunk classification storage."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()
        category_manager.is_leaf_category.return_value = True

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )
            category = Category(id=1, name="Technology", level=3, parent_id=2)

            await workflow._store_chunk_classification(chunk, category, "source123")

            session.add.assert_called_once()
            session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_chunk_classification_not_leaf(self) -> None:
        """Test storing chunk classification with non-leaf category."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()
        category_manager.is_leaf_category.return_value = False
        category_manager.hierarchy_depth = 3

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            chunk = TextChunk(
                content="test", token_count=1, start_position=0, end_position=4
            )
            category = Category(id=1, name="Technology", level=1, parent_id=None)

            with pytest.raises(ValidationError) as exc_info:
                await workflow._store_chunk_classification(chunk, category, "source123")

            assert "Can only link chunks to leaf categories" in str(exc_info.value)

    def test_get_workflow_metrics_summary_empty(self) -> None:
        """Test workflow metrics summary with no data."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            summary = workflow.get_workflow_metrics_summary()

            assert summary["total_workflows"] == 0
            assert summary["total_chunks"] == 0
            assert summary["success_rate"] == 0.0
            assert summary["average_latency_ms"] == 0.0
            assert summary["total_llm_calls"] == 0
            assert summary["average_levels_per_chunk"] == 0.0

    def test_get_workflow_metrics_summary_with_data(self) -> None:
        """Test workflow metrics summary with data."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            # Add sample metrics
            workflow.metrics_history = [
                ClassificationWorkflowMetrics(
                    workflow_id="workflow1",
                    chunks_processed=5,
                    chunks_successful=4,
                    chunks_failed=1,
                    total_latency_ms=2000,
                    total_llm_calls=12,
                    average_levels_per_chunk=2.5,
                    timestamp=datetime.now(),
                    model_name="openai:gpt-4o-mini",
                ),
                ClassificationWorkflowMetrics(
                    workflow_id="workflow2",
                    chunks_processed=3,
                    chunks_successful=3,
                    chunks_failed=0,
                    total_latency_ms=1500,
                    total_llm_calls=9,
                    average_levels_per_chunk=3.0,
                    timestamp=datetime.now(),
                    model_name="openai:gpt-4o-mini",
                ),
            ]

            summary = workflow.get_workflow_metrics_summary()

            assert summary["total_workflows"] == 2
            assert summary["total_chunks"] == 8
            assert summary["total_successful"] == 7
            assert summary["total_failed"] == 1
            assert summary["success_rate"] == 7 / 8
            assert summary["average_latency_ms"] == 1750.0  # (2000 + 1500) / 2
            assert summary["total_llm_calls"] == 21
            assert summary["average_llm_calls_per_chunk"] == 21 / 8
            # Weighted average: (4*2.5 + 3*3.0) / 7 = 19/7 ≈ 2.71
            assert abs(summary["average_levels_per_chunk"] - 19 / 7) < 0.01

    def test_clear_metrics(self) -> None:
        """Test clearing workflow metrics."""
        session = self.create_mock_session()
        category_manager = self.create_mock_category_manager()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            workflow = IterativeClassificationWorkflow(session, category_manager)

            # Add some metrics
            workflow.metrics_history = [Mock(), Mock()]
            assert len(workflow.metrics_history) == 2

            workflow.clear_metrics()
            assert len(workflow.metrics_history) == 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_iterative_classifier(self) -> None:
        """Test iterative classifier creation utility."""
        session = Mock()
        category_manager = Mock()

        with (
            patch(
                "memoir_ai.classification.iterative_classifier.BatchCategoryClassifier"
            ),
            patch(
                "memoir_ai.classification.iterative_classifier.create_classification_agent"
            ),
        ):
            classifier = create_iterative_classifier(
                session,
                category_manager,
                model_name="anthropic:claude-3",
                batch_size=10,
            )

            assert isinstance(classifier, IterativeClassificationWorkflow)
            assert classifier.db_session == session
            assert classifier.category_manager == category_manager
            assert classifier.model_name == "anthropic:claude-3"
            assert classifier.batch_size == 10
