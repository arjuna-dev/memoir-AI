"""
Tests for result aggregator.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from memoir_ai.aggregation.budget_manager import (
    BudgetConfig,
    BudgetManager,
    PromptLimitingStrategy,
    TokenEstimate,
)
from memoir_ai.aggregation.pruning_engine import (
    PruningEngine,
    PruningResult,
    PruningStrategy,
)
from memoir_ai.aggregation.result_aggregator import (
    AggregationResult,
    ResultAggregator,
    create_result_aggregator,
)
from memoir_ai.query.chunk_retrieval import ChunkResult, QueryResult
from memoir_ai.query.query_strategy_engine import (
    LLMCallResponse,
    QueryClassificationResult,
)


class TestAggregationResult:
    """Test AggregationResult functionality."""

    def test_aggregation_result_creation(self) -> None:
        """Test AggregationResult creation."""
        now = datetime.now()

        chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="Test content",
                category_path="Tech > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=now,
            )
        ]

        token_estimate = TokenEstimate(
            fixed_prompt_tokens=100,
            chunks_total_tokens=200,
            total_tokens=300,
            fixed_prompt_chars=400,
            chunks_total_chars=800,
            total_chars=1200,
            within_budget=True,
        )

        result = AggregationResult(
            final_chunks=chunks,
            final_prompt_text="Test prompt",
            token_estimate=token_estimate,
            within_budget=True,
            strategy_used=PromptLimitingStrategy.PRUNE,
            processing_latency_ms=150,
        )

        assert len(result.final_chunks) == 1
        assert result.final_prompt_text == "Test prompt"
        assert result.within_budget is True
        assert result.strategy_used == PromptLimitingStrategy.PRUNE
        assert result.processing_latency_ms == 150


class TestResultAggregator:
    """Test ResultAggregator functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create budget manager
        config = BudgetConfig(
            max_token_budget=2000,  # Larger budget to accommodate default headroom
            prompt_limiting_strategy=PromptLimitingStrategy.PRUNE,
            use_rankings=True,
        )
        self.budget_manager = BudgetManager(config=config)

        # Create pruning engine
        self.pruning_engine = PruningEngine(
            token_counter_func=self.budget_manager.count_tokens
        )

        # Create aggregator
        self.aggregator = ResultAggregator(
            budget_manager=self.budget_manager, pruning_engine=self.pruning_engine
        )

        # Create test data
        base_time = datetime.now()

        self.chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="High relevance content with detailed information",
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=base_time,
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Medium relevance content",
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=3,
                created_at=base_time + timedelta(minutes=1),
            ),
            ChunkResult(
                chunk_id=3,
                text_content="Low relevance content with some text",
                category_path="Technology > Web",
                category_id_path="1/3",
                ranked_relevance=1,
                created_at=base_time + timedelta(minutes=2),
            ),
        ]

        self.query_result = QueryResult(
            chunks=self.chunks,
            responses=[
                LLMCallResponse(
                    llm_output=QueryClassificationResult(
                        category="AI", ranked_relevance=5
                    ),
                    timestamp=base_time,
                    latency_ms=100,
                )
            ],
            total_latency_ms=200,
            total_chunks=len(self.chunks),
            successful_paths=2,
            failed_paths=0,
        )

    def test_initialization(self) -> None:
        """Test ResultAggregator initialization."""
        assert self.aggregator.budget_manager == self.budget_manager
        assert self.aggregator.pruning_engine == self.pruning_engine

    def test_initialization_default_pruning_engine(self) -> None:
        """Test ResultAggregator initialization with default pruning engine."""
        aggregator = ResultAggregator(budget_manager=self.budget_manager)

        assert aggregator.budget_manager == self.budget_manager
        assert aggregator.pruning_engine is not None
        assert isinstance(aggregator.pruning_engine, PruningEngine)

    def test_order_chunks_deterministically(self) -> None:
        """Test deterministic chunk ordering."""
        # Create chunks with different paths and timestamps
        base_time = datetime.now()

        unordered_chunks = [
            ChunkResult(
                chunk_id=3,
                text_content="Third",
                category_path="B > Category",
                category_id_path="2/3",
                ranked_relevance=1,
                created_at=base_time + timedelta(minutes=2),
            ),
            ChunkResult(
                chunk_id=1,
                text_content="First",
                category_path="A > Category",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=base_time,
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Second",
                category_path="A > Category",
                category_id_path="1/2",
                ranked_relevance=3,
                created_at=base_time + timedelta(minutes=1),
            ),
        ]

        ordered = self.aggregator._order_chunks_deterministically(unordered_chunks)

        # Should be ordered by category_path, then created_at, then chunk_id
        assert ordered[0].category_path == "A > Category"
        assert ordered[1].category_path == "A > Category"
        assert ordered[2].category_path == "B > Category"

        # Within same path, should be ordered by time
        assert ordered[0].created_at <= ordered[1].created_at

    @pytest.mark.asyncio
    async def test_aggregate_results_within_budget(self) -> None:
        """Test aggregation when results are within budget."""
        # Mock budget manager to return within-budget estimate
        mock_estimate = TokenEstimate(
            fixed_prompt_tokens=100,
            chunks_total_tokens=200,
            total_tokens=300,
            fixed_prompt_chars=400,
            chunks_total_chars=800,
            total_chars=1200,
            within_budget=True,
        )

        with patch.object(
            self.budget_manager, "estimate_budget_usage", return_value=mock_estimate
        ):
            result = await self.aggregator.aggregate_results(
                query_result=self.query_result,
                query_text="Test query",
                contextual_helper="Test context",
            )

        assert result.within_budget is True
        assert len(result.final_chunks) == len(self.chunks)
        assert result.pruning_result is None
        assert result.error_message is None
        assert result.processing_latency_ms > 0

    @pytest.mark.asyncio
    async def test_aggregate_results_over_budget_pruning(self) -> None:
        """Test aggregation when over budget with pruning strategy."""
        # Mock budget manager to return over-budget estimate
        mock_estimate = TokenEstimate(
            fixed_prompt_tokens=100,
            chunks_total_tokens=1200,
            total_tokens=1300,
            fixed_prompt_chars=400,
            chunks_total_chars=4800,
            total_chars=5200,
            within_budget=False,
            tokens_over_budget=300,
        )

        # Mock pruning result
        mock_pruning_result = PruningResult(
            kept_chunks=self.chunks[:2],  # Keep first 2 chunks
            dropped_chunks=self.chunks[2:],  # Drop last chunk
            dropped_paths=["Technology > Web"],
            original_count=3,
            kept_count=2,
            dropped_count=1,
            original_tokens=1200,
            kept_tokens=800,
            dropped_tokens=400,
            strategy_used=PruningStrategy.RANKING_BASED,
            target_tokens=900,
        )

        with (
            patch.object(
                self.budget_manager, "estimate_budget_usage", return_value=mock_estimate
            ),
            patch.object(
                self.pruning_engine, "prune_chunks", return_value=mock_pruning_result
            ),
            patch.object(
                self.budget_manager,
                "validate_final_prompt",
                return_value=(True, 900, "Valid"),
            ),
        ):
            result = await self.aggregator.aggregate_results(
                query_result=self.query_result, query_text="Test query"
            )

        assert result.within_budget is True  # After pruning
        assert len(result.final_chunks) == 2
        assert result.pruning_result == mock_pruning_result
        assert result.dropped_paths == ["Technology > Web"]
        assert result.strategy_used == PromptLimitingStrategy.PRUNE

    @pytest.mark.asyncio
    async def test_aggregate_results_pruning_impossible(self) -> None:
        """Test aggregation when pruning cannot help."""
        # Mock budget manager to return estimate where fixed prompt exceeds budget
        mock_estimate = TokenEstimate(
            fixed_prompt_tokens=1200,  # Exceeds budget
            chunks_total_tokens=500,
            total_tokens=1700,
            fixed_prompt_chars=4800,
            chunks_total_chars=2000,
            total_chars=6800,
            within_budget=False,
            tokens_over_budget=700,
        )

        with patch.object(
            self.budget_manager, "estimate_budget_usage", return_value=mock_estimate
        ):
            result = await self.aggregator.aggregate_results(
                query_result=self.query_result,
                query_text="Very long query that exceeds the budget by itself",
            )

        assert result.within_budget is False
        assert len(result.final_chunks) == 0
        assert "Fixed prompt exceeds budget" in result.error_message
        assert len(result.dropped_paths) == len(
            set(chunk.category_path for chunk in self.chunks)
        )

    @pytest.mark.asyncio
    async def test_aggregate_results_all_chunks_dropped(self) -> None:
        """Test aggregation when all chunks are dropped during pruning."""
        mock_estimate = TokenEstimate(
            fixed_prompt_tokens=100,
            chunks_total_tokens=1200,
            total_tokens=1300,
            fixed_prompt_chars=400,
            chunks_total_chars=4800,
            total_chars=5200,
            within_budget=False,
        )

        # Mock pruning result with no kept chunks
        mock_pruning_result = PruningResult(
            kept_chunks=[],
            dropped_chunks=self.chunks,
            dropped_paths=["Technology > AI", "Technology > Web"],
            original_count=3,
            kept_count=0,
            dropped_count=3,
            original_tokens=1200,
            kept_tokens=0,
            dropped_tokens=1200,
            strategy_used=PruningStrategy.RANKING_BASED,
            target_tokens=900,
        )

        with (
            patch.object(
                self.budget_manager, "estimate_budget_usage", return_value=mock_estimate
            ),
            patch.object(
                self.pruning_engine, "prune_chunks", return_value=mock_pruning_result
            ),
        ):
            result = await self.aggregator.aggregate_results(
                query_result=self.query_result, query_text="Test query"
            )

        assert result.within_budget is False
        assert len(result.final_chunks) == 0
        assert "All chunks were dropped" in result.error_message

    @pytest.mark.asyncio
    async def test_aggregate_results_summarization_strategy(self) -> None:
        """Test aggregation with summarization strategy (not implemented)."""
        # Create config with summarization strategy
        config = BudgetConfig(
            max_token_budget=2000,  # Larger budget to accommodate default headroom
            prompt_limiting_strategy=PromptLimitingStrategy.SUMMARIZE,
        )
        budget_manager = BudgetManager(config=config)
        aggregator = ResultAggregator(budget_manager=budget_manager)

        mock_estimate = TokenEstimate(
            fixed_prompt_tokens=100,
            chunks_total_tokens=1200,
            total_tokens=1300,
            fixed_prompt_chars=400,
            chunks_total_chars=4800,
            total_chars=5200,
            within_budget=False,
        )

        with patch.object(
            budget_manager, "estimate_budget_usage", return_value=mock_estimate
        ):
            result = await aggregator.aggregate_results(
                query_result=self.query_result, query_text="Test query"
            )

        assert result.within_budget is False
        assert "Summarization strategy not yet implemented" in result.error_message

    @pytest.mark.asyncio
    async def test_aggregate_results_error_handling(self) -> None:
        """Test error handling during aggregation."""
        # Mock budget manager to raise an exception
        with patch.object(
            self.budget_manager,
            "estimate_budget_usage",
            side_effect=Exception("Test error"),
        ):
            result = await self.aggregator.aggregate_results(
                query_result=self.query_result, query_text="Test query"
            )

        assert result.within_budget is False
        assert "Aggregation failed: Test error" in result.error_message
        assert result.processing_latency_ms > 0

    def test_construct_final_prompt(self) -> None:
        """Test final prompt construction."""
        prompt = self.aggregator._construct_final_prompt(
            query_text="What is AI?",
            contextual_helper="Focus on technical aspects",
            wrapper_text="Please provide a detailed answer:",
            chunks=self.chunks[:2],
        )

        assert "Query: What is AI?" in prompt
        assert "Context: Focus on technical aspects" in prompt
        assert "Please provide a detailed answer:" in prompt
        assert "Relevant Content:" in prompt
        assert "[1]" in prompt  # Chunk numbering
        assert "[2]" in prompt
        assert self.chunks[0].text_content in prompt
        assert self.chunks[1].text_content in prompt

    def test_construct_final_prompt_minimal(self) -> None:
        """Test final prompt construction with minimal inputs."""
        prompt = self.aggregator._construct_final_prompt(
            query_text="Test query", contextual_helper=None, wrapper_text="", chunks=[]
        )

        assert "Query: Test query" in prompt
        assert "Context:" not in prompt
        assert "Relevant Content:" not in prompt

    def test_analyze_aggregation_requirements(self) -> None:
        """Test aggregation requirements analysis."""
        analysis = self.aggregator.analyze_aggregation_requirements(
            chunks=self.chunks,
            query_text="Test query",
            contextual_helper="Test context",
        )

        assert "token_estimate" in analysis
        assert "budget_config" in analysis
        assert "chunk_analysis" in analysis
        assert "action_required" in analysis

        assert analysis["chunk_analysis"]["total_chunks"] == len(self.chunks)
        assert "unique_paths" in analysis["chunk_analysis"]
        assert "ranking_distribution" in analysis["chunk_analysis"]

    def test_analyze_ranking_distribution(self) -> None:
        """Test ranking distribution analysis."""
        distribution = self.aggregator._analyze_ranking_distribution(self.chunks)

        # Should count occurrences of each ranking
        assert isinstance(distribution, dict)
        assert 5 in distribution  # High relevance
        assert 3 in distribution  # Medium relevance
        assert 1 in distribution  # Low relevance

    def test_get_aggregation_statistics(self) -> None:
        """Test getting aggregation statistics."""
        stats = self.aggregator.get_aggregation_statistics()

        assert "budget_manager" in stats
        assert "pruning_engine" in stats
        assert "supported_strategies" in stats

        assert "prune" in stats["supported_strategies"]
        assert "summarize" in stats["supported_strategies"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_result_aggregator(self) -> None:
        """Test create_result_aggregator function."""
        aggregator = create_result_aggregator(
            max_token_budget=2000,
            strategy=PromptLimitingStrategy.SUMMARIZE,
            model_name="gpt-4o-mini",
            use_rankings=False,
        )

        assert isinstance(aggregator, ResultAggregator)
        assert aggregator.budget_manager.config.max_token_budget == 2000
        assert (
            aggregator.budget_manager.config.prompt_limiting_strategy
            == PromptLimitingStrategy.SUMMARIZE
        )
        assert aggregator.budget_manager.config.model_name == "gpt-4o-mini"
        assert aggregator.budget_manager.config.use_rankings is False

    def test_create_result_aggregator_defaults(self) -> None:
        """Test create_result_aggregator with defaults."""
        aggregator = create_result_aggregator(max_token_budget=1500)

        assert isinstance(aggregator, ResultAggregator)
        assert aggregator.budget_manager.config.max_token_budget == 1500
        assert (
            aggregator.budget_manager.config.prompt_limiting_strategy
            == PromptLimitingStrategy.PRUNE
        )
        assert aggregator.budget_manager.config.use_rankings is True
