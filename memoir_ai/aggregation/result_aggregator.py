"""
Result aggregation system for MemoirAI.

This module provides comprehensive result aggregation with token budget
management, pruning, and preparation for final LLM prompts.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..query.chunk_retrieval import ChunkResult, QueryResult
from ..query.query_strategy_engine import CategoryPath
from .budget_manager import (
    BudgetManager,
    BudgetConfig,
    TokenEstimate,
    PromptLimitingStrategy,
)
from .pruning_engine import PruningEngine, PruningResult, PruningStrategy
from ..exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregating query results with budget management."""

    # Final content
    final_chunks: List[ChunkResult]
    final_prompt_text: str

    # Budget information
    token_estimate: TokenEstimate
    within_budget: bool

    # Processing results
    pruning_result: Optional[PruningResult] = None
    summarization_result: Optional[Dict[str, Any]] = None

    # Metadata
    strategy_used: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE
    dropped_paths: List[str] = field(default_factory=list)
    processing_latency_ms: int = 0

    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ResultAggregator:
    """
    Aggregates query results and manages token budgets.

    Features:
    - Token budget estimation and validation
    - Pruning-based budget management
    - Deterministic chunk ordering
    - Comprehensive error handling
    - Performance tracking
    """

    def __init__(
        self,
        budget_manager: BudgetManager,
        pruning_engine: Optional[PruningEngine] = None,
    ):
        """
        Initialize result aggregator.

        Args:
            budget_manager: Budget manager for token counting and validation
            pruning_engine: Optional pruning engine (will create default if None)
        """
        self.budget_manager = budget_manager
        self.pruning_engine = pruning_engine or PruningEngine(
            token_counter_func=budget_manager.count_tokens
        )

    async def aggregate_results(
        self,
        query_result: QueryResult,
        query_text: str,
        contextual_helper: Optional[str] = None,
        wrapper_text: str = "",
    ) -> AggregationResult:
        """
        Aggregate query results with budget management.

        Args:
            query_result: Query result with chunks and metadata
            query_text: Original query text
            contextual_helper: Optional contextual information
            wrapper_text: Additional wrapper text for prompt

        Returns:
            Aggregation result with budget-managed content
        """
        start_time = datetime.now()

        logger.info(
            f"Aggregating {len(query_result.chunks)} chunks with "
            f"{self.budget_manager.config.prompt_limiting_strategy.value} strategy"
        )

        try:
            # Step 1: Order chunks deterministically
            ordered_chunks = self._order_chunks_deterministically(query_result.chunks)

            # Step 2: Estimate token usage
            chunk_texts = [chunk.text_content for chunk in ordered_chunks]
            token_estimate = self.budget_manager.estimate_budget_usage(
                query_text=query_text,
                contextual_helper=contextual_helper,
                wrapper_text=wrapper_text,
                chunks_text=chunk_texts,
            )

            logger.info(
                f"Token estimate: {token_estimate.total_tokens}/{self.budget_manager.config.max_token_budget} "
                f"(within budget: {token_estimate.within_budget})"
            )

            # Step 3: Apply budget management if needed
            if token_estimate.within_budget:
                # No budget management needed
                final_chunks = ordered_chunks
                final_prompt = self._construct_final_prompt(
                    query_text, contextual_helper, wrapper_text, final_chunks
                )

                result = AggregationResult(
                    final_chunks=final_chunks,
                    final_prompt_text=final_prompt,
                    token_estimate=token_estimate,
                    within_budget=True,
                    strategy_used=self.budget_manager.config.prompt_limiting_strategy,
                )
            else:
                # Budget exceeded - apply strategy
                if (
                    self.budget_manager.config.prompt_limiting_strategy
                    == PromptLimitingStrategy.PRUNE
                ):
                    result = await self._apply_pruning_strategy(
                        ordered_chunks,
                        token_estimate,
                        query_text,
                        contextual_helper,
                        wrapper_text,
                    )
                else:
                    # Summarization not implemented in this task
                    result = AggregationResult(
                        final_chunks=[],
                        final_prompt_text="",
                        token_estimate=token_estimate,
                        within_budget=False,
                        error_message="Summarization strategy not yet implemented",
                    )

            # Step 4: Calculate processing time
            end_time = datetime.now()
            result.processing_latency_ms = max(
                1, int((end_time - start_time).total_seconds() * 1000)
            )

            logger.info(
                f"Aggregation completed: {len(result.final_chunks)} final chunks, "
                f"{result.processing_latency_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Error during result aggregation: {e}")

            # Calculate processing time even for errors
            end_time = datetime.now()
            processing_time = max(
                1, int((end_time - start_time).total_seconds() * 1000)
            )

            return AggregationResult(
                final_chunks=[],
                final_prompt_text="",
                token_estimate=TokenEstimate(0, 0, 0, 0, 0, 0, False),
                within_budget=False,
                error_message=f"Aggregation failed: {str(e)}",
                processing_latency_ms=processing_time,
            )

    def _order_chunks_deterministically(
        self, chunks: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Order chunks deterministically according to requirements.

        Order by:
        1. Category path (for path-level ordering)
        2. created_at ascending
        3. chunk_id ascending
        """
        return sorted(chunks, key=lambda c: (c.category_path, c.created_at, c.chunk_id))

    async def _apply_pruning_strategy(
        self,
        chunks: List[ChunkResult],
        token_estimate: TokenEstimate,
        query_text: str,
        contextual_helper: Optional[str],
        wrapper_text: str,
    ) -> AggregationResult:
        """Apply pruning strategy to fit within budget."""

        # Calculate available tokens for chunks
        effective_budget = self._effective_budget_from_estimate(token_estimate)
        available_tokens = effective_budget - token_estimate.fixed_prompt_tokens

        if available_tokens <= 0:
            return AggregationResult(
                final_chunks=[],
                final_prompt_text="",
                token_estimate=token_estimate,
                within_budget=False,
                error_message="Fixed prompt exceeds budget, pruning cannot help",
                dropped_paths=list(set(chunk.category_path for chunk in chunks)),
            )

        # Determine pruning strategy
        use_rankings = self.budget_manager.config.use_rankings
        strategy = (
            PruningStrategy.RANKING_BASED
            if use_rankings
            else PruningStrategy.DETERMINISTIC_ORDER
        )

        # Perform pruning
        pruning_result = self.pruning_engine.prune_chunks(
            chunks=chunks,
            target_tokens=max(0, available_tokens),
            strategy=strategy,
            use_rankings=use_rankings,
        )

        logger.info(
            f"Pruning completed: kept {pruning_result.kept_count}/{pruning_result.original_count} chunks, "
            f"dropped {len(pruning_result.dropped_paths)} paths"
        )

        # Check if all chunks were dropped
        if not pruning_result.kept_chunks:
            return AggregationResult(
                final_chunks=[],
                final_prompt_text="",
                token_estimate=token_estimate,
                within_budget=False,
                pruning_result=pruning_result,
                error_message="All chunks were dropped during pruning, budget too restrictive",
                dropped_paths=pruning_result.dropped_paths,
            )

        # Construct final prompt with pruned chunks
        final_prompt = self._construct_final_prompt(
            query_text, contextual_helper, wrapper_text, pruning_result.kept_chunks
        )

        # Validate final prompt is within budget
        is_valid, final_token_count, validation_message = (
            self.budget_manager.validate_final_prompt(final_prompt)
        )

        if not is_valid:
            return AggregationResult(
                final_chunks=pruning_result.kept_chunks,
                final_prompt_text=final_prompt,
                token_estimate=token_estimate,
                within_budget=False,
                pruning_result=pruning_result,
                error_message=f"Final prompt still exceeds budget after pruning: {validation_message}",
                dropped_paths=pruning_result.dropped_paths,
            )

        # Success - create updated token estimate
        updated_estimate = TokenEstimate(
            fixed_prompt_tokens=token_estimate.fixed_prompt_tokens,
            chunks_total_tokens=pruning_result.kept_tokens,
            total_tokens=final_token_count,
            fixed_prompt_chars=token_estimate.fixed_prompt_chars,
            chunks_total_chars=sum(
                len(chunk.text_content) for chunk in pruning_result.kept_chunks
            ),
            total_chars=len(final_prompt),
            within_budget=True,
        )

        return AggregationResult(
            final_chunks=pruning_result.kept_chunks,
            final_prompt_text=final_prompt,
            token_estimate=updated_estimate,
            within_budget=True,
            pruning_result=pruning_result,
            strategy_used=PromptLimitingStrategy.PRUNE,
            dropped_paths=pruning_result.dropped_paths,
        )

    def _effective_budget_from_estimate(self, estimate: TokenEstimate) -> int:
        """Derive effective budget using estimate data when available."""
        if estimate.tokens_over_budget and estimate.tokens_over_budget > 0:
            return max(0, estimate.total_tokens - estimate.tokens_over_budget)
        return self.budget_manager.config.max_token_budget

    def _construct_final_prompt(
        self,
        query_text: str,
        contextual_helper: Optional[str],
        wrapper_text: str,
        chunks: List[ChunkResult],
    ) -> str:
        """
        Construct the final prompt text.

        Args:
            query_text: User's query
            contextual_helper: Optional context
            wrapper_text: Additional wrapper text
            chunks: Final chunks to include

        Returns:
            Complete final prompt text
        """
        prompt_parts = []

        # Add query
        if query_text:
            prompt_parts.append(f"Query: {query_text}")

        # Add contextual helper
        if contextual_helper:
            prompt_parts.append(f"Context: {contextual_helper}")

        # Add wrapper text
        if wrapper_text:
            prompt_parts.append(wrapper_text)

        # Add chunks
        if chunks:
            prompt_parts.append("Relevant Content:")
            for i, chunk in enumerate(chunks, 1):
                chunk_text = f"[{i}] {chunk.text_content}"
                if chunk.category_path:
                    chunk_text += f" (from: {chunk.category_path})"
                prompt_parts.append(chunk_text)

        return "\n\n".join(prompt_parts)

    def analyze_aggregation_requirements(
        self,
        chunks: List[ChunkResult],
        query_text: str,
        contextual_helper: Optional[str] = None,
        wrapper_text: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze aggregation requirements without performing aggregation.

        Args:
            chunks: Chunks to analyze
            query_text: Query text
            contextual_helper: Optional context
            wrapper_text: Wrapper text

        Returns:
            Dictionary with analysis results
        """
        # Estimate token usage
        chunk_texts = [chunk.text_content for chunk in chunks]
        token_estimate = self.budget_manager.estimate_budget_usage(
            query_text=query_text,
            contextual_helper=contextual_helper,
            wrapper_text=wrapper_text,
            chunks_text=chunk_texts,
        )

        # Analyze pruning impact if needed
        pruning_analysis = None
        if not token_estimate.within_budget:
            available_tokens = (
                self.budget_manager.config.max_token_budget
                - token_estimate.fixed_prompt_tokens
            )

            if available_tokens > 0:
                use_rankings = self.budget_manager.config.use_rankings
                strategy = (
                    PruningStrategy.RANKING_BASED
                    if use_rankings
                    else PruningStrategy.DETERMINISTIC_ORDER
                )

                pruning_analysis = self.pruning_engine.analyze_pruning_impact(
                    chunks=chunks, target_tokens=available_tokens, strategy=strategy
                )

        return {
            "token_estimate": {
                "fixed_prompt_tokens": token_estimate.fixed_prompt_tokens,
                "chunks_total_tokens": token_estimate.chunks_total_tokens,
                "total_tokens": token_estimate.total_tokens,
                "within_budget": token_estimate.within_budget,
                "tokens_over_budget": token_estimate.tokens_over_budget,
            },
            "budget_config": {
                "max_token_budget": self.budget_manager.config.max_token_budget,
                "strategy": self.budget_manager.config.prompt_limiting_strategy.value,
                "use_rankings": self.budget_manager.config.use_rankings,
            },
            "chunk_analysis": {
                "total_chunks": len(chunks),
            "unique_paths": len(set(chunk.category_path for chunk in chunks)),
                "ranking_distribution": self._analyze_ranking_distribution(chunks),
            },
            "pruning_analysis": pruning_analysis,
            "action_required": not token_estimate.within_budget,
        }

    def _analyze_ranking_distribution(
        self, chunks: List[ChunkResult]
    ) -> Dict[int, int]:
        """Analyze the distribution of ranking scores in chunks."""
        distribution = {}
        for chunk in chunks:
            rank = chunk.ranked_relevance
            distribution[rank] = distribution.get(rank, 0) + 1
        return distribution

    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """
        Get aggregation system statistics.

        Returns:
            Dictionary with system statistics
        """
        return {
            "budget_manager": self.budget_manager.get_budget_statistics(),
            "pruning_engine": {
                "preserve_path_diversity": self.pruning_engine.preserve_path_diversity,
                "available_strategies": [
                    strategy.value for strategy in PruningStrategy
                ],
            },
            "supported_strategies": [
                strategy.value for strategy in PromptLimitingStrategy
            ],
        }


# Utility functions
def create_result_aggregator(
    max_token_budget: int,
    strategy: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE,
    model_name: str = "gpt-4",
    use_rankings: bool = True,
    **kwargs,
) -> ResultAggregator:
    """
    Create a result aggregator with default configuration.

    Args:
        max_token_budget: Maximum token budget
        strategy: Prompt limiting strategy
        model_name: Model name for token counting
        use_rankings: Whether to use ranking-based pruning
        **kwargs: Additional configuration options

    Returns:
        Configured ResultAggregator
    """
    from .budget_manager import create_budget_manager
    from .pruning_engine import create_pruning_engine

    budget_manager = create_budget_manager(
        max_token_budget=max_token_budget,
        strategy=strategy,
        model_name=model_name,
        use_rankings=use_rankings,
        **kwargs,
    )

    pruning_engine = create_pruning_engine(
        token_counter_func=budget_manager.count_tokens
    )

    return ResultAggregator(
        budget_manager=budget_manager, pruning_engine=pruning_engine
    )
