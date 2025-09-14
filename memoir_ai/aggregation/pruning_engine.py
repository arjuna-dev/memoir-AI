"""
Pruning engine for MemoirAI token budget management.

This module provides ranking-based and deterministic order pruning
strategies to fit content within token budgets.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..query.chunk_retrieval import ChunkResult
from ..exceptions import ValidationError

logger = logging.getLogger(__name__)


class PruningStrategy(Enum):
    """Available pruning strategies."""

    RANKING_BASED = "ranking_based"
    DETERMINISTIC_ORDER = "deterministic_order"


@dataclass
class PruningResult:
    """Result of pruning operation."""

    # Results
    kept_chunks: List[ChunkResult]
    dropped_chunks: List[ChunkResult]
    dropped_paths: List[str]

    # Metrics
    original_count: int
    kept_count: int
    dropped_count: int

    # Token information
    original_tokens: int
    kept_tokens: int
    dropped_tokens: int

    # Strategy used
    strategy_used: PruningStrategy
    target_tokens: int

    @property
    def pruning_ratio(self) -> float:
        """Calculate the ratio of content that was pruned."""
        if self.original_count == 0:
            return 0.0
        return self.dropped_count / self.original_count

    @property
    def token_reduction_ratio(self) -> float:
        """Calculate the ratio of tokens that were reduced."""
        if self.original_tokens == 0:
            return 0.0
        return self.dropped_tokens / self.original_tokens


class PruningEngine:
    """
    Engine for pruning chunks to fit within token budgets.

    Features:
    - Ranking-based pruning using relevance scores
    - Deterministic order pruning for consistent results
    - Path-level tracking for dropped content
    - Comprehensive metrics and reporting
    """

    def __init__(
        self,
        token_counter_func: Optional[callable] = None,
        preserve_path_diversity: bool = True,
    ):
        """
        Initialize pruning engine.

        Args:
            token_counter_func: Function to count tokens in text
            preserve_path_diversity: Whether to preserve chunks from different paths
        """
        self.token_counter_func = token_counter_func or self._default_token_counter
        self.preserve_path_diversity = preserve_path_diversity

    def _default_token_counter(self, text: str) -> int:
        """Default token counter using character-based estimation."""
        return max(1, len(text) // 4)  # Rough estimate: 4 chars per token

    def prune_chunks(
        self,
        chunks: List[ChunkResult],
        target_tokens: int,
        strategy: PruningStrategy = PruningStrategy.RANKING_BASED,
        use_rankings: bool = True,
    ) -> PruningResult:
        """
        Prune chunks to fit within target token budget.

        Args:
            chunks: List of chunks to prune
            target_tokens: Target token count to achieve
            strategy: Pruning strategy to use
            use_rankings: Whether to use ranking information

        Returns:
            Pruning result with kept and dropped chunks
        """
        if not chunks:
            return self._create_empty_result(target_tokens, strategy)

        # Calculate original metrics
        original_tokens = sum(
            self.token_counter_func(chunk.text_content) for chunk in chunks
        )

        # If already within budget, no pruning needed
        if original_tokens <= target_tokens:
            return PruningResult(
                kept_chunks=chunks.copy(),
                dropped_chunks=[],
                dropped_paths=[],
                original_count=len(chunks),
                kept_count=len(chunks),
                dropped_count=0,
                original_tokens=original_tokens,
                kept_tokens=original_tokens,
                dropped_tokens=0,
                strategy_used=strategy,
                target_tokens=target_tokens,
            )

        # Perform pruning based on strategy
        if strategy == PruningStrategy.RANKING_BASED and use_rankings:
            return self._prune_by_rankings(chunks, target_tokens, original_tokens)
        else:
            return self._prune_by_deterministic_order(
                chunks, target_tokens, original_tokens
            )

    def _prune_by_rankings(
        self,
        chunks: List[ChunkResult],
        target_tokens: int,
        original_tokens: int,
    ) -> PruningResult:
        """
        Prune chunks using ranking-based strategy.

        Keeps highest-ranked chunks first, dropping lowest-ranked chunks
        until the target token count is achieved.
        """
        # Sort chunks by ranked_relevance (descending) then by deterministic order
        sorted_chunks = sorted(
            chunks, key=lambda c: (-c.ranked_relevance, c.created_at, c.chunk_id)
        )

        kept_chunks = []
        current_tokens = 0

        # Add chunks in ranking order until budget is reached
        for chunk in sorted_chunks:
            chunk_tokens = self.token_counter_func(chunk.text_content)

            if current_tokens + chunk_tokens <= target_tokens:
                kept_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Check if we should try to fit this chunk by dropping others
                if self.preserve_path_diversity:
                    # Try to preserve path diversity by checking if this chunk
                    # is from a path not yet represented
                    existing_paths = {chunk.category_path for chunk in kept_chunks}
                    if chunk.category_path not in existing_paths:
                        # This chunk is from a new path, try to make room
                        if self._try_make_room_for_chunk(
                            kept_chunks, chunk, target_tokens
                        ):
                            kept_chunks.append(chunk)
                            current_tokens = sum(
                                self.token_counter_func(c.text_content)
                                for c in kept_chunks
                            )
                            continue

                # Can't fit this chunk, stop here
                break

        # Calculate dropped chunks and paths
        kept_chunk_ids = {chunk.chunk_id for chunk in kept_chunks}
        dropped_chunks = [
            chunk for chunk in chunks if chunk.chunk_id not in kept_chunk_ids
        ]

        dropped_paths = list(set(chunk.category_path for chunk in dropped_chunks))
        kept_tokens = sum(
            self.token_counter_func(chunk.text_content) for chunk in kept_chunks
        )
        dropped_tokens = original_tokens - kept_tokens

        return PruningResult(
            kept_chunks=kept_chunks,
            dropped_chunks=dropped_chunks,
            dropped_paths=dropped_paths,
            original_count=len(chunks),
            kept_count=len(kept_chunks),
            dropped_count=len(dropped_chunks),
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            dropped_tokens=dropped_tokens,
            strategy_used=PruningStrategy.RANKING_BASED,
            target_tokens=target_tokens,
        )

    def _prune_by_deterministic_order(
        self,
        chunks: List[ChunkResult],
        target_tokens: int,
        original_tokens: int,
    ) -> PruningResult:
        """
        Prune chunks using deterministic order strategy.

        Keeps chunks in their original deterministic order (created_at, chunk_id)
        until the target token count is achieved.
        """
        # Sort chunks by deterministic order
        sorted_chunks = sorted(chunks, key=lambda c: (c.created_at, c.chunk_id))

        kept_chunks = []
        current_tokens = 0

        # Add chunks in deterministic order until budget is reached
        for chunk in sorted_chunks:
            chunk_tokens = self.token_counter_func(chunk.text_content)

            if current_tokens + chunk_tokens <= target_tokens:
                kept_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Can't fit this chunk, stop here
                break

        # Calculate dropped chunks and paths
        kept_chunk_ids = {chunk.chunk_id for chunk in kept_chunks}
        dropped_chunks = [
            chunk for chunk in chunks if chunk.chunk_id not in kept_chunk_ids
        ]

        dropped_paths = list(set(chunk.category_path for chunk in dropped_chunks))
        kept_tokens = sum(
            self.token_counter_func(chunk.text_content) for chunk in kept_chunks
        )
        dropped_tokens = original_tokens - kept_tokens

        return PruningResult(
            kept_chunks=kept_chunks,
            dropped_chunks=dropped_chunks,
            dropped_paths=dropped_paths,
            original_count=len(chunks),
            kept_count=len(kept_chunks),
            dropped_count=len(dropped_chunks),
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            dropped_tokens=dropped_tokens,
            strategy_used=PruningStrategy.DETERMINISTIC_ORDER,
            target_tokens=target_tokens,
        )

    def _try_make_room_for_chunk(
        self,
        kept_chunks: List[ChunkResult],
        new_chunk: ChunkResult,
        target_tokens: int,
    ) -> bool:
        """
        Try to make room for a new chunk by dropping lower-ranked chunks.

        Args:
            kept_chunks: Currently kept chunks (will be modified)
            new_chunk: New chunk to try to fit
            target_tokens: Target token budget

        Returns:
            True if room was made, False otherwise
        """
        new_chunk_tokens = self.token_counter_func(new_chunk.text_content)
        current_tokens = sum(
            self.token_counter_func(c.text_content) for c in kept_chunks
        )

        if current_tokens + new_chunk_tokens <= target_tokens:
            return True  # Already fits

        # Sort kept chunks by ranking (ascending) to drop lowest-ranked first
        kept_chunks.sort(
            key=lambda c: (c.ranked_relevance, -c.created_at.timestamp(), -c.chunk_id)
        )

        tokens_to_free = (current_tokens + new_chunk_tokens) - target_tokens
        freed_tokens = 0

        # Try to drop chunks to make room
        chunks_to_remove = []
        for chunk in kept_chunks:
            if freed_tokens >= tokens_to_free:
                break

            # Don't drop chunks with higher or equal ranking
            if chunk.ranked_relevance >= new_chunk.ranked_relevance:
                continue

            chunk_tokens = self.token_counter_func(chunk.text_content)
            chunks_to_remove.append(chunk)
            freed_tokens += chunk_tokens

        # Remove the chunks if we freed enough space
        if freed_tokens >= tokens_to_free:
            for chunk in chunks_to_remove:
                kept_chunks.remove(chunk)
            return True

        return False

    def _create_empty_result(
        self,
        target_tokens: int,
        strategy: PruningStrategy,
    ) -> PruningResult:
        """Create an empty pruning result."""
        return PruningResult(
            kept_chunks=[],
            dropped_chunks=[],
            dropped_paths=[],
            original_count=0,
            kept_count=0,
            dropped_count=0,
            original_tokens=0,
            kept_tokens=0,
            dropped_tokens=0,
            strategy_used=strategy,
            target_tokens=target_tokens,
        )

    def analyze_pruning_impact(
        self,
        chunks: List[ChunkResult],
        target_tokens: int,
        strategy: PruningStrategy = PruningStrategy.RANKING_BASED,
    ) -> Dict[str, Any]:
        """
        Analyze the impact of pruning without actually performing it.

        Args:
            chunks: Chunks to analyze
            target_tokens: Target token budget
            strategy: Pruning strategy to simulate

        Returns:
            Dictionary with pruning impact analysis
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "estimated_kept_chunks": 0,
                "estimated_kept_tokens": 0,
                "estimated_pruning_ratio": 0.0,
                "paths_affected": [],
                "ranking_distribution": {},
            }

        total_tokens = sum(
            self.token_counter_func(chunk.text_content) for chunk in chunks
        )

        # Simulate pruning to get estimates
        if strategy == PruningStrategy.RANKING_BASED:
            sorted_chunks = sorted(
                chunks, key=lambda c: (-c.ranked_relevance, c.created_at, c.chunk_id)
            )
        else:
            sorted_chunks = sorted(chunks, key=lambda c: (c.created_at, c.chunk_id))

        kept_tokens = 0
        kept_count = 0

        for chunk in sorted_chunks:
            chunk_tokens = self.token_counter_func(chunk.text_content)
            if kept_tokens + chunk_tokens <= target_tokens:
                kept_tokens += chunk_tokens
                kept_count += 1
            else:
                break

        # Analyze ranking distribution
        ranking_dist = {}
        for chunk in chunks:
            rank = chunk.ranked_relevance
            if rank not in ranking_dist:
                ranking_dist[rank] = 0
            ranking_dist[rank] += 1

        # Identify affected paths
        paths_affected = list(set(chunk.category_path for chunk in chunks[kept_count:]))

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "estimated_kept_chunks": kept_count,
            "estimated_kept_tokens": kept_tokens,
            "estimated_pruning_ratio": (len(chunks) - kept_count) / len(chunks),
            "estimated_token_reduction": (total_tokens - kept_tokens) / total_tokens,
            "paths_affected": paths_affected,
            "ranking_distribution": ranking_dist,
            "strategy": strategy.value,
            "target_tokens": target_tokens,
        }


# Utility functions
def create_pruning_engine(
    token_counter_func: Optional[callable] = None,
    preserve_path_diversity: bool = True,
    **kwargs,
) -> PruningEngine:
    """
    Create a pruning engine with default configuration.

    Args:
        token_counter_func: Function to count tokens
        preserve_path_diversity: Whether to preserve path diversity
        **kwargs: Additional configuration options

    Returns:
        Configured PruningEngine
    """
    return PruningEngine(
        token_counter_func=token_counter_func,
        preserve_path_diversity=preserve_path_diversity,
        **kwargs,
    )


def prune_chunks_simple(
    chunks: List[ChunkResult],
    target_tokens: int,
    use_rankings: bool = True,
    token_counter_func: Optional[callable] = None,
) -> Tuple[List[ChunkResult], List[str]]:
    """
    Simple utility function for pruning chunks.

    Args:
        chunks: Chunks to prune
        target_tokens: Target token count
        use_rankings: Whether to use ranking-based pruning
        token_counter_func: Optional token counter function

    Returns:
        Tuple of (kept_chunks, dropped_paths)
    """
    engine = create_pruning_engine(token_counter_func=token_counter_func)

    strategy = (
        PruningStrategy.RANKING_BASED
        if use_rankings
        else PruningStrategy.DETERMINISTIC_ORDER
    )

    result = engine.prune_chunks(
        chunks=chunks,
        target_tokens=target_tokens,
        strategy=strategy,
        use_rankings=use_rankings,
    )

    return result.kept_chunks, result.dropped_paths
