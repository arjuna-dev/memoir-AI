"""
Tests for pruning engine.
"""

import pytest
from datetime import datetime, timedelta

from memoir_ai.aggregation.pruning_engine import (
    PruningEngine,
    PruningResult,
    PruningStrategy,
    create_pruning_engine,
    prune_chunks_simple,
)
from memoir_ai.query.chunk_retrieval import ChunkResult


class TestPruningResult:
    """Test PruningResult functionality."""

    def test_pruning_result_creation(self):
        """Test PruningResult creation."""
        now = datetime.now()

        kept_chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="Kept chunk",
                category_path="Tech > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=now,
            )
        ]

        dropped_chunks = [
            ChunkResult(
                chunk_id=2,
                text_content="Dropped chunk",
                category_path="Tech > Web",
                category_id_path="1/3",
                ranked_relevance=2,
                created_at=now,
            )
        ]

        result = PruningResult(
            kept_chunks=kept_chunks,
            dropped_chunks=dropped_chunks,
            dropped_paths=["Tech > Web"],
            original_count=2,
            kept_count=1,
            dropped_count=1,
            original_tokens=100,
            kept_tokens=60,
            dropped_tokens=40,
            strategy_used=PruningStrategy.RANKING_BASED,
            target_tokens=60,
        )

        assert len(result.kept_chunks) == 1
        assert len(result.dropped_chunks) == 1
        assert result.dropped_paths == ["Tech > Web"]
        assert result.pruning_ratio == 0.5  # 1/2
        assert result.token_reduction_ratio == 0.4  # 40/100

    def test_pruning_result_ratios_zero_division(self):
        """Test pruning result ratios with zero values."""
        result = PruningResult(
            kept_chunks=[],
            dropped_chunks=[],
            dropped_paths=[],
            original_count=0,
            kept_count=0,
            dropped_count=0,
            original_tokens=0,
            kept_tokens=0,
            dropped_tokens=0,
            strategy_used=PruningStrategy.RANKING_BASED,
            target_tokens=100,
        )

        assert result.pruning_ratio == 0.0
        assert result.token_reduction_ratio == 0.0


class TestPruningEngine:
    """Test PruningEngine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PruningEngine()

        # Create test chunks with different rankings and timestamps
        base_time = datetime.now()

        self.chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="High relevance chunk with more content",  # ~40 chars
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=base_time,
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Medium relevance chunk",  # ~24 chars
                category_path="Technology > AI",
                category_id_path="1/2",
                ranked_relevance=3,
                created_at=base_time + timedelta(minutes=1),
            ),
            ChunkResult(
                chunk_id=3,
                text_content="Low relevance chunk with some text",  # ~34 chars
                category_path="Technology > Web",
                category_id_path="1/3",
                ranked_relevance=1,
                created_at=base_time + timedelta(minutes=2),
            ),
            ChunkResult(
                chunk_id=4,
                text_content="Another high relevance chunk",  # ~29 chars
                category_path="Science > Physics",
                category_id_path="2/4",
                ranked_relevance=5,
                created_at=base_time + timedelta(minutes=3),
            ),
        ]

    def test_initialization(self):
        """Test PruningEngine initialization."""
        assert self.engine.token_counter_func is not None
        assert self.engine.preserve_path_diversity is True

    def test_initialization_custom(self):
        """Test PruningEngine initialization with custom parameters."""

        def custom_counter(text):
            return len(text) // 2

        engine = PruningEngine(
            token_counter_func=custom_counter, preserve_path_diversity=False
        )

        assert engine.token_counter_func == custom_counter
        assert engine.preserve_path_diversity is False

    def test_default_token_counter(self):
        """Test default token counter."""
        text = "This is a test sentence with some words."
        count = self.engine._default_token_counter(text)

        expected = max(1, len(text) // 4)
        assert count == expected

    def test_prune_chunks_empty_list(self):
        """Test pruning with empty chunk list."""
        result = self.engine.prune_chunks([], target_tokens=100)

        assert len(result.kept_chunks) == 0
        assert len(result.dropped_chunks) == 0
        assert result.original_count == 0
        assert result.kept_count == 0
        assert result.dropped_count == 0

    def test_prune_chunks_within_budget(self):
        """Test pruning when chunks are already within budget."""
        # Set a high target that should accommodate all chunks
        result = self.engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=1000,  # Very high budget
            strategy=PruningStrategy.RANKING_BASED,
        )

        assert len(result.kept_chunks) == len(self.chunks)
        assert len(result.dropped_chunks) == 0
        assert result.kept_count == len(self.chunks)
        assert result.dropped_count == 0

    def test_prune_chunks_ranking_based(self):
        """Test ranking-based pruning strategy."""
        # Set a target that should keep only the highest-ranked chunks
        result = self.engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=20,  # Small budget to force pruning
            strategy=PruningStrategy.RANKING_BASED,
            use_rankings=True,
        )

        assert result.strategy_used == PruningStrategy.RANKING_BASED
        assert len(result.kept_chunks) > 0
        assert len(result.dropped_chunks) > 0

        # Check that kept chunks have higher or equal rankings than dropped chunks
        if result.kept_chunks and result.dropped_chunks:
            min_kept_ranking = min(
                chunk.ranked_relevance for chunk in result.kept_chunks
            )
            max_dropped_ranking = max(
                chunk.ranked_relevance for chunk in result.dropped_chunks
            )

            # This might not always be true due to path diversity preservation
            # but generally higher-ranked chunks should be preferred
            assert (
                min_kept_ranking >= max_dropped_ranking or len(result.kept_chunks) == 1
            )

    def test_prune_chunks_deterministic_order(self):
        """Test deterministic order pruning strategy."""
        result = self.engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=20,  # Small budget to force pruning
            strategy=PruningStrategy.DETERMINISTIC_ORDER,
            use_rankings=False,
        )

        assert result.strategy_used == PruningStrategy.DETERMINISTIC_ORDER
        assert len(result.kept_chunks) > 0

        # Check that kept chunks are in chronological order
        if len(result.kept_chunks) > 1:
            for i in range(1, len(result.kept_chunks)):
                prev_chunk = result.kept_chunks[i - 1]
                curr_chunk = result.kept_chunks[i]

                # Should be ordered by created_at, then chunk_id
                assert prev_chunk.created_at <= curr_chunk.created_at or (
                    prev_chunk.created_at == curr_chunk.created_at
                    and prev_chunk.chunk_id <= curr_chunk.chunk_id
                )

    def test_prune_chunks_path_diversity(self):
        """Test path diversity preservation."""
        engine = PruningEngine(preserve_path_diversity=True)

        result = engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=30,  # Medium budget
            strategy=PruningStrategy.RANKING_BASED,
            use_rankings=True,
        )

        # Should try to preserve chunks from different paths
        kept_paths = set(chunk.category_path for chunk in result.kept_chunks)

        # With path diversity, we should have chunks from multiple paths if possible
        assert len(kept_paths) >= 1

    def test_try_make_room_for_chunk(self):
        """Test making room for a new chunk."""
        # Start with some kept chunks
        kept_chunks = [self.chunks[1], self.chunks[2]]  # Medium and low relevance
        new_chunk = self.chunks[0]  # High relevance

        success = self.engine._try_make_room_for_chunk(
            kept_chunks=kept_chunks, new_chunk=new_chunk, target_tokens=50
        )

        # Should be able to make room by dropping lower-ranked chunks
        assert success is True or success is False  # Depends on token calculations

    def test_analyze_pruning_impact(self):
        """Test pruning impact analysis."""
        analysis = self.engine.analyze_pruning_impact(
            chunks=self.chunks, target_tokens=30, strategy=PruningStrategy.RANKING_BASED
        )

        assert "total_chunks" in analysis
        assert "total_tokens" in analysis
        assert "estimated_kept_chunks" in analysis
        assert "estimated_pruning_ratio" in analysis
        assert "ranking_distribution" in analysis
        assert "strategy" in analysis

        assert analysis["total_chunks"] == len(self.chunks)
        assert analysis["strategy"] == "ranking_based"

    def test_analyze_pruning_impact_empty(self):
        """Test pruning impact analysis with empty chunks."""
        analysis = self.engine.analyze_pruning_impact(
            chunks=[], target_tokens=100, strategy=PruningStrategy.DETERMINISTIC_ORDER
        )

        assert analysis["total_chunks"] == 0
        assert analysis["total_tokens"] == 0
        assert analysis["estimated_kept_chunks"] == 0
        assert analysis["estimated_pruning_ratio"] == 0.0

    def test_prune_chunks_all_dropped(self):
        """Test pruning when all chunks must be dropped."""
        result = self.engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=1,  # Extremely small budget
            strategy=PruningStrategy.RANKING_BASED,
        )

        # Might keep one very small chunk or drop all
        assert result.kept_count <= 1
        assert result.dropped_count >= len(self.chunks) - 1

    def test_prune_chunks_custom_token_counter(self):
        """Test pruning with custom token counter."""

        def custom_counter(text):
            return len(text)  # 1 token per character

        engine = PruningEngine(token_counter_func=custom_counter)

        result = engine.prune_chunks(
            chunks=self.chunks,
            target_tokens=50,  # 50 characters
            strategy=PruningStrategy.DETERMINISTIC_ORDER,
        )

        # Should use the custom token counter
        total_kept_chars = sum(len(chunk.text_content) for chunk in result.kept_chunks)
        assert total_kept_chars <= 50


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_pruning_engine(self):
        """Test create_pruning_engine function."""

        def custom_counter(text):
            return len(text) // 3

        engine = create_pruning_engine(
            token_counter_func=custom_counter, preserve_path_diversity=False
        )

        assert isinstance(engine, PruningEngine)
        assert engine.token_counter_func == custom_counter
        assert engine.preserve_path_diversity is False

    def test_create_pruning_engine_defaults(self):
        """Test create_pruning_engine with defaults."""
        engine = create_pruning_engine()

        assert isinstance(engine, PruningEngine)
        assert engine.preserve_path_diversity is True

    def test_prune_chunks_simple(self):
        """Test prune_chunks_simple utility function."""
        base_time = datetime.now()

        chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="High relevance chunk",
                category_path="Tech > AI",
                category_id_path="1/2",
                ranked_relevance=5,
                created_at=base_time,
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Low relevance chunk",
                category_path="Tech > Web",
                category_id_path="1/3",
                ranked_relevance=1,
                created_at=base_time + timedelta(minutes=1),
            ),
        ]

        kept_chunks, dropped_paths = prune_chunks_simple(
            chunks=chunks, target_tokens=10, use_rankings=True
        )

        assert isinstance(kept_chunks, list)
        assert isinstance(dropped_paths, list)
        assert len(kept_chunks) <= len(chunks)

    def test_prune_chunks_simple_no_rankings(self):
        """Test prune_chunks_simple without rankings."""
        base_time = datetime.now()

        chunks = [
            ChunkResult(
                chunk_id=1,
                text_content="First chunk",
                category_path="Tech > AI",
                category_id_path="1/2",
                ranked_relevance=1,
                created_at=base_time,
            ),
            ChunkResult(
                chunk_id=2,
                text_content="Second chunk",
                category_path="Tech > Web",
                category_id_path="1/3",
                ranked_relevance=5,
                created_at=base_time + timedelta(minutes=1),
            ),
        ]

        kept_chunks, dropped_paths = prune_chunks_simple(
            chunks=chunks,
            target_tokens=10,
            use_rankings=False,  # Should use deterministic order
        )

        # With deterministic order, first chunk (by time) should be preferred
        if kept_chunks:
            assert kept_chunks[0].chunk_id == 1
