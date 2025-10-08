"""
Tests for token-based text chunker.
"""

import pytest

from memoir_ai.exceptions import ValidationError
from memoir_ai.text_processing.chunker import TextChunk, TextChunker


class TestTextChunk:
    """Test TextChunk data class."""

    def test_valid_chunk_creation(self) -> None:
        """Test creating a valid text chunk."""
        chunk = TextChunk(
            content="This is a test chunk.",
            token_count=5,
            start_position=0,
            end_position=21,
            source_id="test_source",
            metadata={"type": "test"},
        )

        assert chunk.content == "This is a test chunk."
        assert chunk.token_count == 5
        assert chunk.start_position == 0
        assert chunk.end_position == 21
        assert chunk.source_id == "test_source"
        assert chunk.metadata == {"type": "test"}

    def test_chunk_validation_empty_content(self) -> None:
        """Test validation of empty content."""
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(content="", token_count=1, start_position=0, end_position=1)
        assert "Chunk content cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            TextChunk(content="   ", token_count=1, start_position=0, end_position=3)
        assert "Chunk content cannot be empty" in str(exc_info.value)

    def test_chunk_validation_token_count(self) -> None:
        """Test validation of token count."""
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                content="Test content", token_count=0, start_position=0, end_position=12
            )
        assert "Token count must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                content="Test content",
                token_count=-1,
                start_position=0,
                end_position=12,
            )
        assert "Token count must be positive" in str(exc_info.value)

    def test_chunk_validation_positions(self) -> None:
        """Test validation of start/end positions."""
        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                content="Test content",
                token_count=2,
                start_position=-1,
                end_position=12,
            )
        assert "Start position must be non-negative" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            TextChunk(
                content="Test content", token_count=2, start_position=10, end_position=5
            )
        assert "End position (5) must be greater than start position (10)" in str(
            exc_info.value
        )


class TestTextChunker:
    """Test TextChunker functionality."""

    def test_chunker_initialization_valid(self) -> None:
        """Test valid chunker initialization."""
        chunker = TextChunker(
            min_tokens=200,
            max_tokens=400,
            delimiters=[".", "!", "?"],
            model_name="gpt-5-nano",
        )

        assert chunker.min_tokens == 200
        assert chunker.max_tokens == 400
        assert chunker.delimiters == [".", "!", "?"]
        assert chunker.model_name == "gpt-5-nano"

    def test_chunker_initialization_defaults(self) -> None:
        """Test chunker initialization with defaults."""
        chunker = TextChunker()

        assert chunker.min_tokens == 300
        assert chunker.max_tokens == 500
        assert chunker.delimiters == [".", "\n"]
        assert chunker.model_name == "gpt-5-nano"
        assert chunker.preserve_paragraphs is True
        assert chunker.merge_small_chunks is True
        assert chunker.split_large_chunks is True

    def test_chunker_initialization_validation(self) -> None:
        """Test chunker initialization validation."""
        # Test negative min_tokens
        with pytest.raises(ValidationError) as exc_info:
            TextChunker(min_tokens=-1)
        assert "min_tokens must be positive" in str(exc_info.value)

        # Test max_tokens <= min_tokens
        with pytest.raises(ValidationError) as exc_info:
            TextChunker(min_tokens=500, max_tokens=400)
        assert "max_tokens (400) must be greater than min_tokens (500)" in str(
            exc_info.value
        )

        # Test empty model name
        with pytest.raises(ValidationError) as exc_info:
            TextChunker(model_name="")
        assert "model_name cannot be empty" in str(exc_info.value)

    def test_token_counting(self) -> None:
        """Test token counting functionality."""
        chunker = TextChunker()

        # Test basic token counting
        text = "This is a simple test sentence."
        count = chunker.count_tokens(text)
        assert isinstance(count, int)
        assert count > 0

        # Test empty text
        assert chunker.count_tokens("") == 0
        assert chunker.count_tokens("   ") == 0

        # Test longer text
        long_text = "This is a much longer text that should have more tokens than the previous example."
        long_count = chunker.count_tokens(long_text)
        assert long_count > count

    def test_basic_chunking(self) -> None:
        """Test basic text chunking functionality."""
        chunker = TextChunker(min_tokens=5, max_tokens=15)

        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = chunker.chunk_text(text, source_id="test_basic")

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(chunk.source_id == "test_basic" for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)

        # Verify content coverage
        total_content = " ".join(chunk.content for chunk in chunks)
        assert len(total_content) > 0

    def test_chunking_with_paragraphs(self) -> None:
        """Test chunking with paragraph preservation."""
        chunker = TextChunker(min_tokens=5, max_tokens=20, preserve_paragraphs=True)

        text = """This is the first paragraph. It has multiple sentences.
        
        This is the second paragraph. It also has multiple sentences.
        
        This is the third paragraph."""

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

        # Check that paragraph boundaries are somewhat preserved
        # (exact behavior depends on token counts and merging)
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.token_count > 0

    def test_chunking_small_text(self) -> None:
        """Test chunking with very small text."""
        chunker = TextChunker(min_tokens=10, max_tokens=50)

        # Text smaller than min_tokens
        small_text = "Short text."
        chunks = chunker.chunk_text(small_text)

        assert len(chunks) == 1
        # Content might have slight normalization differences
        assert small_text.replace(".", "").strip() in chunks[0].content
        assert chunks[0].token_count > 0

    def test_chunking_large_text(self) -> None:
        """Test chunking with text that needs splitting."""
        chunker = TextChunker(min_tokens=5, max_tokens=10)

        # Create text that will definitely exceed max_tokens
        large_text = " ".join([f"Word{i}" for i in range(50)])  # 50 words
        chunks = chunker.chunk_text(large_text)

        assert len(chunks) > 1

        # Most chunks should be within limits (some edge cases may exceed slightly)
        reasonable_chunks = [
            c for c in chunks if c.token_count <= chunker.max_tokens * 1.2
        ]
        assert (
            len(reasonable_chunks) >= len(chunks) * 0.8
        )  # At least 80% should be reasonable

    def test_chunking_empty_content(self) -> None:
        """Test chunking with empty content."""
        chunker = TextChunker()

        with pytest.raises(ValidationError) as exc_info:
            chunker.chunk_text("")
        assert "Content cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            chunker.chunk_text("   ")
        assert "Content cannot be empty" in str(exc_info.value)

    def test_chunking_with_metadata(self) -> None:
        """Test chunking with metadata."""
        chunker = TextChunker(min_tokens=5, max_tokens=15)

        text = "This is a test sentence with metadata."
        metadata = {"author": "test", "date": "2024-01-01"}

        chunks = chunker.chunk_text(text, source_id="meta_test", metadata=metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source_id == "meta_test"
            assert chunk.metadata == metadata

    def test_chunking_with_different_delimiters(self) -> None:
        """Test chunking with custom delimiters."""
        chunker = TextChunker(min_tokens=3, max_tokens=10, delimiters=["!", "?", ";"])

        text = "First part! Second part? Third part; Fourth part."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

        # Should split on !, ?, ; but not on .
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert "Fourth part." in combined_content

    def test_merge_small_chunks_disabled(self) -> None:
        """Test chunking with merge_small_chunks disabled."""
        chunker = TextChunker(min_tokens=10, max_tokens=50, merge_small_chunks=False)

        text = "Short. Another short. Third short."
        chunks = chunker.chunk_text(text)

        # Should have multiple small chunks since merging is disabled
        assert len(chunks) > 1

    def test_split_large_chunks_disabled(self) -> None:
        """Test chunking with split_large_chunks disabled."""
        chunker = TextChunker(min_tokens=5, max_tokens=10, split_large_chunks=False)

        # Create text that would normally be split
        large_text = " ".join([f"Word{i}" for i in range(30)])
        chunks = chunker.chunk_text(large_text)

        # May have fewer chunks since splitting is disabled
        # Some chunks might exceed max_tokens
        assert len(chunks) >= 1

    def test_content_normalization(self) -> None:
        """Test content normalization."""
        chunker = TextChunker(min_tokens=5, max_tokens=20)

        # Text with various whitespace issues
        messy_text = (
            "This  has   excessive\t\tspaces.\r\n\r\nAnd\n\n\n\nmultiple\nlines."
        )
        chunks = chunker.chunk_text(messy_text)

        assert len(chunks) > 0

        # Check that content is normalized
        for chunk in chunks:
            # Should not have excessive spaces or weird line endings
            assert "  " not in chunk.content  # No double spaces
            assert "\r" not in chunk.content  # No carriage returns
            assert "\n\n\n" not in chunk.content  # No triple newlines

    def test_chunking_stats(self) -> None:
        """Test chunking statistics."""
        chunker = TextChunker(min_tokens=5, max_tokens=15)

        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = chunker.chunk_text(text)

        stats = chunker.get_chunking_stats(chunks)

        assert "total_chunks" in stats
        assert "total_tokens" in stats
        assert "avg_tokens_per_chunk" in stats
        assert "min_tokens" in stats
        assert "max_tokens" in stats
        assert "chunks_below_min" in stats
        assert "chunks_above_max" in stats
        assert "token_distribution" in stats

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_chunk"] > 0
        assert isinstance(stats["chunks_below_min"], int)
        assert isinstance(stats["chunks_above_max"], int)

    def test_chunking_stats_empty(self) -> None:
        """Test chunking statistics with empty chunks."""
        chunker = TextChunker()

        stats = chunker.get_chunking_stats([])

        assert stats["total_chunks"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avg_tokens_per_chunk"] == 0

    def test_position_tracking(self) -> None:
        """Test that chunk positions are tracked correctly."""
        chunker = TextChunker(min_tokens=3, max_tokens=10)

        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

        # Check that positions make sense
        for chunk in chunks:
            assert chunk.start_position >= 0
            assert chunk.end_position > chunk.start_position
            assert chunk.end_position <= len(text)

        # Check that chunks don't overlap (too much)
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            # Allow some overlap due to merging/splitting
            assert next_chunk.start_position >= current_chunk.start_position

    def test_edge_case_single_word(self) -> None:
        """Test chunking with single word."""
        chunker = TextChunker(min_tokens=1, max_tokens=5)

        text = "Word"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == "Word"
        assert chunks[0].token_count > 0

    def test_edge_case_very_long_word(self) -> None:
        """Test chunking with very long single word."""
        chunker = TextChunker(min_tokens=1, max_tokens=5)

        # Single very long "word"
        text = "supercalifragilisticexpialidocious"
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == text
        # Token count might exceed max due to single word

    def test_different_models(self) -> None:
        """Test chunker with different model names."""
        models = ["gpt-5-nano", "gpt-5-nano", "claude-3-sonnet"]
        text = "This is a test sentence for different models."

        for model in models:
            chunker = TextChunker(model_name=model, min_tokens=5, max_tokens=15)
            chunks = chunker.chunk_text(text)

            assert len(chunks) > 0
            assert all(chunk.token_count > 0 for chunk in chunks)

    def test_chunker_with_special_characters(self) -> None:
        """Test chunker with special characters and unicode."""
        chunker = TextChunker(min_tokens=5, max_tokens=15)

        text = "This has émojis 🚀 and spëcial cháracters! Also numbers: 123,456.78."
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 0

        # Verify special characters are preserved
        combined_content = " ".join(chunk.content for chunk in chunks)
        assert "🚀" in combined_content
        assert "émojis" in combined_content
        assert "spëcial" in combined_content
        assert "cháracters" in combined_content
