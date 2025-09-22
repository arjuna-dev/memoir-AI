"""
Token-based text chunker for MemoirAI.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from litellm import token_counter

from ..exceptions import ValidationError


@dataclass
class TextChunk:
    """Represents a processed text chunk with token information."""

    content: str
    token_count: int
    start_position: int
    end_position: int
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not self.content or not self.content.strip():
            raise ValidationError(
                "Chunk content cannot be empty", field="content", value=self.content
            )

        if self.token_count <= 0:
            raise ValidationError(
                f"Token count must be positive, got {self.token_count}",
                field="token_count",
                value=self.token_count,
            )

        if self.start_position < 0:
            raise ValidationError(
                f"Start position must be non-negative, got {self.start_position}",
                field="start_position",
                value=self.start_position,
            )

        if self.end_position <= self.start_position:
            raise ValidationError(
                f"End position ({self.end_position}) must be greater than start position ({self.start_position})",
                field="end_position",
                value=self.end_position,
            )


class TextChunker:
    """
    Token-based text chunker with configurable size constraints and delimiter handling.

    Features:
    - Token counting using liteLLM for accurate model-specific counts
    - Configurable delimiters for natural text boundaries
    - Paragraph boundary preservation
    - Chunk merging for undersized chunks
    - Chunk splitting for oversized chunks
    - Comprehensive validation and error handling
    """

    def __init__(
        self,
        min_tokens: int = 300,
        max_tokens: int = 500,
        delimiters: Optional[List[str]] = None,
        model_name: str = "gpt-3.5-turbo",
        preserve_paragraphs: bool = True,
        merge_small_chunks: bool = True,
        split_large_chunks: bool = True,
    ) -> None:
        """
        Initialize text chunker with configuration.

        Args:
            min_tokens: Minimum tokens per chunk (default 300)
            max_tokens: Maximum tokens per chunk (default 500)
            delimiters: List of delimiters for splitting (default: [".", "\n"])
            model_name: Model name for token counting (default: "gpt-3.5-turbo")
            preserve_paragraphs: Whether to preserve paragraph boundaries
            merge_small_chunks: Whether to merge chunks below min_tokens
            split_large_chunks: Whether to split chunks above max_tokens
        """
        # Validate parameters
        if min_tokens <= 0:
            raise ValidationError(
                f"min_tokens must be positive, got {min_tokens}",
                field="min_tokens",
                value=min_tokens,
            )

        if max_tokens <= min_tokens:
            raise ValidationError(
                f"max_tokens ({max_tokens}) must be greater than min_tokens ({min_tokens})",
                field="max_tokens",
                value=max_tokens,
            )

        if not model_name or not model_name.strip():
            raise ValidationError(
                "model_name cannot be empty", field="model_name", value=model_name
            )

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.delimiters = delimiters or [".", "\n"]
        self.model_name = model_name.strip()
        self.preserve_paragraphs = preserve_paragraphs
        self.merge_small_chunks = merge_small_chunks
        self.split_large_chunks = split_large_chunks

        # Compile delimiter patterns for efficient splitting
        self._compile_delimiter_patterns()

    def _compile_delimiter_patterns(self) -> None:
        """Compile regex patterns for delimiters."""
        # Escape special regex characters in delimiters
        escaped_delimiters = [re.escape(d) for d in self.delimiters]

        # Create pattern that matches any delimiter
        self.delimiter_pattern = re.compile(f"({'|'.join(escaped_delimiters)})")

        # Pattern for paragraph boundaries (double newlines)
        self.paragraph_pattern = re.compile(r"\n\s*\n")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using liteLLM.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text or not text.strip():
            return 0

        try:
            return int(token_counter(model=self.model_name, text=text))
        except Exception as e:
            # Fallback to simple word count estimation if token counting fails
            # This is a rough approximation: ~0.75 tokens per word
            word_count = len(text.split())
            estimated_tokens = max(1, int(word_count * 0.75))

            # Log warning in production
            print(
                f"Warning: Token counting failed for model {self.model_name}, "
                f"using word count estimation: {estimated_tokens} tokens. Error: {e}"
            )

            return estimated_tokens

    def chunk_text(
        self,
        content: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Split text into chunks respecting token size constraints and boundaries.

        Args:
            content: Text content to chunk
            source_id: Optional source identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        if not content or not content.strip():
            raise ValidationError(
                "Content cannot be empty", field="content", value=content
            )

        # Normalize content (remove excessive whitespace)
        normalized_content = self._normalize_content(content)

        # Step 1: Initial splitting by delimiters
        initial_segments = self._split_by_delimiters(normalized_content)

        # Step 2: Preserve paragraph boundaries if enabled
        if self.preserve_paragraphs:
            initial_segments = self._preserve_paragraph_boundaries(initial_segments)

        # Step 3: Create initial chunks with token counts
        initial_chunks = self._create_initial_chunks(
            initial_segments, normalized_content, source_id, metadata
        )

        # Step 4: Merge small chunks if enabled
        if self.merge_small_chunks:
            initial_chunks = self._merge_small_chunks(initial_chunks)

        # Step 5: Split large chunks if enabled
        if self.split_large_chunks:
            initial_chunks = self._split_large_chunks(initial_chunks)

        # Step 6: Final validation and cleanup
        final_chunks = self._validate_and_cleanup_chunks(initial_chunks)

        return final_chunks

    def _normalize_content(self, content: str) -> str:
        """Normalize content by cleaning up whitespace."""
        # Remove excessive whitespace while preserving paragraph structure
        content = re.sub(r"\r\n", "\n", content)  # Normalize line endings
        content = re.sub(r"\r", "\n", content)  # Handle old Mac line endings
        content = re.sub(r"\n{3,}", "\n\n", content)  # Limit consecutive newlines
        content = re.sub(r"[ \t]+", " ", content)  # Normalize spaces and tabs
        content = content.strip()

        return content

    def _split_by_delimiters(self, content: str) -> List[str]:
        """Split content by configured delimiters."""
        segments = []
        current_pos = 0

        # Find all delimiter matches
        for match in self.delimiter_pattern.finditer(content):
            # Add text before delimiter
            if match.start() > current_pos:
                segment = content[current_pos : match.start()].strip()
                if segment:
                    segments.append(segment)

            # Add delimiter with following text until next delimiter or end
            delimiter_start = match.start()
            next_match = None

            # Find next delimiter
            remaining_content = content[match.end() :]
            next_delimiter_match = self.delimiter_pattern.search(remaining_content)

            if next_delimiter_match:
                # Include text up to next delimiter
                segment_end = match.end() + next_delimiter_match.start()
                segment = content[delimiter_start:segment_end].strip()
            else:
                # Include rest of content
                segment = content[delimiter_start:].strip()

            if segment:
                segments.append(segment)
                current_pos = delimiter_start + len(segment)

        # Handle case where no delimiters found
        if not segments and content.strip():
            segments.append(content.strip())

        return segments

    def _preserve_paragraph_boundaries(self, segments: List[str]) -> List[str]:
        """Ensure paragraph boundaries are preserved."""
        if not self.preserve_paragraphs:
            return segments

        preserved_segments = []

        for segment in segments:
            # Split by paragraph boundaries
            paragraphs = self.paragraph_pattern.split(segment)

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    preserved_segments.append(paragraph)

        return preserved_segments

    def _create_initial_chunks(
        self,
        segments: List[str],
        original_content: str,
        source_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> List[TextChunk]:
        """Create initial chunks from segments."""
        chunks = []
        current_pos = 0

        for segment in segments:
            # Find segment position in original content
            segment_start = original_content.find(segment, current_pos)
            if segment_start == -1:
                # Fallback: approximate position
                segment_start = current_pos

            segment_end = segment_start + len(segment)

            # Count tokens
            token_count = self.count_tokens(segment)

            # Create chunk
            chunk = TextChunk(
                content=segment,
                token_count=token_count,
                start_position=segment_start,
                end_position=segment_end,
                source_id=source_id,
                metadata=metadata.copy() if metadata else None,
            )

            chunks.append(chunk)
            current_pos = segment_end

        return chunks

    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are below minimum token threshold."""
        if not chunks:
            return chunks

        merged_chunks = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if current chunk is too small and can be merged
            if (
                current_chunk.token_count < self.min_tokens
                and current_chunk.token_count + next_chunk.token_count
                <= self.max_tokens
            ):
                # Merge chunks
                merged_content = current_chunk.content + " " + next_chunk.content
                merged_token_count = self.count_tokens(merged_content)

                current_chunk = TextChunk(
                    content=merged_content,
                    token_count=merged_token_count,
                    start_position=current_chunk.start_position,
                    end_position=next_chunk.end_position,
                    source_id=current_chunk.source_id,
                    metadata=current_chunk.metadata,
                )
            else:
                # Can't merge, add current chunk and move to next
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        # Add the last chunk
        merged_chunks.append(current_chunk)

        return merged_chunks

    def _split_large_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Split chunks that exceed maximum token threshold."""
        split_chunks = []

        for chunk in chunks:
            if chunk.token_count <= self.max_tokens:
                split_chunks.append(chunk)
            else:
                # Split the chunk
                sub_chunks = self._split_single_chunk(chunk)
                split_chunks.extend(sub_chunks)

        return split_chunks

    def _split_single_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """Split a single large chunk into smaller chunks."""
        content = chunk.content
        words = content.split()

        if not words:
            return [chunk]

        sub_chunks = []
        current_words: List[str] = []
        current_start = chunk.start_position

        for word in words:
            # Test adding this word
            test_content = " ".join(current_words + [word])
            test_token_count = self.count_tokens(test_content)

            if test_token_count <= self.max_tokens:
                # Safe to add word
                current_words.append(word)
            else:
                # Would exceed limit, create chunk with current words
                if current_words:
                    chunk_content = " ".join(current_words)
                    chunk_token_count = self.count_tokens(chunk_content)

                    # Estimate positions (approximate)
                    chunk_length = len(chunk_content)
                    chunk_end = current_start + chunk_length

                    sub_chunk = TextChunk(
                        content=chunk_content,
                        token_count=chunk_token_count,
                        start_position=current_start,
                        end_position=chunk_end,
                        source_id=chunk.source_id,
                        metadata=chunk.metadata,
                    )

                    sub_chunks.append(sub_chunk)
                    current_start = chunk_end + 1  # +1 for space

                # Start new chunk with current word
                current_words = [word]

        # Handle remaining words
        if current_words:
            chunk_content = " ".join(current_words)
            chunk_token_count = self.count_tokens(chunk_content)

            sub_chunk = TextChunk(
                content=chunk_content,
                token_count=chunk_token_count,
                start_position=current_start,
                end_position=chunk.end_position,
                source_id=chunk.source_id,
                metadata=chunk.metadata,
            )

            sub_chunks.append(sub_chunk)

        return sub_chunks if sub_chunks else [chunk]

    def _validate_and_cleanup_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Final validation and cleanup of chunks."""
        valid_chunks = []

        for chunk in chunks:
            # Skip empty chunks
            if not chunk.content.strip():
                continue

            # Warn about chunks that are still too small or large
            if chunk.token_count < self.min_tokens:
                print(
                    f"Warning: Chunk below minimum tokens ({chunk.token_count} < {self.min_tokens}): "
                    f"{chunk.content[:50]}..."
                )

            if chunk.token_count > self.max_tokens:
                print(
                    f"Warning: Chunk above maximum tokens ({chunk.token_count} > {self.max_tokens}): "
                    f"{chunk.content[:50]}..."
                )

            valid_chunks.append(chunk)

        return valid_chunks

    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "chunks_below_min": 0,
                "chunks_above_max": 0,
            }

        token_counts = [chunk.token_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunks_below_min": sum(
                1 for count in token_counts if count < self.min_tokens
            ),
            "chunks_above_max": sum(
                1 for count in token_counts if count > self.max_tokens
            ),
            "token_distribution": {
                "p25": sorted(token_counts)[len(token_counts) // 4],
                "p50": sorted(token_counts)[len(token_counts) // 2],
                "p75": sorted(token_counts)[3 * len(token_counts) // 4],
            },
        }
