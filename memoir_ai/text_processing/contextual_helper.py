"""
Contextual helper generation system for MemoirAI.
"""

import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from litellm import token_counter

from .chunker import TextChunk
from ..exceptions import ValidationError


@dataclass
class ContextualHelperData:
    """Data structure for contextual helper information."""

    author: Optional[str] = None
    date: Optional[str] = None
    topic: Optional[str] = None
    description: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None


class ContextualHelperGenerator:
    """
    Generates contextual helpers for sources to provide context in LLM prompts.

    Features:
    - Automatic helper generation from metadata and content
    - Manual helper creation with user input validation
    - ISO 8601 date validation
    - Token counting and budget management
    - Helper storage and versioning
    """

    def __init__(
        self,
        auto_source_identification: bool = True,
        max_tokens: int = 300,
        derivation_budget_tokens: int = 2000,
        max_chunks_for_derivation: int = 5,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Initialize contextual helper generator.

        Args:
            auto_source_identification: Whether to auto-generate helpers
            max_tokens: Maximum tokens for contextual helper (default 300)
            derivation_budget_tokens: Token budget for content analysis (default 2000)
            max_chunks_for_derivation: Max chunks to analyze (default 5)
            model_name: Model name for token counting
        """
        self.auto_source_identification = auto_source_identification
        self.max_tokens = max_tokens
        self.derivation_budget_tokens = derivation_budget_tokens
        self.max_chunks_for_derivation = max_chunks_for_derivation
        self.model_name = model_name

        # Validate parameters
        if max_tokens <= 0:
            raise ValidationError(
                f"max_tokens must be positive, got {max_tokens}",
                field="max_tokens",
                value=max_tokens,
            )

        if derivation_budget_tokens <= 0:
            raise ValidationError(
                f"derivation_budget_tokens must be positive, got {derivation_budget_tokens}",
                field="derivation_budget_tokens",
                value=derivation_budget_tokens,
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using liteLLM."""
        if not text or not text.strip():
            return 0

        try:
            return token_counter(model=self.model_name, text=text)
        except Exception:
            # Fallback to word count estimation
            word_count = len(text.split())
            return max(1, int(word_count * 0.75))

    def generate_contextual_helper(
        self,
        source_id: str,
        chunks: List[TextChunk],
        metadata: Optional[Dict[str, Any]] = None,
        user_provided_helper: Optional[str] = None,
    ) -> str:
        """
        Generate contextual helper for a source.

        Args:
            source_id: Source identifier
            chunks: List of text chunks from the source
            metadata: Optional metadata about the source
            user_provided_helper: Optional user-provided helper text

        Returns:
            Generated contextual helper text
        """
        if not source_id or not source_id.strip():
            raise ValidationError(
                "source_id cannot be empty", field="source_id", value=source_id
            )

        # If user provided helper, validate and return it
        if user_provided_helper:
            return self._validate_and_format_helper(
                user_provided_helper, is_user_provided=True
            )

        # If auto identification is disabled, require user input
        if not self.auto_source_identification:
            raise ValidationError(
                "Auto source identification is disabled. User must provide contextual helper.",
                field="auto_source_identification",
                value=False,
            )

        # Auto-generate helper
        return self._auto_generate_helper(source_id, chunks, metadata or {})

    def _auto_generate_helper(
        self, source_id: str, chunks: List[TextChunk], metadata: Dict[str, Any]
    ) -> str:
        """Auto-generate contextual helper from metadata and content."""

        # Extract helper data from various sources
        helper_data = self._extract_helper_data(source_id, chunks, metadata)

        # Analyze content for additional context
        content_analysis = self._analyze_content(chunks)

        # Merge content analysis into helper data
        if content_analysis.get("detected_topic") and not helper_data.topic:
            helper_data.topic = content_analysis["detected_topic"]

        if content_analysis.get("detected_title") and not helper_data.title:
            helper_data.title = content_analysis["detected_title"]

        if content_analysis.get("summary") and not helper_data.description:
            helper_data.description = content_analysis["summary"]

        # Generate helper text
        helper_text = self._compose_helper_text(helper_data)

        # Validate and format
        return self._validate_and_format_helper(helper_text, is_user_provided=False)

    def _extract_helper_data(
        self, source_id: str, chunks: List[TextChunk], metadata: Dict[str, Any]
    ) -> ContextualHelperData:
        """Extract helper data from metadata and source information."""

        helper_data = ContextualHelperData()

        # Extract from metadata
        helper_data.author = metadata.get("author") or metadata.get("created_by")
        helper_data.date = (
            metadata.get("date")
            or metadata.get("created_date")
            or metadata.get("processing_date")
        )
        helper_data.topic = metadata.get("topic") or metadata.get("subject")
        helper_data.description = metadata.get("description") or metadata.get("summary")
        helper_data.source_type = metadata.get("document_type") or metadata.get("type")
        helper_data.title = metadata.get("title") or metadata.get("name")

        # Extract from source_id (filename analysis)
        if not helper_data.title:
            helper_data.title = self._extract_title_from_source_id(source_id)

        # Extract from first chunk if available
        if chunks and not helper_data.title:
            potential_title = self._extract_title_from_content(chunks[0].content)
            if potential_title:
                helper_data.title = potential_title

        return helper_data

    def _extract_title_from_source_id(self, source_id: str) -> Optional[str]:
        """Extract potential title from source ID (filename)."""
        # Remove file extensions
        title = re.sub(r"\.[^.]+$", "", source_id)

        # Replace underscores and hyphens with spaces
        title = re.sub(r"[_-]+", " ", title)

        # Remove common prefixes/suffixes
        title = re.sub(r"^(doc|document|file|text)_?", "", title, flags=re.IGNORECASE)
        title = re.sub(r"_?(doc|document|file|text)$", "", title, flags=re.IGNORECASE)

        # Clean up whitespace
        title = " ".join(title.split())

        # Return if it looks like a meaningful title
        if len(title) > 2 and len(title.split()) <= 10:
            return title.title()

        return None

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract potential title from content (first line or header)."""
        lines = content.strip().split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Check if first line looks like a title
        if (
            len(first_line) > 5
            and len(first_line) < 100
            and not first_line.endswith(".")
            and len(first_line.split()) <= 15
        ):

            # Remove common markdown headers
            title = re.sub(r"^#+\s*", "", first_line)
            title = title.strip()

            if title:
                return title

        return None

    def _analyze_content(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Analyze content to extract topic, summary, and other information."""
        if not chunks:
            return {}

        # Collect content for analysis within budget
        analysis_content = []
        total_tokens = 0

        for chunk in chunks[: self.max_chunks_for_derivation]:
            chunk_tokens = chunk.token_count
            if total_tokens + chunk_tokens <= self.derivation_budget_tokens:
                analysis_content.append(chunk.content)
                total_tokens += chunk_tokens
            else:
                # Add partial content if it fits
                remaining_budget = self.derivation_budget_tokens - total_tokens
                if remaining_budget > 50:  # Minimum useful content
                    # Estimate how much content we can include
                    words_per_token = (
                        len(chunk.content.split()) / chunk_tokens
                        if chunk_tokens > 0
                        else 1
                    )
                    max_words = int(remaining_budget * words_per_token)
                    words = chunk.content.split()[:max_words]
                    partial_content = " ".join(words)
                    analysis_content.append(partial_content)
                break

        combined_content = "\n".join(analysis_content)

        # Simple content analysis
        analysis = {}

        # Detect potential topic from content
        analysis["detected_topic"] = self._detect_topic(combined_content)

        # Detect potential title from headers
        analysis["detected_title"] = self._detect_title_from_headers(combined_content)

        # Generate summary
        analysis["summary"] = self._generate_simple_summary(combined_content)

        return analysis

    def _detect_topic(self, content: str) -> Optional[str]:
        """Detect topic from content using simple heuristics."""
        # Look for common topic indicators
        topic_patterns = [
            r"(?:about|regarding|concerning|on the topic of)\s+([^.!?]+)",
            r"(?:this (?:document|paper|article|text) (?:discusses|covers|examines))\s+([^.!?]+)",
            r"(?:subject|topic):\s*([^.!?\n]+)",
            r"(?:document|paper|article|text)\s+(?:discusses|covers|examines)\s+([^.!?]+)",  # Better pattern
        ]

        for pattern in topic_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3 and len(topic) < 50:
                    return topic

        # Look for repeated key terms
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())
        if words:
            word_freq = {}
            for word in words:
                if word not in [
                    "this",
                    "that",
                    "with",
                    "from",
                    "they",
                    "have",
                    "been",
                    "will",
                    "were",
                    "said",
                ]:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get most frequent meaningful words
            if word_freq:
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
                    :3
                ]
                if top_words[0][1] >= 3:  # Word appears at least 3 times
                    return " ".join([word for word, count in top_words if count >= 2])

        return None

    def _detect_title_from_headers(self, content: str) -> Optional[str]:
        """Detect title from markdown headers or formatted text."""
        lines = content.split("\n")

        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()

            # Markdown headers
            if re.match(r"^#+\s+", line):
                title = re.sub(r"^#+\s+", "", line).strip()
                if len(title) > 3 and len(title) < 80:
                    return title

            # All caps lines (potential titles)
            if (
                line.isupper()
                and len(line) > 5
                and len(line) < 80
                and len(line.split()) <= 12
            ):
                return line.title()

            # Lines that look like titles (title case, reasonable length)
            if (
                line.istitle()
                and len(line) > 5
                and len(line) < 80
                and not line.endswith(".")
                and len(line.split()) <= 12
            ):
                return line

        return None

    def _generate_simple_summary(self, content: str) -> Optional[str]:
        """Generate a simple summary from content."""
        if not content:
            return None

        # Take first few sentences as summary
        sentences = re.split(r"[.!?]+", content)
        summary_sentences = []
        summary_tokens = 0
        max_summary_tokens = min(
            100, self.max_tokens // 3
        )  # Use 1/3 of budget for summary

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self.count_tokens(sentence)
            if summary_tokens + sentence_tokens <= max_summary_tokens:
                summary_sentences.append(sentence)
                summary_tokens += sentence_tokens
            else:
                break

            # Stop at reasonable summary length
            if len(summary_sentences) >= 3:
                break

        if summary_sentences:
            summary = ". ".join(summary_sentences)
            if not summary.endswith("."):
                summary += "."
            return summary

        return None

    def _compose_helper_text(self, helper_data: ContextualHelperData) -> str:
        """Compose contextual helper text from extracted data."""
        parts = []

        # Add title if available
        if helper_data.title:
            parts.append(f"Document: {helper_data.title}")

        # Add author and date
        author_date_parts = []
        if helper_data.author:
            author_date_parts.append(f"Author: {helper_data.author}")
        if helper_data.date:
            author_date_parts.append(f"Date: {helper_data.date}")

        if author_date_parts:
            parts.append(", ".join(author_date_parts))

        # Add topic
        if helper_data.topic:
            parts.append(f"Topic: {helper_data.topic}")

        # Add description/summary
        if helper_data.description:
            parts.append(f"Summary: {helper_data.description}")
        elif helper_data.source_type:
            parts.append(f"Type: {helper_data.source_type}")

        # Join parts into coherent text
        if parts:
            helper_text = ". ".join(parts)
            if not helper_text.endswith("."):
                helper_text += "."
            return helper_text

        # Fallback if no information available
        return "Document content for classification and retrieval."

    def _validate_and_format_helper(
        self, helper_text: str, is_user_provided: bool = False
    ) -> str:
        """Validate and format contextual helper text."""
        if not helper_text or not helper_text.strip():
            raise ValidationError(
                "Contextual helper text cannot be empty",
                field="helper_text",
                value=helper_text,
            )

        # Clean up the text
        helper_text = helper_text.strip()

        # Ensure it ends with a period (but don't add double periods)
        if not helper_text.endswith("."):
            helper_text += "."

        # Check token count
        token_count = self.count_tokens(helper_text)

        if token_count > self.max_tokens:
            if is_user_provided:
                raise ValidationError(
                    f"Contextual helper exceeds maximum tokens ({token_count} > {self.max_tokens})",
                    field="helper_text",
                    value=helper_text,
                )
            else:
                # Truncate auto-generated helper
                helper_text = self._truncate_helper(helper_text)

        return helper_text

    def _truncate_helper(self, helper_text: str) -> str:
        """Truncate helper text to fit within token budget."""
        sentences = re.split(r"[.!?]+", helper_text)
        truncated_sentences = []
        total_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_with_period = sentence + "."
            sentence_tokens = self.count_tokens(sentence_with_period)

            if total_tokens + sentence_tokens <= self.max_tokens:
                truncated_sentences.append(sentence)
                total_tokens += sentence_tokens
            else:
                break

        if truncated_sentences:
            result = ". ".join(truncated_sentences)
            if not result.endswith("."):
                result += "."
            return result

        # If even first sentence is too long, truncate by words
        words = helper_text.split()
        truncated_words = []
        total_tokens = 0

        for word in words:
            test_text = " ".join(truncated_words + [word])
            test_tokens = self.count_tokens(test_text)

            if test_tokens <= self.max_tokens - 1:  # Leave room for period
                truncated_words.append(word)
                total_tokens = test_tokens
            else:
                break

        if truncated_words:
            result = " ".join(truncated_words)
            if not result.endswith("."):
                result += "."
            return result

        # Absolute fallback
        return "Document content for classification."

    def collect_user_helper_data(self) -> ContextualHelperData:
        """
        Collect contextual helper data from user input.

        This is a placeholder for interactive user input collection.
        In a real implementation, this would prompt the user for input.
        """
        # This would be implemented with actual user input collection
        # For now, return empty data structure
        return ContextualHelperData()

    def validate_iso_date(self, date_str: str) -> bool:
        """Validate ISO 8601 date format."""
        if not date_str or date_str.lower() == "unknown":
            return True

        # Common ISO 8601 formats
        iso_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",  # YYYY-MM-DD
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",  # YYYY-MM-DDTHH:MM:SS
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",  # YYYY-MM-DDTHH:MM:SSZ
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",  # With timezone
            r"^\d{4} \d{2} \d{2}$",  # YYYY MM DD (space separated)
        ]

        for pattern in iso_patterns:
            if re.match(pattern, date_str):
                # Try to parse the date to ensure it's valid
                try:
                    if "T" in date_str:
                        # Handle datetime formats
                        date_part = date_str.split("T")[0]
                        datetime.strptime(date_part, "%Y-%m-%d")
                    elif " " in date_str:
                        # Handle space-separated format
                        datetime.strptime(date_str, "%Y %m %d")
                    else:
                        # Handle date-only format
                        datetime.strptime(date_str, "%Y-%m-%d")
                    return True
                except ValueError:
                    continue

        return False

    def create_user_provided_helper(
        self, author: str, date: str, topic: str, description: str
    ) -> str:
        """
        Create contextual helper from user-provided information.

        Args:
            author: Document author
            date: Document creation date (ISO 8601 format)
            topic: Document topic
            description: Short description of the document

        Returns:
            Formatted contextual helper text
        """
        # Validate inputs
        if not all([author, date, topic, description]):
            raise ValidationError(
                "All fields (author, date, topic, description) are required",
                field="user_input",
                value={
                    "author": author,
                    "date": date,
                    "topic": topic,
                    "description": description,
                },
            )

        # Validate date format
        if not self.validate_iso_date(date):
            raise ValidationError(
                f"Date must be in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS), got: {date}",
                field="date",
                value=date,
            )

        # Validate description length (200 tokens max as per requirements)
        description_tokens = self.count_tokens(description)
        if description_tokens > 200:
            raise ValidationError(
                f"Description exceeds 200 token limit ({description_tokens} tokens)",
                field="description",
                value=description,
            )

        # Create helper data
        helper_data = ContextualHelperData(
            author=author if author.lower() != "unknown" else None,
            date=date if date.lower() != "unknown" else None,
            topic=topic if topic.lower() != "unknown" else None,
            description=description if description.lower() != "unknown" else None,
        )

        # Compose and validate helper text
        helper_text = self._compose_helper_text(helper_data)
        return self._validate_and_format_helper(helper_text, is_user_provided=True)
