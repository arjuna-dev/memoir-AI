"""
Data models and type definitions for MemoirAI.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class QueryStrategy(Enum):
    """Query traversal strategies for category hierarchy."""

    ONE_SHOT = "one_shot"
    WIDE_BRANCH = "wide_branch"
    ZOOM_IN = "zoom_in"
    BRANCH_OUT = "branch_out"


class PromptLimitingStrategy(Enum):
    """Strategies for handling token budget limits."""

    PRUNE = "PRUNE"
    SUMMARIZE = "SUMMARIZE"


@dataclass
class TextChunk:
    """Represents a processed text chunk."""

    content: str
    token_count: int
    start_position: int
    end_position: int
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Category:
    """Represents a category in the hierarchy."""

    id: int
    name: str
    level: int
    parent_id: Optional[int] = None
    slug: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class CategoryPath:
    """Represents a path through the category hierarchy."""

    path: List[Category]
    ranked_relevance: int


@dataclass
class ChunkWithPath:
    """Represents a chunk with its category path."""

    chunk_id: int
    content: str
    token_count: int
    category_path: str
    category_id: int
    source_id: Optional[str] = None
    created_at: Optional[datetime] = None
    ranked_relevance: Optional[int] = None


@dataclass
class ClassificationResult:
    """Result of chunk classification."""

    chunk_id: int
    category: str
    ranked_relevance: int
    level: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class LLMResponse:
    """Response metadata from LLM calls."""

    timestamp: datetime
    latency_ms: float
    model: str
    tokens_prompt: int
    tokens_completion: int
    llm_output: Dict[str, Any]


@dataclass
class IngestionResult:
    """Result of text ingestion operation."""

    source_id: str
    chunks_processed: int
    categories_created: int
    total_tokens: int
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    contextual_helper: Optional[str] = None


@dataclass
class QueryResult:
    """Result of query operation."""

    answer: str
    used_chunks: List[Dict[str, Union[int, str]]]
    dropped_paths: Optional[List[str]] = None
    total_latency: float = 0.0
    responses: List[LLMResponse] = None

    def __post_init__(self):
        if self.responses is None:
            self.responses = []


@dataclass
class CategoryTree:
    """Represents the complete category hierarchy."""

    root_categories: List[Category]
    total_categories: int
    max_depth: int
    category_counts_by_level: Dict[int, int]


@dataclass
class TokenEstimate:
    """Token usage estimation for budget management."""

    fixed_prompt_tokens: int
    chunks_total_tokens: int
    total_tokens: int
    fixed_prompt_chars: int
    chunks_total_chars: int
    total_chars: int


@dataclass
class PruningResult:
    """Result of pruning operation."""

    kept_chunks: List[ChunkWithPath]
    dropped_paths: List[str]
    final_token_count: int


@dataclass
class SummarizationResult:
    """Result of summarization operation."""

    summarized_chunks: List[ChunkWithPath]
    compression_ratio: float
    final_token_count: int
    final_char_count: int
