"""
Pydantic AI schemas for structured LLM interactions.

This module defines all the Pydantic models used for LLM interactions
in the MemoirAI system, including classification, summarization, and
query responses.
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# Classification Schemas
class ChunkClassificationRequest(BaseModel):
    """Schema for individual chunk classification in batch requests."""

    chunk_id: int = Field(description="Unique identifier for the chunk")
    category: str = Field(description="Selected category name for the chunk")


class BatchClassificationResponse(BaseModel):
    """Schema for batch classification responses from LLM."""

    chunks: List[ChunkClassificationRequest] = Field(
        description="List of chunk classifications with IDs and categories"
    )


class CategoryTree(BaseModel):
    """Schema for hierarchical category tree structure."""

    name: str = Field(description="Name of the category at this level", min_length=1)
    child: Optional["CategoryTree"] = Field(
        description="Child category in the hierarchy", default=None
    )


class HierarchicalChunkClassification(BaseModel):
    """Schema for individual chunk classification with hierarchical categories."""

    chunk_id: int = Field(description="Unique identifier for the chunk")
    category_tree: CategoryTree = Field(
        description="Hierarchical category tree for the chunk"
    )


class HierarchicalBatchClassificationResponse(BaseModel):
    """Schema for hierarchical batch classification responses from LLM."""

    classifications: List[HierarchicalChunkClassification] = Field(
        description="List of chunk classifications with hierarchical category trees"
    )


# Update forward references for recursive models
CategoryTree.model_rebuild()


class CategorySelection(BaseModel):
    """Schema for single category selection with relevance ranking."""

    category: str = Field(
        description="Name of the selected or new category", min_length=1
    )
    ranked_relevance: int = Field(
        description="Relevance ranking from 1 to N (N being most relevant)", ge=1
    )


class QueryCategorySelection(BaseModel):
    """Schema for query-based category selection during retrieval."""

    category: str = Field(description="Selected category name", min_length=1)
    ranked_relevance: int = Field(description="Relevance ranking from 1 to N", ge=1)


class QueryCategorySelectionList(BaseModel):
    """Schema for returning multiple query-based category selections."""

    selections: List[QueryCategorySelection] = Field(
        description="Ordered list of category selections from most to least relevant",
        min_length=1,
    )


# Summarization Schemas
class ChunkSummary(BaseModel):
    """Schema for individual chunk summary."""

    chunk_id: int = Field(description="Unique identifier for the chunk")
    summary: str = Field(description="Summarized content of the chunk")


class SummarizationResponse(BaseModel):
    """Schema for batch summarization responses from LLM."""

    summaries: List[ChunkSummary] = Field(
        description="List of chunk summaries with IDs and summarized content"
    )


# Query Response Schemas
class FinalAnswer(BaseModel):
    """Schema for final answer responses from LLM."""

    answer: str = Field(description="Final answer to the user query")


# Contextual Helper Generation Schemas
class ContextualHelperGeneration(BaseModel):
    """Schema for LLM-generated contextual helpers."""

    helper_text: str = Field(
        description="Generated contextual helper text", max_length=300
    )
    confidence: float = Field(
        description="Confidence score for the generated helper",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


# Category Management Schemas
class CategoryCreation(BaseModel):
    """Schema for creating new categories via LLM."""

    category_name: str = Field(description="Name of the new category to create")
    justification: str = Field(
        description="Brief justification for creating this category", max_length=200
    )


class CategoryLimitResponse(BaseModel):
    """Schema for LLM response when category limits are reached."""

    selected_category: str = Field(description="Selected existing category")
    reason: str = Field(
        description="Reason for selecting this category over creating a new one",
        max_length=150,
    )


# Error Handling Schemas
class LLMError(BaseModel):
    """Schema for structured LLM error responses."""

    error_type: Literal["validation", "timeout", "rate_limit", "model_error"] = Field(
        description="Type of error encountered"
    )
    message: str = Field(description="Human-readable error message")
    retry_suggested: bool = Field(
        description="Whether retrying the operation is recommended", default=True
    )


# Validation Schemas
class ValidationResult(BaseModel):
    """Schema for validation results."""

    is_valid: bool = Field(description="Whether the input is valid")
    errors: List[str] = Field(
        description="List of validation errors if any", default_factory=list
    )
    warnings: List[str] = Field(
        description="List of validation warnings if any", default_factory=list
    )


# Response Metadata Schemas
class LLMResponseMetadata(BaseModel):
    """Schema for LLM response metadata."""

    model: str = Field(description="Model used for the response")
    timestamp: str = Field(description="ISO timestamp of the response")
    latency_ms: int = Field(description="Response latency in milliseconds")
    tokens_prompt: Optional[int] = Field(
        description="Number of tokens in the prompt", default=None
    )
    tokens_completion: Optional[int] = Field(
        description="Number of tokens in the completion", default=None
    )
    temperature: Optional[float] = Field(
        description="Temperature used for generation", default=None
    )


# Combined Response Schemas
class ClassificationResult(BaseModel):
    """Schema for complete classification result with metadata."""

    classification: Union[CategorySelection, BatchClassificationResponse]
    metadata: LLMResponseMetadata
    validation: ValidationResult


class SummarizationResult(BaseModel):
    """Schema for complete summarization result with metadata."""

    summarization: SummarizationResponse
    metadata: LLMResponseMetadata
    validation: ValidationResult


class QueryResult(BaseModel):
    """Schema for complete query result with metadata."""

    answer: FinalAnswer
    metadata: LLMResponseMetadata
    validation: ValidationResult


# Configuration Schemas
class ModelConfiguration(BaseModel):
    """Schema for model configuration."""

    model_name: str = Field(description="Name of the model to use")
    temperature: float = Field(
        description="Temperature for generation", ge=0.0, le=2.0, default=0.0
    )
    max_tokens: Optional[int] = Field(
        description="Maximum tokens for completion", default=None
    )
    timeout: int = Field(description="Timeout in seconds", ge=1, default=30)
    retry_attempts: int = Field(description="Number of retry attempts", ge=0, default=3)


# Native Output Support Detection
NATIVE_OUTPUT_SUPPORTED_MODELS = {"openai", "grok", "gemini"}


def supports_native_output(model_name: str) -> bool:
    """
    Check if a model supports native structured output.

    Args:
        model_name: The model name (e.g., "openai:gpt-5-nano", "anthropic:claude-3")

    Returns:
        True if the model supports native structured output
    """
    if not model_name:
        return False

    provider = model_name.split(":")[0].lower()
    return provider in NATIVE_OUTPUT_SUPPORTED_MODELS


# Schema Registry
CLASSIFICATION_SCHEMAS = {
    "single": CategorySelection,
    "batch": BatchClassificationResponse,
    "hierarchical_batch": HierarchicalBatchClassificationResponse,
    "query": QueryCategorySelection,
    "query_multi": QueryCategorySelectionList,
}

SUMMARIZATION_SCHEMAS = {
    "batch": SummarizationResponse,
}

ANSWER_SCHEMAS = {
    "final": FinalAnswer,
}

ALL_SCHEMAS = {
    **CLASSIFICATION_SCHEMAS,
    **SUMMARIZATION_SCHEMAS,
    **ANSWER_SCHEMAS,
    "contextual_helper": ContextualHelperGeneration,
    "category_creation": CategoryCreation,
    "category_limit": CategoryLimitResponse,
    "category_tree": CategoryTree,
    "hierarchical_chunk_classification": HierarchicalChunkClassification,
    "validation": ValidationResult,
    "error": LLMError,
}
