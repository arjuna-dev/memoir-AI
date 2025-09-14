"""
LLM interaction module for MemoirAI.

This module provides Pydantic AI schemas, agents, and utilities for
structured LLM interactions in the MemoirAI system.
"""

from .schemas import (
    # Classification schemas
    CategorySelection,
    BatchClassificationResponse,
    QueryCategorySelection,
    ChunkClassificationRequest,
    # Summarization schemas
    SummarizationResponse,
    ChunkSummary,
    # Answer schemas
    FinalAnswer,
    # Helper schemas
    ContextualHelperGeneration,
    CategoryCreation,
    CategoryLimitResponse,
    # Metadata and result schemas
    LLMResponseMetadata,
    ClassificationResult,
    SummarizationResult,
    QueryResult,
    ValidationResult,
    LLMError,
    # Configuration schemas
    ModelConfiguration,
    # Utility functions
    supports_native_output,
    # Schema registries
    CLASSIFICATION_SCHEMAS,
    SUMMARIZATION_SCHEMAS,
    ANSWER_SCHEMAS,
    ALL_SCHEMAS,
)

from .agents import (
    # Agent factory
    AgentFactory,
    get_agent_factory,
    # Convenience functions
    create_classification_agent,
    create_batch_classification_agent,
    create_query_classification_agent,
    create_summarization_agent,
    create_final_answer_agent,
    create_contextual_helper_agent,
    create_category_creation_agent,
    create_category_limit_agent,
    # Validation functions
    validate_model_name,
    get_supported_providers,
    get_native_output_providers,
)

__all__ = [
    # Schemas
    "CategorySelection",
    "BatchClassificationResponse",
    "QueryCategorySelection",
    "ChunkClassificationRequest",
    "SummarizationResponse",
    "ChunkSummary",
    "FinalAnswer",
    "ContextualHelperGeneration",
    "CategoryCreation",
    "CategoryLimitResponse",
    "LLMResponseMetadata",
    "ClassificationResult",
    "SummarizationResult",
    "QueryResult",
    "ValidationResult",
    "LLMError",
    "ModelConfiguration",
    # Schema utilities
    "supports_native_output",
    "CLASSIFICATION_SCHEMAS",
    "SUMMARIZATION_SCHEMAS",
    "ANSWER_SCHEMAS",
    "ALL_SCHEMAS",
    # Agents
    "AgentFactory",
    "get_agent_factory",
    "create_classification_agent",
    "create_batch_classification_agent",
    "create_query_classification_agent",
    "create_summarization_agent",
    "create_final_answer_agent",
    "create_contextual_helper_agent",
    "create_category_creation_agent",
    "create_category_limit_agent",
    # Validation
    "validate_model_name",
    "get_supported_providers",
    "get_native_output_providers",
]
