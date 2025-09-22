"""
LLM interaction module for MemoirAI.

This module provides Pydantic AI schemas, agents, and utilities for
structured LLM interactions in the MemoirAI system.
"""

from .agents import (  # Agent factory; Convenience functions; Validation functions
    AgentFactory,
    create_batch_classification_agent,
    create_category_creation_agent,
    create_category_limit_agent,
    create_classification_agent,
    create_contextual_helper_agent,
    create_final_answer_agent,
    create_query_classification_agent,
    create_summarization_agent,
    get_agent_factory,
    get_native_output_providers,
    get_supported_providers,
    validate_model_name,
)
from .schemas import (  # Classification schemas; Summarization schemas; Answer schemas; Helper schemas; Metadata and result schemas; Configuration schemas; Utility functions; Schema registries
    ALL_SCHEMAS,
    ANSWER_SCHEMAS,
    CLASSIFICATION_SCHEMAS,
    SUMMARIZATION_SCHEMAS,
    BatchClassificationResponse,
    CategoryCreation,
    CategoryLimitResponse,
    CategorySelection,
    ChunkClassificationRequest,
    ChunkSummary,
    ClassificationResult,
    ContextualHelperGeneration,
    FinalAnswer,
    LLMError,
    LLMResponseMetadata,
    ModelConfiguration,
    QueryCategorySelection,
    QueryResult,
    SummarizationResponse,
    SummarizationResult,
    ValidationResult,
    supports_native_output,
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
