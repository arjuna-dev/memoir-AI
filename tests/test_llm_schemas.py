"""
Tests for LLM schemas.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from memoir_ai.llm.schemas import (
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


class TestClassificationSchemas:
    """Test classification-related schemas."""

    def test_category_selection_valid(self) -> None:
        """Test valid CategorySelection creation."""
        selection = CategorySelection(category="Machine Learning", ranked_relevance=3)

        assert selection.category == "Machine Learning"
        assert selection.ranked_relevance == 3

    def test_category_selection_invalid_relevance(self) -> None:
        """Test CategorySelection with invalid relevance."""
        with pytest.raises(ValidationError) as exc_info:
            CategorySelection(category="Test", ranked_relevance=0)  # Should be >= 1

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_chunk_classification_request(self) -> None:
        """Test ChunkClassificationRequest creation."""
        request = ChunkClassificationRequest(chunk_id=123, category="Technology")

        assert request.chunk_id == 123
        assert request.category == "Technology"

    def test_batch_classification_response(self) -> None:
        """Test BatchClassificationResponse creation."""
        chunks = [
            ChunkClassificationRequest(chunk_id=1, category="AI"),
            ChunkClassificationRequest(chunk_id=2, category="ML"),
        ]

        response = BatchClassificationResponse(chunks=chunks)

        assert len(response.chunks) == 2
        assert response.chunks[0].chunk_id == 1
        assert response.chunks[0].category == "AI"
        assert response.chunks[1].chunk_id == 2
        assert response.chunks[1].category == "ML"

    def test_query_category_selection(self) -> None:
        """Test QueryCategorySelection creation."""
        selection = QueryCategorySelection(category="Research", ranked_relevance=5)

        assert selection.category == "Research"
        assert selection.ranked_relevance == 5


class TestSummarizationSchemas:
    """Test summarization-related schemas."""

    def test_chunk_summary(self) -> None:
        """Test ChunkSummary creation."""
        summary = ChunkSummary(
            chunk_id=42, summary="This is a summary of the chunk content."
        )

        assert summary.chunk_id == 42
        assert summary.summary == "This is a summary of the chunk content."

    def test_summarization_response(self) -> None:
        """Test SummarizationResponse creation."""
        summaries = [
            ChunkSummary(chunk_id=1, summary="Summary 1"),
            ChunkSummary(chunk_id=2, summary="Summary 2"),
        ]

        response = SummarizationResponse(summaries=summaries)

        assert len(response.summaries) == 2
        assert response.summaries[0].chunk_id == 1
        assert response.summaries[0].summary == "Summary 1"


class TestAnswerSchemas:
    """Test answer-related schemas."""

    def test_final_answer(self) -> None:
        """Test FinalAnswer creation."""
        answer = FinalAnswer(answer="This is the final answer to the user's question.")

        assert answer.answer == "This is the final answer to the user's question."


class TestHelperSchemas:
    """Test helper-related schemas."""

    def test_contextual_helper_generation(self) -> None:
        """Test ContextualHelperGeneration creation."""
        helper = ContextualHelperGeneration(
            helper_text="Document about AI research from 2024.", confidence=0.95
        )

        assert helper.helper_text == "Document about AI research from 2024."
        assert helper.confidence == 0.95

    def test_contextual_helper_generation_defaults(self) -> None:
        """Test ContextualHelperGeneration with defaults."""
        helper = ContextualHelperGeneration(helper_text="Simple helper text.")

        assert helper.helper_text == "Simple helper text."
        assert helper.confidence == 1.0

    def test_contextual_helper_generation_invalid_confidence(self) -> None:
        """Test ContextualHelperGeneration with invalid confidence."""
        with pytest.raises(ValidationError) as exc_info:
            ContextualHelperGeneration(
                helper_text="Test", confidence=1.5  # Should be <= 1.0
            )

        assert "less than or equal to 1" in str(exc_info.value)

    def test_category_creation(self) -> None:
        """Test CategoryCreation creation."""
        creation = CategoryCreation(
            category_name="New Category",
            justification="This category is needed for better organization.",
        )

        assert creation.category_name == "New Category"
        assert (
            creation.justification == "This category is needed for better organization."
        )

    def test_category_limit_response(self) -> None:
        """Test CategoryLimitResponse creation."""
        response = CategoryLimitResponse(
            selected_category="Existing Category",
            reason="Best fit among available options.",
        )

        assert response.selected_category == "Existing Category"
        assert response.reason == "Best fit among available options."


class TestMetadataSchemas:
    """Test metadata and result schemas."""

    def test_llm_response_metadata(self) -> None:
        """Test LLMResponseMetadata creation."""
        metadata = LLMResponseMetadata(
            model="openai:gpt-4o-mini",
            timestamp="2024-01-01T10:00:00Z",
            latency_ms=1500,
            tokens_prompt=100,
            tokens_completion=50,
            temperature=0.0,
        )

        assert metadata.model == "openai:gpt-4o-mini"
        assert metadata.timestamp == "2024-01-01T10:00:00Z"
        assert metadata.latency_ms == 1500
        assert metadata.tokens_prompt == 100
        assert metadata.tokens_completion == 50
        assert metadata.temperature == 0.0

    def test_llm_response_metadata_minimal(self) -> None:
        """Test LLMResponseMetadata with minimal fields."""
        metadata = LLMResponseMetadata(
            model="anthropic:claude-3",
            timestamp="2024-01-01T10:00:00Z",
            latency_ms=2000,
        )

        assert metadata.model == "anthropic:claude-3"
        assert metadata.tokens_prompt is None
        assert metadata.tokens_completion is None
        assert metadata.temperature is None

    def test_validation_result(self) -> None:
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=False, errors=["Error 1", "Error 2"], warnings=["Warning 1"]
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.errors[0] == "Error 1"
        assert result.warnings[0] == "Warning 1"

    def test_validation_result_defaults(self) -> None:
        """Test ValidationResult with defaults."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_llm_error(self) -> None:
        """Test LLMError creation."""
        error = LLMError(
            error_type="validation",
            message="Schema validation failed",
            retry_suggested=False,
        )

        assert error.error_type == "validation"
        assert error.message == "Schema validation failed"
        assert error.retry_suggested is False

    def test_llm_error_defaults(self) -> None:
        """Test LLMError with defaults."""
        error = LLMError(error_type="timeout", message="Request timed out")

        assert error.error_type == "timeout"
        assert error.message == "Request timed out"
        assert error.retry_suggested is True


class TestConfigurationSchemas:
    """Test configuration schemas."""

    def test_model_configuration(self) -> None:
        """Test ModelConfiguration creation."""
        config = ModelConfiguration(
            model_name="openai:gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            retry_attempts=5,
        )

        assert config.model_name == "openai:gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 60
        assert config.retry_attempts == 5

    def test_model_configuration_defaults(self) -> None:
        """Test ModelConfiguration with defaults."""
        config = ModelConfiguration(model_name="test:model")

        assert config.model_name == "test:model"
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.timeout == 30
        assert config.retry_attempts == 3

    def test_model_configuration_invalid_temperature(self) -> None:
        """Test ModelConfiguration with invalid temperature."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfiguration(
                model_name="test:model", temperature=3.0  # Should be <= 2.0
            )

        assert "less than or equal to 2" in str(exc_info.value)

    def test_model_configuration_invalid_timeout(self) -> None:
        """Test ModelConfiguration with invalid timeout."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfiguration(model_name="test:model", timeout=0)  # Should be >= 1

        assert "greater than or equal to 1" in str(exc_info.value)


class TestResultSchemas:
    """Test combined result schemas."""

    def test_classification_result(self) -> None:
        """Test ClassificationResult creation."""
        classification = CategorySelection(category="Test", ranked_relevance=1)
        metadata = LLMResponseMetadata(
            model="test:model", timestamp="2024-01-01T10:00:00Z", latency_ms=1000
        )
        validation = ValidationResult(is_valid=True)

        result = ClassificationResult(
            classification=classification, metadata=metadata, validation=validation
        )

        assert isinstance(result.classification, CategorySelection)
        assert result.classification.category == "Test"
        assert result.metadata.model == "test:model"
        assert result.validation.is_valid is True

    def test_summarization_result(self) -> None:
        """Test SummarizationResult creation."""
        summarization = SummarizationResponse(
            summaries=[ChunkSummary(chunk_id=1, summary="Test summary")]
        )
        metadata = LLMResponseMetadata(
            model="test:model", timestamp="2024-01-01T10:00:00Z", latency_ms=1000
        )
        validation = ValidationResult(is_valid=True)

        result = SummarizationResult(
            summarization=summarization, metadata=metadata, validation=validation
        )

        assert isinstance(result.summarization, SummarizationResponse)
        assert len(result.summarization.summaries) == 1
        assert result.metadata.model == "test:model"
        assert result.validation.is_valid is True

    def test_query_result(self) -> None:
        """Test QueryResult creation."""
        answer = FinalAnswer(answer="Test answer")
        metadata = LLMResponseMetadata(
            model="test:model", timestamp="2024-01-01T10:00:00Z", latency_ms=1000
        )
        validation = ValidationResult(is_valid=True)

        result = QueryResult(answer=answer, metadata=metadata, validation=validation)

        assert isinstance(result.answer, FinalAnswer)
        assert result.answer.answer == "Test answer"
        assert result.metadata.model == "test:model"
        assert result.validation.is_valid is True


class TestUtilityFunctions:
    """Test utility functions."""

    def test_supports_native_output_openai(self) -> None:
        """Test native output support detection for OpenAI."""
        assert supports_native_output("openai:gpt-4o-mini") is True
        assert supports_native_output("openai:gpt-3.5-turbo") is True

    def test_supports_native_output_anthropic(self) -> None:
        """Test native output support detection for Anthropic."""
        assert supports_native_output("anthropic:claude-3") is False

    def test_supports_native_output_grok(self) -> None:
        """Test native output support detection for Grok."""
        assert supports_native_output("grok:grok-1") is True

    def test_supports_native_output_gemini(self) -> None:
        """Test native output support detection for Gemini."""
        assert supports_native_output("gemini:gemini-pro") is True

    def test_supports_native_output_invalid(self) -> None:
        """Test native output support detection for invalid models."""
        assert supports_native_output("") is False
        assert supports_native_output("invalid") is False
        assert supports_native_output("unknown:model") is False

    def test_schema_registries(self) -> None:
        """Test schema registries."""
        assert "single" in CLASSIFICATION_SCHEMAS
        assert "batch" in CLASSIFICATION_SCHEMAS
        assert "query" in CLASSIFICATION_SCHEMAS

        assert "batch" in SUMMARIZATION_SCHEMAS

        assert "final" in ANSWER_SCHEMAS

        assert "contextual_helper" in ALL_SCHEMAS
        assert "category_creation" in ALL_SCHEMAS
        assert "validation" in ALL_SCHEMAS
        assert "error" in ALL_SCHEMAS

    def test_schema_registry_types(self) -> None:
        """Test schema registry contains correct types."""
        assert CLASSIFICATION_SCHEMAS["single"] == CategorySelection
        assert CLASSIFICATION_SCHEMAS["batch"] == BatchClassificationResponse
        assert CLASSIFICATION_SCHEMAS["query"] == QueryCategorySelection

        assert SUMMARIZATION_SCHEMAS["batch"] == SummarizationResponse

        assert ANSWER_SCHEMAS["final"] == FinalAnswer

        assert ALL_SCHEMAS["contextual_helper"] == ContextualHelperGeneration
        assert ALL_SCHEMAS["category_creation"] == CategoryCreation
        assert ALL_SCHEMAS["validation"] == ValidationResult
        assert ALL_SCHEMAS["error"] == LLMError
