#!/usr/bin/env python3
"""
Example usage of Pydantic AI schemas and agents.

This example demonstrates how to use the Pydantic AI schemas and agents
for structured LLM interactions in the MemoirAI system.
"""

import asyncio
import json
from typing import List

from memoir_ai.llm import (  # Schemas; Agents; Utilities
    AgentFactory,
    BatchClassificationResponse,
    CategorySelection,
    ChunkClassificationRequest,
    ChunkSummary,
    FinalAnswer,
    LLMResponseMetadata,
    ModelConfiguration,
    SummarizationResponse,
    ValidationResult,
    create_batch_classification_agent,
    create_classification_agent,
    create_final_answer_agent,
    create_summarization_agent,
    get_native_output_providers,
    get_supported_providers,
    supports_native_output,
    validate_model_name,
)


def demonstrate_schemas():
    """Demonstrate Pydantic AI schema usage."""
    print("=== Pydantic AI Schemas Demonstration ===\n")

    # 1. Category Selection Schema
    print("1. Category Selection Schema:")
    print("-" * 30)

    selection = CategorySelection(category="Machine Learning", ranked_relevance=3)

    print(f"Category: {selection.category}")
    print(f"Relevance: {selection.ranked_relevance}")
    print(f"JSON: {selection.model_dump_json()}")
    print()

    # 2. Batch Classification Schema
    print("2. Batch Classification Schema:")
    print("-" * 33)

    batch_response = BatchClassificationResponse(
        chunks=[
            ChunkClassificationRequest(chunk_id=1, category="AI Research"),
            ChunkClassificationRequest(chunk_id=2, category="Machine Learning"),
            ChunkClassificationRequest(chunk_id=3, category="Data Science"),
        ]
    )

    print(f"Number of chunks: {len(batch_response.chunks)}")
    for chunk in batch_response.chunks:
        print(f"  Chunk {chunk.chunk_id}: {chunk.category}")
    print(f"JSON: {batch_response.model_dump_json()}")
    print()

    # 3. Summarization Schema
    print("3. Summarization Schema:")
    print("-" * 24)

    summary_response = SummarizationResponse(
        summaries=[
            ChunkSummary(
                chunk_id=1, summary="Introduction to AI concepts and applications."
            ),
            ChunkSummary(
                chunk_id=2, summary="Machine learning algorithms and their use cases."
            ),
        ]
    )

    print(f"Number of summaries: {len(summary_response.summaries)}")
    for summary in summary_response.summaries:
        print(f"  Chunk {summary.chunk_id}: {summary.summary}")
    print()

    # 4. Final Answer Schema
    print("4. Final Answer Schema:")
    print("-" * 21)

    answer = FinalAnswer(
        answer="Based on the retrieved content, machine learning is a subset of AI that enables computers to learn from data without explicit programming."
    )

    print(f"Answer: {answer.answer}")
    print()

    # 5. Metadata Schema
    print("5. Response Metadata Schema:")
    print("-" * 27)

    metadata = LLMResponseMetadata(
        model="openai:gpt-4o-mini",
        timestamp="2024-01-01T10:00:00Z",
        latency_ms=1500,
        tokens_prompt=100,
        tokens_completion=50,
        temperature=0.0,
    )

    print(f"Model: {metadata.model}")
    print(f"Latency: {metadata.latency_ms}ms")
    print(
        f"Tokens (prompt/completion): {metadata.tokens_prompt}/{metadata.tokens_completion}"
    )
    print()

    # 6. Validation Schema
    print("6. Validation Result Schema:")
    print("-" * 27)

    validation = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=["Temperature is set to 0.0, responses will be deterministic"],
    )

    print(f"Valid: {validation.is_valid}")
    print(f"Errors: {validation.errors}")
    print(f"Warnings: {validation.warnings}")
    print()


def demonstrate_model_configuration():
    """Demonstrate model configuration."""
    print("=== Model Configuration ===\n")

    # Default configuration
    default_config = ModelConfiguration(model_name="openai:gpt-4o-mini")
    print(f"Default config: {default_config.model_dump_json()}")

    # Custom configuration
    custom_config = ModelConfiguration(
        model_name="anthropic:claude-3",
        temperature=0.7,
        max_tokens=2000,
        timeout=60,
        retry_attempts=5,
    )
    print(f"Custom config: {custom_config.model_dump_json()}")
    print()


def demonstrate_native_output_support():
    """Demonstrate native output support detection."""
    print("=== Native Output Support Detection ===\n")

    test_models = [
        "openai:gpt-4o-mini",
        "openai:gpt-3.5-turbo",
        "anthropic:claude-3",
        "grok:grok-1",
        "gemini:gemini-pro",
        "ollama:llama2",
        "invalid-model",
    ]

    for model in test_models:
        supported = supports_native_output(model)
        print(f"{model:<25} -> {'Supported' if supported else 'Not supported'}")

    print()


def demonstrate_model_validation():
    """Demonstrate model name validation."""
    print("=== Model Name Validation ===\n")

    test_names = [
        "openai:gpt-4o-mini",
        "anthropic:claude-3",
        "provider:model-name",
        "invalid-name",
        ":no-provider",
        "no-model:",
        "",
        None,
    ]

    for name in test_names:
        valid = validate_model_name(name)
        print(f"{str(name):<20} -> {'Valid' if valid else 'Invalid'}")

    print()


def demonstrate_agent_factory():
    """Demonstrate AgentFactory usage."""
    print("=== Agent Factory Demonstration ===\n")

    # Create factory with default configuration
    factory = AgentFactory()
    print(f"Default model: {factory.default_config.model_name}")
    print(f"Default temperature: {factory.default_config.temperature}")
    print()

    # Create factory with custom configuration
    custom_config = ModelConfiguration(
        model_name="anthropic:claude-3", temperature=0.5, timeout=45
    )

    custom_factory = AgentFactory(custom_config)
    print(f"Custom model: {custom_factory.default_config.model_name}")
    print(f"Custom temperature: {custom_factory.default_config.temperature}")
    print()

    # Demonstrate agent creation (mocked for example)
    print("Creating agents (would normally create actual Pydantic AI agents):")

    try:
        # These would normally create real agents, but will fail without proper setup
        # In a real application, you'd have proper LLM API keys configured
        print("- Classification agent: [Would create with proper API setup]")
        print("- Batch classification agent: [Would create with proper API setup]")
        print("- Summarization agent: [Would create with proper API setup]")
        print("- Final answer agent: [Would create with proper API setup]")
    except Exception as e:
        print(f"Note: Agent creation requires proper LLM API configuration: {e}")

    print()

    # Demonstrate cache management
    cache_info = factory.get_cache_info()
    print(f"Cache info: {cache_info}")
    print()


def demonstrate_provider_info():
    """Demonstrate provider information functions."""
    print("=== Provider Information ===\n")

    supported_providers = get_supported_providers()
    print(f"Supported providers: {', '.join(supported_providers)}")

    native_output_providers = get_native_output_providers()
    print(f"Native output providers: {', '.join(native_output_providers)}")
    print()


def demonstrate_schema_validation():
    """Demonstrate schema validation with invalid data."""
    print("=== Schema Validation Examples ===\n")

    # Valid data
    print("1. Valid CategorySelection:")
    try:
        valid_selection = CategorySelection(category="Technology", ranked_relevance=5)
        print(
            f"   Success: {valid_selection.category} (relevance: {valid_selection.ranked_relevance})"
        )
    except Exception as e:
        print(f"   Error: {e}")

    # Invalid data - negative relevance
    print("\n2. Invalid CategorySelection (negative relevance):")
    try:
        invalid_selection = CategorySelection(
            category="Technology", ranked_relevance=-1  # Should be >= 1
        )
        print(f"   Success: {invalid_selection.category}")
    except Exception as e:
        print(f"   Error: {e}")

    # Invalid data - missing field
    print("\n3. Invalid BatchClassificationResponse (missing chunks):")
    try:
        invalid_batch = BatchClassificationResponse()  # Missing required 'chunks' field
        print(f"   Success: {len(invalid_batch.chunks)} chunks")
    except Exception as e:
        print(f"   Error: {e}")

    # Valid complex data
    print("\n4. Valid complex SummarizationResponse:")
    try:
        valid_summary = SummarizationResponse(
            summaries=[
                ChunkSummary(chunk_id=1, summary="First summary"),
                ChunkSummary(chunk_id=2, summary="Second summary"),
            ]
        )
        print(f"   Success: {len(valid_summary.summaries)} summaries")
        for i, summary in enumerate(valid_summary.summaries, 1):
            print(f"     {i}. Chunk {summary.chunk_id}: {summary.summary}")
    except Exception as e:
        print(f"   Error: {e}")

    print()


def demonstrate_json_serialization():
    """Demonstrate JSON serialization and deserialization."""
    print("=== JSON Serialization/Deserialization ===\n")

    # Create a complex object
    original = BatchClassificationResponse(
        chunks=[
            ChunkClassificationRequest(chunk_id=1, category="AI"),
            ChunkClassificationRequest(chunk_id=2, category="ML"),
            ChunkClassificationRequest(chunk_id=3, category="Data Science"),
        ]
    )

    # Serialize to JSON
    json_str = original.model_dump_json(indent=2)
    print("Serialized to JSON:")
    print(json_str)
    print()

    # Deserialize from JSON
    json_data = json.loads(json_str)
    reconstructed = BatchClassificationResponse(**json_data)

    print("Reconstructed from JSON:")
    print(f"Number of chunks: {len(reconstructed.chunks)}")
    for chunk in reconstructed.chunks:
        print(f"  Chunk {chunk.chunk_id}: {chunk.category}")

    # Verify they're equivalent
    print(f"\nOriginal == Reconstructed: {original == reconstructed}")
    print()


def main():
    """Run all demonstrations."""
    print("=== MemoirAI Pydantic AI Schemas and Agents Example ===\n")

    demonstrate_schemas()
    demonstrate_model_configuration()
    demonstrate_native_output_support()
    demonstrate_model_validation()
    demonstrate_agent_factory()
    demonstrate_provider_info()
    demonstrate_schema_validation()
    demonstrate_json_serialization()

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
