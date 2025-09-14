#!/usr/bin/env python3
"""
Example usage of the Batch Classification System.

This example demonstrates how to use the BatchCategoryClassifier
for efficient multi-chunk classification in the MemoirAI system.
"""

import asyncio
from typing import List

from memoir_ai.classification import (
    BatchCategoryClassifier,
    ClassificationResult,
    create_batch_classifier,
    validate_batch_size,
)
from memoir_ai.text_processing.chunker import TextChunk
from memoir_ai.database.models import Category


def create_sample_chunks() -> List[TextChunk]:
    """Create sample text chunks for demonstration."""
    sample_texts = [
        "Machine learning algorithms are transforming how we process data and make predictions.",
        "Climate change research shows significant warming trends over the past century.",
        "Quantum computing promises to revolutionize cryptography and complex calculations.",
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
        "Artificial intelligence applications in healthcare are improving diagnostic accuracy.",
        "Blockchain technology enables secure and decentralized transaction processing.",
        "Space exploration missions are advancing our understanding of the universe.",
        "Biotechnology innovations are leading to new treatments for genetic diseases.",
    ]

    chunks = []
    for i, text in enumerate(sample_texts):
        chunk = TextChunk(
            content=text,
            token_count=len(text.split()),  # Simple token approximation
            start_position=i * 100,
            end_position=(i + 1) * 100 - 1,
        )
        # Add chunk_id for batch processing
        chunk.chunk_id = i + 1
        chunks.append(chunk)

    return chunks


def create_sample_categories() -> List[Category]:
    """Create sample categories for demonstration."""
    categories = [
        Category(id=1, name="Technology", level=1, parent_id=None),
        Category(id=2, name="Science", level=1, parent_id=None),
        Category(id=3, name="Environment", level=1, parent_id=None),
        Category(id=4, name="Healthcare", level=1, parent_id=None),
    ]
    return categories


def demonstrate_batch_classifier_creation():
    """Demonstrate different ways to create batch classifiers."""
    print("=== Batch Classifier Creation ===\n")

    # Method 1: Direct instantiation
    print("1. Direct instantiation:")
    try:
        classifier1 = BatchCategoryClassifier(
            model_name="openai:gpt-4", batch_size=3, max_retries=2, temperature=0.1
        )
        print(f"   Created classifier with batch_size={classifier1.batch_size}")
        print(f"   Model: {classifier1.model_name}")
        print(f"   Max retries: {classifier1.max_retries}")
    except Exception as e:
        print(f"   Note: Requires proper LLM API setup: {e}")

    print()

    # Method 2: Using utility function
    print("2. Using utility function:")
    try:
        classifier2 = create_batch_classifier(
            model_name="anthropic:claude-3", batch_size=5, hierarchy_depth=4
        )
        print(f"   Created classifier with batch_size={classifier2.batch_size}")
        print(f"   Hierarchy depth: {classifier2.hierarchy_depth}")
    except Exception as e:
        print(f"   Note: Requires proper LLM API setup: {e}")

    print()


def demonstrate_batch_size_validation():
    """Demonstrate batch size validation."""
    print("=== Batch Size Validation ===\n")

    test_sizes = [0, 1, 5, 10, 25, 50, 51, 100]

    for size in test_sizes:
        is_valid = validate_batch_size(size)
        status = "Valid" if is_valid else "Invalid"
        print(f"   Batch size {size:3d}: {status}")

    print()


def demonstrate_prompt_creation():
    """Demonstrate batch prompt creation."""
    print("=== Batch Prompt Creation ===\n")

    try:
        classifier = BatchCategoryClassifier(batch_size=3)

        # Create sample data
        chunks = create_sample_chunks()[:2]  # Use first 2 chunks
        categories = create_sample_categories()[:2]  # Use first 2 categories
        contextual_helper = (
            "Research document about emerging technologies and their applications"
        )

        # Create batch prompt
        prompt = classifier._create_batch_prompt(
            chunks=chunks,
            existing_categories=categories,
            contextual_helper=contextual_helper,
            level=1,
        )

        print("Generated batch prompt:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)

    except Exception as e:
        print(f"Note: Requires proper setup: {e}")

    print()


def demonstrate_response_validation():
    """Demonstrate batch response validation."""
    print("=== Response Validation Examples ===\n")

    try:
        from memoir_ai.llm.schemas import (
            BatchClassificationResponse,
            ChunkClassificationRequest,
        )

        classifier = BatchCategoryClassifier()

        # Valid response
        print("1. Valid response:")
        valid_response = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category="Technology"),
                ChunkClassificationRequest(chunk_id=2, category="Science"),
            ]
        )

        is_valid = classifier._validate_batch_response(valid_response, 2)
        print(f"   Response with 2 chunks: {'Valid' if is_valid else 'Invalid'}")

        # Invalid response - wrong count
        print("\n2. Invalid response (wrong count):")
        invalid_response = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category="Technology"),
            ]
        )

        is_valid = classifier._validate_batch_response(invalid_response, 2)
        print(
            f"   Response with 1 chunk (expected 2): {'Valid' if is_valid else 'Invalid'}"
        )

        # Invalid response - duplicate IDs
        print("\n3. Invalid response (duplicate IDs):")
        duplicate_response = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category="Technology"),
                ChunkClassificationRequest(chunk_id=1, category="Science"),
            ]
        )

        is_valid = classifier._validate_batch_response(duplicate_response, 2)
        print(f"   Response with duplicate IDs: {'Valid' if is_valid else 'Invalid'}")

        # Invalid response - empty category
        print("\n4. Invalid response (empty category):")
        empty_response = BatchClassificationResponse(
            chunks=[
                ChunkClassificationRequest(chunk_id=1, category=""),
                ChunkClassificationRequest(chunk_id=2, category="Science"),
            ]
        )

        is_valid = classifier._validate_batch_response(empty_response, 2)
        print(f"   Response with empty category: {'Valid' if is_valid else 'Invalid'}")

    except Exception as e:
        print(f"Note: Requires proper setup: {e}")

    print()


def demonstrate_metrics_tracking():
    """Demonstrate metrics tracking and reporting."""
    print("=== Metrics Tracking ===\n")

    try:
        classifier = BatchCategoryClassifier()

        # Simulate some batch metrics
        from memoir_ai.classification.batch_classifier import BatchClassificationMetrics
        from datetime import datetime

        # Add sample metrics
        classifier.metrics_history = [
            BatchClassificationMetrics(
                batch_id="batch_001",
                chunks_sent=5,
                chunks_successful=4,
                chunks_failed=1,
                chunks_retried=1,
                total_latency_ms=1200,
                llm_calls=2,
                timestamp=datetime.now(),
                model_name="openai:gpt-4",
            ),
            BatchClassificationMetrics(
                batch_id="batch_002",
                chunks_sent=3,
                chunks_successful=3,
                chunks_failed=0,
                chunks_retried=0,
                total_latency_ms=800,
                llm_calls=1,
                timestamp=datetime.now(),
                model_name="openai:gpt-4",
            ),
            BatchClassificationMetrics(
                batch_id="batch_003",
                chunks_sent=7,
                chunks_successful=6,
                chunks_failed=1,
                chunks_retried=2,
                total_latency_ms=1500,
                llm_calls=3,
                timestamp=datetime.now(),
                model_name="openai:gpt-4",
            ),
        ]

        # Get metrics summary
        summary = classifier.get_metrics_summary()

        print("Metrics Summary:")
        print(f"   Total batches: {summary['total_batches']}")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Successful: {summary['total_successful']}")
        print(f"   Failed: {summary['total_failed']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Average latency: {summary['average_latency_ms']:.0f}ms")
        print(f"   Total LLM calls: {summary['total_llm_calls']}")
        print(f"   Avg chunks per batch: {summary['average_chunks_per_batch']:.1f}")

        print("\nIndividual Batch Details:")
        for i, metrics in enumerate(classifier.metrics_history, 1):
            print(f"   Batch {i} ({metrics.batch_id}):")
            print(
                f"     Chunks: {metrics.chunks_sent} sent, {metrics.chunks_successful} successful"
            )
            print(f"     Retries: {metrics.chunks_retried}")
            print(f"     Latency: {metrics.total_latency_ms}ms")
            print(f"     LLM calls: {metrics.llm_calls}")

    except Exception as e:
        print(f"Note: {e}")

    print()


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    print("=== Configuration Options ===\n")

    configurations = [
        {
            "name": "Small Batches (Fast)",
            "batch_size": 2,
            "max_retries": 1,
            "temperature": 0.0,
        },
        {
            "name": "Medium Batches (Balanced)",
            "batch_size": 5,
            "max_retries": 3,
            "temperature": 0.1,
        },
        {
            "name": "Large Batches (Efficient)",
            "batch_size": 10,
            "max_retries": 2,
            "temperature": 0.0,
        },
        {
            "name": "Per-Level Category Limits",
            "batch_size": 5,
            "max_categories_per_level": {1: 10, 2: 20, 3: 50},
        },
    ]

    for config in configurations:
        print(f"Configuration: {config['name']}")
        try:
            classifier = BatchCategoryClassifier(
                **{k: v for k, v in config.items() if k != "name"}
            )
            print(f"   Batch size: {classifier.batch_size}")
            print(f"   Max retries: {classifier.max_retries}")
            print(f"   Temperature: {classifier.temperature}")
            print(f"   Category limits: {classifier.max_categories_per_level}")
        except Exception as e:
            print(f"   Error: {e}")
        print()


async def demonstrate_mock_classification():
    """Demonstrate classification with mocked responses."""
    print("=== Mock Classification Example ===\n")

    # This would normally require actual LLM API calls
    # For demonstration, we'll show the structure

    chunks = create_sample_chunks()[:3]  # Use first 3 chunks
    categories = create_sample_categories()
    contextual_helper = "Technology research document covering AI, quantum computing, and renewable energy"

    print("Input chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. {chunk.content[:60]}...")

    print(f"\nExisting categories: {[cat.name for cat in categories]}")
    print(f"Contextual helper: {contextual_helper}")

    # Simulate what the results would look like
    print("\nExpected classification results:")
    expected_results = [
        ClassificationResult(
            chunk_id=1,
            chunk=chunks[0],
            category="Technology",
            success=True,
            latency_ms=800,
        ),
        ClassificationResult(
            chunk_id=2,
            chunk=chunks[1],
            category="Environment",
            success=True,
            latency_ms=800,
        ),
        ClassificationResult(
            chunk_id=3,
            chunk=chunks[2],
            category="Technology",
            success=True,
            latency_ms=800,
        ),
    ]

    for result in expected_results:
        status = "✓" if result.success else "✗"
        print(f"   {status} Chunk {result.chunk_id}: {result.category}")
        if result.retry_count > 0:
            print(f"     (after {result.retry_count} retries)")

    print(f"\nBatch processing would complete in ~800ms with 1 LLM call")
    print("(vs ~2400ms with 3 individual calls)")


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("=== Error Handling Examples ===\n")

    # Invalid configuration
    print("1. Invalid Configuration:")
    try:
        BatchCategoryClassifier(batch_size=0)
    except Exception as e:
        print(f"   Error: {e}")

    try:
        BatchCategoryClassifier(batch_size=100)
    except Exception as e:
        print(f"   Error: {e}")

    try:
        BatchCategoryClassifier(max_retries=-1)
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. Batch Size Validation:")
    invalid_sizes = [-1, 0, 51, 100]
    for size in invalid_sizes:
        is_valid = validate_batch_size(size)
        print(f"   Size {size}: {'Valid' if is_valid else 'Invalid'}")

    print()


def main():
    """Run all demonstrations."""
    print("=== MemoirAI Batch Classification System Example ===\n")

    demonstrate_batch_classifier_creation()
    demonstrate_batch_size_validation()
    demonstrate_prompt_creation()
    demonstrate_response_validation()
    demonstrate_metrics_tracking()
    demonstrate_configuration_options()

    # Async demonstration
    print("Running async demonstration...")
    asyncio.run(demonstrate_mock_classification())
    print()

    demonstrate_error_handling()

    print("=== Key Benefits of Batch Classification ===")
    print("• Reduced LLM API calls (5 chunks in 1 call vs 5 calls)")
    print("• Lower latency (parallel processing)")
    print("• Cost efficiency (fewer API requests)")
    print("• Automatic retry logic for failed chunks")
    print("• Comprehensive metrics and monitoring")
    print("• Configurable batch sizes and retry strategies")
    print()

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
