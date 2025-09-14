#!/usr/bin/env python3
"""
Example usage of the ContextualHelperGenerator.

This example demonstrates how to use the contextual helper generation system
to create context for LLM prompts during classification and retrieval.
"""

from memoir_ai.text_processing.contextual_helper import ContextualHelperGenerator
from memoir_ai.text_processing.chunker import TextChunk


def main():
    """Demonstrate contextual helper generation."""
    print("=== MemoirAI Contextual Helper Generation Example ===\n")

    # Initialize the generator
    generator = ContextualHelperGenerator(
        auto_source_identification=True,
        max_tokens=300,
        derivation_budget_tokens=2000,
        max_chunks_for_derivation=5,
    )

    # Example 1: Auto-generate helper from metadata and content
    print("1. Auto-generating contextual helper from metadata and content:")
    print("-" * 60)

    # Sample chunks from a research paper
    chunks = [
        TextChunk(
            content="# Machine Learning in Healthcare\n\nThis paper examines the application of machine learning algorithms in medical diagnosis and treatment planning.",
            token_count=25,
            start_position=0,
            end_position=140,
        ),
        TextChunk(
            content="Recent advances in deep learning have enabled more accurate prediction of patient outcomes and personalized treatment recommendations.",
            token_count=20,
            start_position=141,
            end_position=270,
        ),
        TextChunk(
            content="We analyzed data from 10,000 patients across multiple hospitals to validate our approach.",
            token_count=16,
            start_position=271,
            end_position=360,
        ),
    ]

    metadata = {
        "author": "Dr. Sarah Johnson",
        "date": "2024-01-15",
        "document_type": "research_paper",
        "topic": "Healthcare AI",
    }

    helper1 = generator.generate_contextual_helper(
        source_id="ml_healthcare_2024.pdf", chunks=chunks, metadata=metadata
    )

    print(f"Generated helper: {helper1}")
    print(f"Token count: {generator.count_tokens(helper1)}")
    print()

    # Example 2: Auto-generate with minimal metadata
    print("2. Auto-generating with minimal metadata:")
    print("-" * 45)

    simple_chunks = [
        TextChunk(
            content="Climate Change Impact Assessment\n\nThis report analyzes the effects of rising temperatures on agricultural productivity.",
            token_count=18,
            start_position=0,
            end_position=120,
        )
    ]

    helper2 = generator.generate_contextual_helper(
        source_id="climate_report_2024", chunks=simple_chunks, metadata={}
    )

    print(f"Generated helper: {helper2}")
    print(f"Token count: {generator.count_tokens(helper2)}")
    print()

    # Example 3: User-provided helper
    print("3. Creating user-provided contextual helper:")
    print("-" * 42)

    try:
        user_helper = generator.create_user_provided_helper(
            author="Prof. Michael Chen",
            date="2024-02-01",
            topic="Renewable Energy",
            description="Comprehensive analysis of solar panel efficiency improvements over the past decade",
        )

        print(f"User-provided helper: {user_helper}")
        print(f"Token count: {generator.count_tokens(user_helper)}")
        print()
    except Exception as e:
        print(f"Error creating user helper: {e}")

    # Example 4: Validation examples
    print("4. Date validation examples:")
    print("-" * 28)

    test_dates = [
        "2024-01-01",  # Valid ISO date
        "2024-01-01T10:30:00",  # Valid ISO datetime
        "2024 01 01",  # Valid space-separated
        "unknown",  # Valid unknown value
        "01/01/2024",  # Invalid format
        "January 1, 2024",  # Invalid format
    ]

    for date in test_dates:
        is_valid = generator.validate_iso_date(date)
        print(f"  {date:<20} -> {'Valid' if is_valid else 'Invalid'}")

    print()

    # Example 5: Helper with auto-identification disabled
    print("5. Manual helper collection mode:")
    print("-" * 32)

    manual_generator = ContextualHelperGenerator(auto_source_identification=False)

    try:
        # This would normally prompt for user input in a real application
        manual_helper = manual_generator.generate_contextual_helper(
            source_id="manual_doc",
            chunks=[],
            metadata={},
            user_provided_helper="This is a manually provided contextual helper for classification purposes.",
        )
        print(f"Manual helper: {manual_helper}")
    except Exception as e:
        print(f"Expected error (auto-identification disabled): {e}")

    print()

    # Example 6: Token budget management
    print("6. Token budget management:")
    print("-" * 27)

    # Create a generator with a small token budget
    small_generator = ContextualHelperGenerator(max_tokens=50)

    long_metadata = {
        "title": "A Very Long Research Paper Title That Exceeds Normal Length Expectations",
        "author": "Dr. Alexander Maximilian Richardson-Thompson III",
        "description": "This is an extremely detailed description that goes on and on about various aspects of the research, including methodology, results, implications, and future work directions.",
    }

    truncated_helper = small_generator.generate_contextual_helper(
        source_id="long_paper", chunks=[], metadata=long_metadata
    )

    print(f"Truncated helper: {truncated_helper}")
    print(
        f"Token count: {small_generator.count_tokens(truncated_helper)} (max: {small_generator.max_tokens})"
    )

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
