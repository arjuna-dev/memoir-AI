"""
Complete query workflow example for MemoirAI.

This example demonstrates the end-to-end query processing pipeline,
from natural language query to retrieved chunks with metadata.
"""

import asyncio
from datetime import datetime

from memoir_ai.query import (
    QueryProcessor,
    QueryStrategy,
    create_query_processor,
    process_natural_language_query,
)


async def main():
    """Demonstrate complete query workflow."""

    print("=== MemoirAI Complete Query Workflow Example ===\n")

    # Note: In a real application, you would have:
    # - A properly initialized database with categories and chunks
    # - A real CategoryManager with actual data
    # - A database session connected to your data

    print("This example shows the structure of the complete query workflow.")
    print("In a real implementation, you would:")
    print("1. Initialize your database with categories and chunks")
    print("2. Create a CategoryManager with your database session")
    print("3. Use the QueryProcessor to process natural language queries")
    print()

    # Example query workflow structure
    example_queries = [
        {
            "query": "machine learning algorithms for natural language processing",
            "strategy": QueryStrategy.ONE_SHOT,
            "description": "Precise search using one-shot strategy",
        },
        {
            "query": "artificial intelligence research papers",
            "strategy": QueryStrategy.WIDE_BRANCH,
            "params": {"n": 3},
            "description": "Broad search using wide branch strategy",
        },
        {
            "query": "deep learning neural networks",
            "strategy": QueryStrategy.ZOOM_IN,
            "params": {"n": 4, "n2": 1},
            "description": "Exploratory search that narrows down",
        },
        {
            "query": "computer vision applications",
            "strategy": QueryStrategy.BRANCH_OUT,
            "params": {"n": 1, "n2": 2},
            "description": "Focused search that expands",
        },
    ]

    for i, example in enumerate(example_queries, 1):
        print(f"--- Example {i}: {example['description']} ---")
        print(f"Query: '{example['query']}'")
        print(f"Strategy: {example['strategy'].value}")

        if "params" in example:
            print(f"Parameters: {example['params']}")

        print("Expected workflow:")
        print("1. Query strategy engine classifies query against category hierarchy")
        print("2. LLM selects relevant categories at each level")
        print("3. System constructs category paths from root to leaf")
        print("4. Chunk retriever fetches content from leaf categories")
        print("5. Result constructor assembles comprehensive response")
        print("6. System returns chunks with metadata and performance metrics")
        print()

        # Show what the result structure would look like
        print("Result structure would include:")
        print("- chunks: List of retrieved text chunks with metadata")
        print("- responses: LLM call responses with timestamps and latency")
        print("- total_latency_ms: Complete processing time")
        print("- successful_paths: Number of successful category paths")
        print("- failed_paths: Number of failed category paths")
        print("- query_text: Original query for reference")
        print("- strategy_used: Strategy that was employed")
        print()


def demonstrate_query_result_structure():
    """Show the structure of query results."""

    print("=== Query Result Structure ===\n")

    print("A complete QueryResult contains:")
    print()

    print("Core Results:")
    print("- chunks: List[ChunkResult] - Retrieved text chunks")
    print("  - chunk_id: Unique identifier")
    print("  - text_content: The actual text content")
    print("  - category_path: Human-readable category path")
    print("  - ranked_relevance: Relevance score from LLM")
    print("  - created_at: When the chunk was created")
    print("  - source_id: Source document identifier")
    print()

    print("- responses: List[LLMCallResponse] - LLM interaction tracking")
    print("  - llm_output: Structured LLM response")
    print("  - timestamp: When the call was made")
    print("  - latency_ms: How long the call took")
    print()

    print("Metadata:")
    print("- total_latency_ms: Complete processing time")
    print("- total_chunks: Number of chunks retrieved")
    print("- successful_paths: Successful category paths")
    print("- failed_paths: Failed category paths")
    print("- dropped_paths: Paths that were dropped (optional)")
    print("- query_text: Original query")
    print("- strategy_used: Strategy employed")
    print("- timestamp: When processing started")
    print()


def demonstrate_error_handling():
    """Show error handling capabilities."""

    print("=== Error Handling ===\n")

    print("The query processing system handles various error scenarios:")
    print()

    print("1. Strategy Execution Errors:")
    print("   - LLM service unavailable")
    print("   - Invalid category selections")
    print("   - Timeout during classification")
    print()

    print("2. Chunk Retrieval Errors:")
    print("   - Database connection issues")
    print("   - Empty leaf categories")
    print("   - SQL execution errors")
    print()

    print("3. Validation Errors:")
    print("   - Invalid query parameters")
    print("   - Malformed strategy parameters")
    print("   - Result consistency issues")
    print()

    print("Error Recovery:")
    print("- Graceful degradation with partial results")
    print("- Detailed error messages for debugging")
    print("- Fallback logic for empty categories")
    print("- Comprehensive logging for troubleshooting")
    print()


def demonstrate_performance_features():
    """Show performance optimization features."""

    print("=== Performance Features ===\n")

    print("The system includes several performance optimizations:")
    print()

    print("1. Deterministic Ordering:")
    print("   - Consistent chunk ordering by created_at and chunk_id")
    print("   - Reproducible results across queries")
    print("   - Efficient pagination support")
    print()

    print("2. Batch Processing:")
    print("   - Multiple chunks processed per LLM call")
    print("   - Configurable batch sizes")
    print("   - Retry logic for failed batches")
    print()

    print("3. Caching and Optimization:")
    print("   - Category hierarchy caching")
    print("   - SQL query optimization")
    print("   - Connection pooling")
    print()

    print("4. Monitoring and Metrics:")
    print("   - Latency tracking for all operations")
    print("   - Success/failure rate monitoring")
    print("   - Performance statistics")
    print()


if __name__ == "__main__":
    print("Running complete query workflow examples...\n")

    # Run the async main function
    asyncio.run(main())

    # Run synchronous demonstrations
    demonstrate_query_result_structure()
    demonstrate_error_handling()
    demonstrate_performance_features()

    print("Examples completed!")
    print()
    print("To use this system in practice:")
    print("1. Set up your database with categories and chunks")
    print("2. Initialize a CategoryManager with your database session")
    print("3. Create a QueryProcessor with your CategoryManager")
    print("4. Call process_query() with your natural language queries")
    print("5. Process the returned QueryResult with chunks and metadata")
