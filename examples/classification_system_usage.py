#!/usr/bin/env python3
"""
Example usage of the complete Classification System.

This example demonstrates how to use the CategoryManager and
IterativeClassificationWorkflow together for hierarchical text classification.
"""

import asyncio
from typing import List
from unittest.mock import Mock

from memoir_ai.classification import (
    CategoryLimitConfig,
    CategoryManager,
    IterativeClassificationWorkflow,
    create_category_manager,
    create_iterative_classifier,
    validate_hierarchy_depth,
)
from memoir_ai.database.models import Category
from memoir_ai.text_processing.chunker import TextChunk


def create_sample_chunks() -> List[TextChunk]:
    """Create sample text chunks for demonstration."""
    sample_texts = [
        "Machine learning algorithms are revolutionizing data analysis and prediction tasks across industries.",
        "Climate change research indicates rising global temperatures and changing weather patterns worldwide.",
        "Quantum computing breakthrough promises exponential speedup for certain computational problems.",
        "Renewable energy adoption accelerates as solar and wind costs continue to decline rapidly.",
        "Artificial intelligence applications in medical diagnosis show promising accuracy improvements over traditional methods.",
        "Blockchain technology enables secure decentralized transactions without traditional banking intermediaries.",
        "Space exploration missions reveal new insights about planetary formation and potential for life.",
        "Gene therapy advances offer hope for treating previously incurable genetic disorders effectively.",
    ]

    chunks = []
    for i, text in enumerate(sample_texts):
        chunk = TextChunk(
            content=text,
            token_count=len(text.split()),  # Simple token approximation
            start_position=i * 100,
            end_position=(i + 1) * 100 - 1,
        )
        chunks.append(chunk)

    return chunks


def demonstrate_category_limit_config():
    """Demonstrate CategoryLimitConfig usage."""
    print("=== Category Limit Configuration ===\n")

    # Default configuration
    print("1. Default Configuration:")
    default_config = CategoryLimitConfig()
    print(f"   Global limit: {default_config.global_limit}")
    print(f"   Per-level limits: {default_config.per_level_limits}")
    print(f"   Limit for level 1: {default_config.get_limit_for_level(1)}")
    print(f"   Limit for level 3: {default_config.get_limit_for_level(3)}")
    print()

    # Custom global configuration
    print("2. Custom Global Configuration:")
    global_config = CategoryLimitConfig(global_limit=50)
    print(f"   Global limit: {global_config.global_limit}")
    print(f"   Limit for level 1: {global_config.get_limit_for_level(1)}")
    print(f"   Limit for level 2: {global_config.get_limit_for_level(2)}")
    print()

    # Per-level configuration
    print("3. Per-Level Configuration:")
    per_level_config = CategoryLimitConfig(
        global_limit=100, per_level_limits={1: 10, 2: 25, 3: 50}
    )
    print(f"   Global limit: {per_level_config.global_limit}")
    print(f"   Per-level limits: {per_level_config.per_level_limits}")
    print(
        f"   Limit for level 1: {per_level_config.get_limit_for_level(1)}"
    )  # 10 (override)
    print(
        f"   Limit for level 2: {per_level_config.get_limit_for_level(2)}"
    )  # 25 (override)
    print(
        f"   Limit for level 3: {per_level_config.get_limit_for_level(3)}"
    )  # 50 (override)
    print(
        f"   Limit for level 4: {per_level_config.get_limit_for_level(4)}"
    )  # 100 (global)
    print()


def demonstrate_category_manager():
    """Demonstrate CategoryManager functionality."""
    print("=== Category Manager Demonstration ===\n")

    # Create mock database session
    mock_session = Mock()
    mock_session.query.return_value = mock_session
    mock_session.filter.return_value = mock_session
    mock_session.order_by.return_value = mock_session
    mock_session.all.return_value = []
    mock_session.first.return_value = None
    mock_session.scalar.return_value = 0
    mock_session.add = Mock()
    mock_session.flush = Mock()

    # Create category manager
    print("1. Creating Category Manager:")
    try:
        manager = CategoryManager(
            db_session=mock_session,
            hierarchy_depth=3,
            category_limits={1: 5, 2: 10, 3: 20},
        )
        print(f"   Hierarchy depth: {manager.hierarchy_depth}")
        print(f"   Limit for level 1: {manager.get_category_limit(1)}")
        print(f"   Limit for level 2: {manager.get_category_limit(2)}")
        print(f"   Limit for level 3: {manager.get_category_limit(3)}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # Demonstrate validation
    print("2. Configuration Validation:")
    valid_depths = [1, 3, 50, 100]
    invalid_depths = [0, -1, 101, 150]

    for depth in valid_depths:
        is_valid = validate_hierarchy_depth(depth)
        print(f"   Depth {depth}: {'Valid' if is_valid else 'Invalid'}")

    for depth in invalid_depths:
        is_valid = validate_hierarchy_depth(depth)
        print(f"   Depth {depth}: {'Valid' if is_valid else 'Invalid'}")
    print()

    # Demonstrate category operations
    print("3. Category Operations:")
    try:
        # Mock some existing categories
        existing_categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="Science", level=1, parent_id=None),
        ]
        mock_session.all.return_value = existing_categories

        categories = manager.get_existing_categories(1)
        print(f"   Retrieved {len(categories)} level 1 categories")

        # Check if can create new category
        mock_session.scalar.return_value = 3  # 3 existing categories
        can_create = manager.can_create_category(1)
        print(f"   Can create new level 1 category: {can_create} (3/5 used)")

        # Simulate at limit
        mock_session.scalar.return_value = 5  # 5 existing categories (at limit)
        can_create = manager.can_create_category(1)
        print(f"   Can create new level 1 category: {can_create} (5/5 used)")

    except Exception as e:
        print(f"   Error: {e}")
    print()


def demonstrate_hierarchy_validation():
    """Demonstrate hierarchy validation."""
    print("=== Hierarchy Validation ===\n")

    mock_session = Mock()
    mock_session.query.return_value = mock_session
    mock_session.filter.return_value = mock_session
    mock_session.all.return_value = []

    manager = CategoryManager(mock_session, hierarchy_depth=3)

    # Valid hierarchy
    print("1. Valid Hierarchy:")
    valid_categories = [
        Category(id=1, name="Technology", level=1, parent_id=None),
        Category(id=2, name="AI", level=2, parent_id=1),
        Category(id=3, name="ML", level=3, parent_id=2),
    ]
    mock_session.all.return_value = valid_categories

    # Mock get_category_by_id
    def mock_get_category_by_id(cat_id):
        categories = {
            1: valid_categories[0],
            2: valid_categories[1],
            3: valid_categories[2],
        }
        return categories.get(cat_id)

    manager.get_category_by_id = mock_get_category_by_id

    errors = manager.validate_category_hierarchy()
    print(f"   Validation errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"     - {error}")
    else:
        print("     No errors found - hierarchy is valid!")
    print()

    # Invalid hierarchy
    print("2. Invalid Hierarchy:")
    invalid_categories = [
        Category(
            id=1, name="Technology", level=1, parent_id=999
        ),  # Level 1 with parent
        Category(id=2, name="AI", level=5, parent_id=1),  # Invalid level
    ]
    mock_session.all.return_value = invalid_categories

    errors = manager.validate_category_hierarchy()
    print(f"   Validation errors: {len(errors)}")
    for error in errors:
        print(f"     - {error}")
    print()


async def demonstrate_iterative_classification():
    """Demonstrate iterative classification workflow."""
    print("=== Iterative Classification Workflow ===\n")

    # Create mock components
    mock_session = Mock()
    mock_session.add = Mock()
    mock_session.flush = Mock()

    mock_category_manager = Mock()
    mock_category_manager.hierarchy_depth = 3
    mock_category_manager.limits = Mock()
    mock_category_manager.limits.global_limit = 128
    mock_category_manager.is_leaf_category.return_value = True

    print("1. Workflow Configuration:")
    try:
        # This would normally require proper LLM API setup
        print("   Creating workflow (requires LLM API configuration)...")
        print("   Model: openai:gpt-4")
        print("   Hierarchy depth: 3")
        print("   Batch processing: enabled")
        print("   Batch size: 5")
    except Exception as e:
        print(f"   Note: {e}")
    print()

    print("2. Classification Process:")
    chunks = create_sample_chunks()[:3]  # Use first 3 chunks

    print(f"   Input: {len(chunks)} text chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"     {i}. {chunk.content[:60]}...")

    print("\n   Expected Classification Flow:")
    print("     Level 1: Determine broad category (Technology, Science, etc.)")
    print("     Level 2: Determine sub-category (AI, Climate, etc.)")
    print("     Level 3: Determine specific category (ML, Algorithms, etc.)")
    print("     Storage: Link chunk to final leaf category")
    print()

    print("3. Expected Results:")
    expected_classifications = [
        ["Technology", "AI", "Machine Learning"],
        ["Science", "Climate", "Research"],
        ["Technology", "Computing", "Quantum"],
    ]

    for i, (chunk, classification) in enumerate(
        zip(chunks, expected_classifications), 1
    ):
        print(f"   Chunk {i}: {' → '.join(classification)}")

    print(f"\n   Total LLM calls: {len(chunks) * 3} (3 levels × {len(chunks)} chunks)")
    print("   Storage: 3 chunks linked to leaf categories")
    print()


def demonstrate_metrics_and_monitoring():
    """Demonstrate metrics and monitoring capabilities."""
    print("=== Metrics and Monitoring ===\n")

    # Simulate workflow metrics
    print("1. Workflow Metrics:")
    sample_metrics = {
        "total_workflows": 5,
        "total_chunks": 25,
        "total_successful": 23,
        "total_failed": 2,
        "success_rate": 0.92,
        "average_latency_ms": 1850.0,
        "total_llm_calls": 69,
        "average_llm_calls_per_chunk": 2.76,
        "average_levels_per_chunk": 2.8,
    }

    print(f"   Total workflows: {sample_metrics['total_workflows']}")
    print(f"   Total chunks processed: {sample_metrics['total_chunks']}")
    print(f"   Success rate: {sample_metrics['success_rate']:.1%}")
    print(f"   Average latency: {sample_metrics['average_latency_ms']:.0f}ms")
    print(f"   Total LLM calls: {sample_metrics['total_llm_calls']}")
    print(
        f"   Average LLM calls per chunk: {sample_metrics['average_llm_calls_per_chunk']:.2f}"
    )
    print(
        f"   Average levels per chunk: {sample_metrics['average_levels_per_chunk']:.1f}"
    )
    print()

    # Simulate category statistics
    print("2. Category Statistics:")
    sample_stats = {
        "total_categories": 45,
        "categories_by_level": {1: 8, 2: 15, 3: 22},
        "max_depth": 3,
        "leaf_categories": 22,
        "categories_at_limit": [2],  # Level 2 at limit
    }

    print(f"   Total categories: {sample_stats['total_categories']}")
    print("   Categories by level:")
    for level, count in sample_stats["categories_by_level"].items():
        print(f"     Level {level}: {count} categories")
    print(f"   Max depth used: {sample_stats['max_depth']}")
    print(f"   Leaf categories: {sample_stats['leaf_categories']}")
    if sample_stats["categories_at_limit"]:
        print(f"   Levels at limit: {sample_stats['categories_at_limit']}")
    else:
        print("   No levels at limit")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("=== Error Handling Examples ===\n")

    mock_session = Mock()

    print("1. Configuration Errors:")

    # Invalid hierarchy depth
    try:
        CategoryManager(mock_session, hierarchy_depth=0)
    except Exception as e:
        print(f"   Invalid hierarchy depth: {e}")

    # Invalid category limits
    try:
        CategoryManager(mock_session, category_limits=-1)
    except Exception as e:
        print(f"   Invalid category limits: {e}")

    print()

    print("2. Validation Errors:")

    # Invalid hierarchy depth validation
    invalid_depths = [0, -1, 101, "3", None]
    for depth in invalid_depths:
        is_valid = validate_hierarchy_depth(depth)
        print(f"   Depth {depth}: {'Valid' if is_valid else 'Invalid'}")

    print()

    print("3. Category Creation Errors:")
    manager = CategoryManager(mock_session, hierarchy_depth=3)

    # Empty name
    try:
        manager.create_category("", 1)
    except Exception as e:
        print(f"   Empty name: {type(e).__name__}")

    # Invalid level
    try:
        manager.create_category("Test", 0)
    except Exception as e:
        print(f"   Invalid level: {type(e).__name__}")

    # Level 1 with parent
    try:
        manager.create_category("Test", 1, parent_id=1)
    except Exception as e:
        print(f"   Level 1 with parent: {type(e).__name__}")

    print()


def demonstrate_integration_example():
    """Demonstrate complete integration example."""
    print("=== Complete Integration Example ===\n")

    print("This example shows how the components work together:")
    print()

    print("1. Setup:")
    print("   - Create database session")
    print("   - Initialize CategoryManager with hierarchy depth 3")
    print("   - Configure category limits: {1: 10, 2: 25, 3: 50}")
    print("   - Create IterativeClassificationWorkflow")
    print()

    print("2. Text Processing:")
    print("   - Input: Research paper about AI and machine learning")
    print("   - Chunking: Split into 15 meaningful chunks")
    print("   - Context: Generate contextual helper from metadata")
    print()

    print("3. Classification Process:")
    print("   - Level 1: Classify into 'Technology' (existing category)")
    print("   - Level 2: Classify into 'Artificial Intelligence' (create new)")
    print("   - Level 3: Classify into 'Machine Learning' (create new)")
    print("   - Storage: Link all chunks to leaf categories")
    print()

    print("4. Results:")
    print("   - 15 chunks successfully classified")
    print("   - 3 new categories created")
    print("   - 45 LLM calls total (3 per chunk)")
    print("   - Average processing time: 2.1 seconds")
    print("   - All chunks stored and retrievable")
    print()

    print("5. Benefits:")
    print("   - Hierarchical organization for better retrieval")
    print("   - Category reuse prevents duplication")
    print("   - Configurable limits control growth")
    print("   - Comprehensive metrics for optimization")
    print("   - Robust error handling and recovery")
    print()


def main():
    """Run all demonstrations."""
    print("=== MemoirAI Classification System Example ===\n")

    demonstrate_category_limit_config()
    demonstrate_category_manager()
    demonstrate_hierarchy_validation()

    # Run async demonstration
    print("Running async classification demonstration...")
    asyncio.run(demonstrate_iterative_classification())

    demonstrate_metrics_and_monitoring()
    demonstrate_error_handling()
    demonstrate_integration_example()

    print("=== Key Features Summary ===")
    print("• Hierarchical category management (1-100 levels)")
    print("• Configurable category limits (global and per-level)")
    print("• Iterative LLM-based classification workflow")
    print("• Automatic category reuse and duplicate prevention")
    print("• Comprehensive validation and error handling")
    print("• Detailed metrics and performance monitoring")
    print("• Database transaction management")
    print("• Leaf-level chunk storage for optimal retrieval")
    print()

    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
