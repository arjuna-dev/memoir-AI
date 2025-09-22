"""
Example usage of the Query Strategy Engine.

This example demonstrates how to use the different query strategies
to traverse the category hierarchy and find relevant content.
"""

import asyncio

from memoir_ai.database.models import Category
from memoir_ai.query import (
    QueryStrategy,
    QueryStrategyEngine,
    create_query_strategy_engine,
    validate_strategy_params,
)


async def main():
    """Demonstrate query strategy engine usage."""

    print("=== MemoirAI Query Strategy Engine Example ===\n")

    # Create a query strategy engine
    # In a real application, this would use your actual database
    engine = create_query_strategy_engine(
        max_depth=3, model_name="openai:gpt-4", category_limits={1: 20, 2: 50, 3: 100}
    )

    print("Created QueryStrategyEngine with:")
    print(f"- Max depth: {engine.hierarchy_manager.max_depth}")
    print(f"- Model: {engine.model_name}")
    print(
        f"- Category limits: {engine.hierarchy_manager.category_limits.per_level_limits}"
    )
    print()

    # Example query
    query_text = "machine learning algorithms for natural language processing"
    print(f"Query: '{query_text}'")
    print()

    # Demonstrate different strategies
    strategies_to_test = [
        (QueryStrategy.ONE_SHOT, {}),
        (QueryStrategy.WIDE_BRANCH, {"n": 3}),
        (QueryStrategy.ZOOM_IN, {"n": 4, "n2": 1}),
        (QueryStrategy.BRANCH_OUT, {"n": 1, "n2": 2}),
    ]

    for strategy, params in strategies_to_test:
        print(f"--- {strategy.value.upper()} STRATEGY ---")

        # Get strategy info
        info = engine.get_strategy_info(strategy)
        print(f"Description: {info['description']}")
        print(f"Use case: {info['use_case']}")

        # Validate parameters
        try:
            validated_params = validate_strategy_params(strategy, params)
            print(f"Parameters: {validated_params}")
        except Exception as e:
            print(f"Parameter validation error: {e}")
            continue

        # Note: In a real application, you would execute the strategy here
        # For this example, we'll just show the setup
        print("Strategy configured successfully!")
        print(
            f"Would execute: engine.execute_strategy('{query_text}', {strategy}, {validated_params})"
        )
        print()

    # Demonstrate parameter validation
    print("--- PARAMETER VALIDATION EXAMPLES ---")

    # Valid parameters
    try:
        params = validate_strategy_params(QueryStrategy.WIDE_BRANCH, {"n": 5})
        print(f"✓ Valid WIDE_BRANCH params: {params}")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Invalid parameters
    try:
        params = validate_strategy_params(QueryStrategy.ZOOM_IN, {"n": -1})
        print(f"✓ Invalid params accepted: {params}")
    except Exception as e:
        print(f"✗ Correctly rejected invalid params: {e}")

    print()

    # Show strategy information
    print("--- AVAILABLE STRATEGIES ---")
    for strategy in QueryStrategy:
        info = engine.get_strategy_info(strategy)
        print(f"{strategy.value}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Parameters: {info.get('parameters', 'None')}")
        print(f"  Use case: {info['use_case']}")
        print()


def demonstrate_category_path():
    """Demonstrate CategoryPath functionality."""
    print("=== CategoryPath Example ===\n")

    # Create sample categories
    categories = [
        Category(id=1, name="Technology", level=1, parent_id=None),
        Category(id=2, name="Artificial Intelligence", level=2, parent_id=1),
        Category(id=3, name="Machine Learning", level=3, parent_id=2),
    ]

    # Create a category path
    from memoir_ai.query import CategoryPath

    path = CategoryPath(path=categories, ranked_relevance=8)

    print(f"Category Path: {path.path_string}")
    print(f"Depth: {path.depth}")
    print(f"Leaf category: {path.leaf_category.name}")
    print(f"Ranked relevance: {path.ranked_relevance}")
    print()

    # Demonstrate path equality and hashing
    path2 = CategoryPath(path=categories, ranked_relevance=5)  # Different relevance
    print(f"Paths equal (same categories): {path == path2}")
    print(f"Same hash (for deduplication): {hash(path) == hash(path2)}")
    print()


def demonstrate_query_classification_result():
    """Demonstrate QueryClassificationResult Pydantic model."""
    print("=== QueryClassificationResult Example ===\n")

    from memoir_ai.query import QueryClassificationResult

    # Create classification result
    result = QueryClassificationResult(category="Machine Learning", ranked_relevance=5)

    print(f"Category: {result.category}")
    print(f"Ranked relevance: {result.ranked_relevance}")
    print(f"JSON representation: {result.model_dump_json()}")
    print()


if __name__ == "__main__":
    print("Running Query Strategy Engine examples...\n")

    # Run the async main function
    asyncio.run(main())

    # Run synchronous examples
    demonstrate_category_path()
    demonstrate_query_classification_result()

    print("Examples completed!")
