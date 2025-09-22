"""
Basic usage example for MemoirAI library.
"""

import asyncio

from memoir_ai import MemoirAI, PromptLimitingStrategy, QueryStrategy
from memoir_ai.exceptions import ConfigurationError


async def main():
    """Demonstrate basic MemoirAI usage."""

    print("🚀 MemoirAI Basic Usage Example")
    print("=" * 40)

    # Example 1: Valid configuration
    print("\n1. Creating MemoirAI instance with valid configuration...")
    try:
        memoir = MemoirAI(
            database_url="sqlite:///example.db",
            llm_provider="openai",
            model_name="gpt-4",
            hierarchy_depth=3,
            chunk_min_tokens=300,
            chunk_max_tokens=500,
            batch_size=5,
            max_token_budget=4000,
        )
        print("✅ MemoirAI instance created successfully!")
        print(f"   Database: {memoir.config.database_url}")
        print(f"   Model: {memoir.config.model_name}")
        print(f"   Hierarchy depth: {memoir.config.hierarchy_depth}")
        print(
            f"   Token range: {memoir.config.chunk_min_tokens}-{memoir.config.chunk_max_tokens}"
        )

    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Example 2: Configuration validation
    print("\n2. Testing configuration validation...")
    try:
        invalid_memoir = MemoirAI(
            database_url="sqlite:///test.db",
            max_token_budget=350,  # Too small for chunk_min_tokens=300
            chunk_min_tokens=300,
        )
        print("❌ This should have failed!")
    except ConfigurationError as e:
        print("✅ Configuration validation working correctly:")
        print(f"   Error: {str(e)[:80]}...")

    # Example 3: Environment variable configuration
    print("\n3. Configuration supports environment variables:")
    print("   Set MEMOIR_DATABASE_URL, MEMOIR_MODEL_NAME, etc.")
    print("   to override default configuration values.")

    # Example 4: Category limits configuration
    print("\n4. Category limits configuration examples:")

    # Global limit
    memoir_global = MemoirAI(
        database_url="sqlite:///global.db", max_categories_per_level=50
    )
    print(
        f"   Global limit: {memoir_global.config.get_category_limit(1)} categories per level"
    )

    # Per-level limits
    memoir_per_level = MemoirAI(
        database_url="sqlite:///per_level.db",
        hierarchy_depth=3,
        max_categories_per_level={1: 10, 2: 20, 3: 30},
    )
    print(
        f"   Per-level limits: L1={memoir_per_level.config.get_category_limit(1)}, "
        f"L2={memoir_per_level.config.get_category_limit(2)}, "
        f"L3={memoir_per_level.config.get_category_limit(3)}"
    )

    # Example 5: Future functionality preview
    print("\n5. Future functionality (not yet implemented):")
    print("   📝 Text ingestion with automatic chunking and classification")
    print("   🔍 Natural language queries with multiple strategies")
    print("   📊 Result aggregation with pruning and summarization")
    print("   🗂️  Category tree management and exploration")

    try:
        # These will raise NotImplementedError for now
        await memoir.ingest_text("Sample text", "source_1")
    except NotImplementedError:
        print("   ⏳ Ingestion functionality coming in next tasks...")

    try:
        await memoir.query("What is AI?")
    except NotImplementedError:
        print("   ⏳ Query functionality coming in next tasks...")

    print("\n🎉 Basic setup complete! Ready for implementation of core features.")


if __name__ == "__main__":
    asyncio.run(main())
