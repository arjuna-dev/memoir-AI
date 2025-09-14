"""
Database usage example for MemoirAI library.
"""

import asyncio
from memoir_ai import MemoirAI, Category, Chunk, ContextualHelper, CategoryLimits
from memoir_ai.exceptions import ConfigurationError, DatabaseError


async def main():
    """Demonstrate MemoirAI database functionality."""

    print("🗄️  MemoirAI Database Usage Example")
    print("=" * 40)

    # Example 1: Initialize with database
    print("\n1. Creating MemoirAI instance with database...")
    try:
        memoir = MemoirAI(
            database_url="sqlite:///example_database.db",
            llm_provider="openai",
            model_name="gpt-4",
            hierarchy_depth=3,
            chunk_min_tokens=300,
            chunk_max_tokens=500,
            batch_size=5,
            max_token_budget=4000,
        )
        print("✅ MemoirAI instance created with database!")
        print(f"   Database: {memoir.config.database_url}")

    except (ConfigurationError, DatabaseError) as e:
        print(f"❌ Error: {e}")
        return

    # Example 2: Direct database access
    print("\n2. Working with database models directly...")

    try:
        with memoir._db_manager.get_session() as session:
            # Create category hierarchy
            tech_category = Category(name="Technology", level=1, parent_id=None)
            session.add(tech_category)
            session.flush()  # Get the ID

            ai_category = Category(
                name="Artificial Intelligence", level=2, parent_id=tech_category.id
            )
            session.add(ai_category)
            session.flush()

            nlp_category = Category(
                name="Natural Language Processing", level=3, parent_id=ai_category.id
            )
            session.add(nlp_category)
            session.flush()

            print(f"   Created category hierarchy:")
            print(f"   - {tech_category.name} (Level {tech_category.level})")
            print(f"   - {ai_category.name} (Level {ai_category.level})")
            print(f"   - {nlp_category.name} (Level {nlp_category.level})")

            # Create contextual helper
            helper = ContextualHelper(
                source_id="ai_research_2024",
                helper_text="This document discusses recent advances in AI and NLP research, "
                "including transformer models and large language models.",
                token_count=20,
                is_user_provided=False,
            )
            session.add(helper)

            # Create sample chunks
            chunk1 = Chunk(
                content="Transformer models have revolutionized natural language processing "
                "by introducing the attention mechanism that allows models to focus "
                "on relevant parts of the input sequence.",
                token_count=25,
                category_id=nlp_category.id,
                source_id="ai_research_2024",
            )

            chunk2 = Chunk(
                content="Large language models like GPT and BERT have shown remarkable "
                "capabilities in understanding and generating human-like text.",
                token_count=18,
                category_id=ai_category.id,
                source_id="ai_research_2024",
            )

            session.add_all([chunk1, chunk2])

            # Create category limits
            limits = CategoryLimits(level=1, max_categories=50)
            session.add(limits)

            print(f"   Added contextual helper for source: {helper.source_id}")
            print(f"   Added {2} text chunks")
            print(
                f"   Set category limit: {limits.max_categories} for level {limits.level}"
            )

        print("✅ Database operations completed successfully!")

    except DatabaseError as e:
        print(f"❌ Database error: {e}")
        return

    # Example 3: Query the database
    print("\n3. Querying the database...")

    try:
        with memoir._db_manager.get_session() as session:
            # Get all categories
            categories = session.query(Category).all()
            print(f"   Total categories: {len(categories)}")

            # Get category hierarchy
            for category in categories:
                if category.level == 1:
                    print(f"   📁 {category.name}")
                    for child in category.children:
                        print(f"      📁 {child.name}")
                        for grandchild in child.children:
                            print(f"         📁 {grandchild.name}")

            # Get chunks with their category paths
            chunks = session.query(Chunk).all()
            print(f"\n   Text chunks ({len(chunks)}):")
            for chunk in chunks:
                path = chunk.category.get_path_string()
                content_preview = (
                    chunk.content[:60] + "..."
                    if len(chunk.content) > 60
                    else chunk.content
                )
                print(f"   📄 [{path}] {content_preview}")
                print(f"      Tokens: {chunk.token_count}, Source: {chunk.source_id}")

            # Get contextual helpers
            helpers = session.query(ContextualHelper).all()
            print(f"\n   Contextual helpers ({len(helpers)}):")
            for helper in helpers:
                helper_preview = (
                    helper.helper_text[:80] + "..."
                    if len(helper.helper_text) > 80
                    else helper.helper_text
                )
                print(f"   🔍 {helper.source_id}: {helper_preview}")
                print(
                    f"      Tokens: {helper.token_count}, User provided: {helper.is_user_provided}"
                )

    except DatabaseError as e:
        print(f"❌ Query error: {e}")
        return

    # Example 4: Database info
    print("\n4. Database information...")

    try:
        info = memoir._db_manager.get_table_info()
        print("   Table statistics:")
        for table_name, table_info in info.items():
            if "row_count" in table_info:
                print(f"   - {table_name}: {table_info['row_count']} rows")
            else:
                print(
                    f"   - {table_name}: Error - {table_info.get('error', 'Unknown')}"
                )

    except DatabaseError as e:
        print(f"❌ Info error: {e}")

    # Example 5: Constraint validation
    print("\n5. Testing database constraints...")

    try:
        with memoir._db_manager.get_session() as session:
            # Try to create a category with invalid level
            try:
                invalid_category = Category(name="Invalid", level=0, parent_id=None)
                session.add(invalid_category)
                session.flush()
                print("❌ This should have failed!")
            except Exception as e:
                print("✅ Level validation working: Invalid level rejected")

            session.rollback()  # Reset session

            # Try to create duplicate contextual helper
            try:
                duplicate_helper = ContextualHelper(
                    source_id="ai_research_2024",  # Same as existing
                    helper_text="Duplicate helper",
                    token_count=3,
                )
                session.add(duplicate_helper)
                session.flush()
                print("❌ This should have failed!")
            except Exception as e:
                print("✅ Unique constraint working: Duplicate source_id rejected")

    except DatabaseError as e:
        print(f"❌ Constraint test error: {e}")

    print("\n🎉 Database functionality demonstration complete!")
    print("\n📝 Next steps:")
    print("   - Text chunking and processing")
    print("   - LLM integration for classification")
    print("   - Query processing and result aggregation")

    # Cleanup
    memoir._db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
