"""
Database usage example for MemoirAI library.
"""

import asyncio
import os

from memoir_ai import Category, CategoryLimits, Chunk, ContextualHelper, MemoirAI
from memoir_ai.exceptions import ConfigurationError, DatabaseError

# Ensure local runs have a fallback API key for LLM initialization.
os.environ.setdefault("OPENAI_API_KEY", "test-key")


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
            model_name="gpt-5-nano",
            hierarchy_depth=3,
            chunk_min_tokens=300,
            chunk_max_tokens=500,
            batch_size=5,
            max_token_budget=40000,
        )
        print("✅ MemoirAI instance created with database!")
        print(f"   Database: {memoir.config.database_url}")

    except (ConfigurationError, DatabaseError) as e:
        print(f"❌ Error: {e}")
        return

    # Example 2: Direct database access
    print("\n2. Working with database models directly...")

    try:
        with memoir.db_manager.get_session() as session:
            # Helper to reuse existing categories when they already exist
            def merge_category_trees(source: Category, target: Category) -> None:
                """Merge two category subtrees that share the same logical node."""

                if source.id == target.id:
                    return

                # Move chunks attached to the source category onto the target
                session.query(Chunk).filter_by(category_id=source.id).update(
                    {Chunk.category_id: target.id}, synchronize_session=False
                )

                # Reattach or merge child categories recursively
                children = (
                    session.query(Category)
                    .filter_by(parent_id=source.id)
                    .order_by(Category.id)
                    .all()
                )
                for child in children:
                    existing_child = (
                        session.query(Category)
                        .filter_by(parent_id=target.id, name=child.name)
                        .order_by(Category.id)
                        .first()
                    )
                    if existing_child:
                        merge_category_trees(child, existing_child)
                        session.delete(child)
                    else:
                        child.parent_id = target.id

                session.flush()

            def get_or_create_category(
                *, name: str, level: int, parent: Category | None
            ) -> Category:
                parent_id = parent.id if parent else None
                categories = (
                    session.query(Category)
                    .filter_by(name=name, parent_id=parent_id)
                    .order_by(Category.id)
                    .all()
                )

                if categories:
                    keeper = categories[0]
                    for duplicate in categories[1:]:
                        merge_category_trees(duplicate, keeper)
                        session.delete(duplicate)

                    session.flush()
                    keeper.level = level
                    return keeper

                category = Category(name=name, level=level, parent_id=parent_id)
                session.add(category)
                session.flush()  # Ensure category.id is available for children
                return category

            # Create category hierarchy idempotently
            tech_category = get_or_create_category(
                name="Technology", level=1, parent=None
            )
            ai_category = get_or_create_category(
                name="Artificial Intelligence", level=2, parent=tech_category
            )
            nlp_category = get_or_create_category(
                name="Natural Language Processing", level=3, parent=ai_category
            )

            print("   Using category hierarchy:")
            print(f"   - {tech_category.name} (Level {tech_category.level})")
            print(f"   - {ai_category.name} (Level {ai_category.level})")
            print(f"   - {nlp_category.name} (Level {nlp_category.level})")

            # Create contextual helper
            helper = (
                session.query(ContextualHelper)
                .filter_by(source_id="ai_research_2024")
                .one_or_none()
            )
            if helper is None:
                helper = ContextualHelper(
                    source_id="ai_research_2024",
                    helper_text="This document discusses recent advances in AI and NLP research, "
                    "including transformer models and large language models.",
                    token_count=20,
                    is_user_provided=False,
                )
                session.add(helper)
            else:
                helper.helper_text = (
                    "This document discusses recent advances in AI and NLP research, "
                    "including transformer models and large language models."
                )
                helper.token_count = 20
                helper.is_user_provided = False

            # Create sample chunks
            # Ensure sample chunks are not duplicated on repeated runs
            def get_or_create_chunk(
                *,
                content: str,
                token_count: int,
                category: Category,
                source_id: str,
            ) -> Chunk:
                chunks = (
                    session.query(Chunk)
                    .filter_by(
                        content=content,
                        category_id=category.id,
                        source_id=source_id,
                    )
                    .order_by(Chunk.id)
                    .all()
                )
                if chunks:
                    chunk = chunks[0]
                    for duplicate in chunks[1:]:
                        session.delete(duplicate)
                    chunk.token_count = token_count
                    session.flush()
                    return chunk

                chunk = Chunk(
                    content=content,
                    token_count=token_count,
                    category_id=category.id,
                    source_id=source_id,
                )
                session.add(chunk)
                return chunk

            chunk1 = get_or_create_chunk(
                content="Transformer models have revolutionized natural language processing "
                "by introducing the attention mechanism that allows models to focus "
                "on relevant parts of the input sequence.",
                token_count=25,
                category=nlp_category,
                source_id="ai_research_2024",
            )

            chunk2 = get_or_create_chunk(
                content="Large language models like GPT and BERT have shown remarkable "
                "capabilities in understanding and generating human-like text.",
                token_count=18,
                category=ai_category,
                source_id="ai_research_2024",
            )

            # Create category limits
            limits = session.get(CategoryLimits, 1)
            if limits is None:
                limits = CategoryLimits(level=1, max_categories=50)
                session.add(limits)
            else:
                limits.max_categories = 50

            print(f"   Added contextual helper for source: {helper.source_id}")
            print(
                f"   Added or updated {len([chunk1, chunk2])} text chunks for demo content"
            )
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
        with memoir.db_manager.get_session() as session:
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
        info = memoir.db_manager.get_table_info()
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
        with memoir.db_manager.get_session() as session:
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
                session.rollback()

    except DatabaseError as e:
        print(f"❌ Constraint test error: {e}")

    print("\n🎉 Database functionality demonstration complete!")
    print("\n📝 Next steps:")
    print("   - Text chunking and processing")
    print("   - LLM integration for classification")
    print("   - Query processing and result aggregation")

    # Cleanup
    memoir.db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
