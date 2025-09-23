"""
Database migration usage example for MemoirAI library.
"""

import asyncio
import os
import tempfile

from memoir_ai import MemoirAI
from memoir_ai.config import MemoirAIConfig
from memoir_ai.database import Category, Chunk, MigrationManager
from memoir_ai.exceptions import DatabaseError


async def main():
    """Demonstrate MemoirAI migration system functionality."""

    print("🔄 MemoirAI Migration System Example")
    print("=" * 40)

    # Create a temporary database file for demonstration
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Example 1: Initialize new database with migrations
        print("\n1. Initializing new database with migration system...")

        config = MemoirAIConfig(database_url=f"sqlite:///{db_path}")
        migration_manager = MigrationManager(config)

        # Initialize database
        init_results = migration_manager.initialize_database(create_tables=True)

        print("✅ Database initialized successfully!")
        print(f"   Database URL: {init_results['database_url']}")
        print(f"   Tables created: {init_results['tables_created']}")
        print(f"   Migration initialized: {init_results['migration_initialized']}")
        print(f"   Current revision: {init_results['current_revision']}")
        print(f"   Pending migrations: {len(init_results['pending_migrations'])}")

        # Example 2: Check migration status
        print("\n2. Checking migration status...")

        current_revision = migration_manager.get_current_revision()
        pending_migrations = migration_manager.get_pending_migrations()
        migration_history = migration_manager.get_migration_history()

        print(f"   Current revision: {current_revision}")
        print(f"   Pending migrations: {len(pending_migrations)}")
        print(f"   Migration history: {len(migration_history)} revisions")

        if migration_history:
            print("   Migration history:")
            for i, revision in enumerate(migration_history[:3]):  # Show first 3
                print(f"     {i+1}. {revision['revision']}: {revision['message']}")

        # Example 3: Validate database schema
        print("\n3. Validating database schema...")

        validation = migration_manager.validate_database_schema()

        print(f"   Schema is valid: {validation['is_valid']}")
        print(f"   Tables exist: {validation['tables_exist']}")
        print(f"   Current revision: {validation['current_revision']}")
        print(f"   Pending migrations: {len(validation['pending_migrations'])}")

        if validation["table_info"]:
            print("   Table statistics:")
            for table_name, info in validation["table_info"].items():
                if "row_count" in info:
                    print(f"     - {table_name}: {info['row_count']} rows")

        # Example 4: Add some data and test database functionality
        print("\n4. Testing database functionality with migration system...")

        with migration_manager.db_manager.get_session() as session:
            # Create test data
            tech_category = Category(name="Technology", level=1, parent_id=None)
            session.add(tech_category)
            session.flush()

            ai_category = Category(name="AI", level=2, parent_id=tech_category.id)
            session.add(ai_category)
            session.flush()

            chunk = Chunk(
                content="This is a test chunk about artificial intelligence and machine learning.",
                token_count=12,
                category_id=ai_category.id,
                source_id="migration_test",
            )
            session.add(chunk)

        print("✅ Test data added successfully!")

        # Verify data
        with migration_manager.db_manager.get_session() as session:
            categories = session.query(Category).all()
            chunks = session.query(Chunk).all()

            print(f"   Categories created: {len(categories)}")
            print(f"   Chunks created: {len(chunks)}")

            for category in categories:
                print(f"     📁 {category.name} (Level {category.level})")

            for chunk in chunks:
                path = chunk.category.get_path_string()
                content_preview = (
                    chunk.content[:50] + "..."
                    if len(chunk.content) > 50
                    else chunk.content
                )
                print(f"     📄 [{path}] {content_preview}")

        migration_manager.close()

        # Example 5: Reopen database and verify persistence
        print("\n5. Reopening database to verify persistence...")

        migration_manager2 = MigrationManager(config)

        # Should detect existing database
        init_results2 = migration_manager2.initialize_database(create_tables=True)

        print(f"   Tables created on reopen: {init_results2['tables_created']}")
        print(f"   Current revision: {init_results2['current_revision']}")

        # Verify data persisted
        with migration_manager2.db_manager.get_session() as session:
            categories = session.query(Category).all()
            chunks = session.query(Chunk).all()

            print(f"   Persisted categories: {len(categories)}")
            print(f"   Persisted chunks: {len(chunks)}")

        # Example 6: Integration with MemoirAI class
        print("\n6. Integration with MemoirAI class...")

        memoir = MemoirAI(database_url=f"sqlite:///{db_path}")

        # The MemoirAI class should work with the existing database
        with memoir.db_manager.get_session() as session:
            categories = session.query(Category).all()
            print(f"   MemoirAI can access {len(categories)} existing categories")

        # Example 7: Database reset demonstration
        print("\n7. Database reset functionality...")

        print("   Current data before reset:")
        table_info_before = migration_manager2.db_manager.get_table_info()
        for table_name, info in table_info_before.items():
            if "row_count" in info and info["row_count"] > 0:
                print(f"     - {table_name}: {info['row_count']} rows")

        # Reset database
        reset_results = migration_manager2.reset_database()

        print("✅ Database reset successfully!")
        print(f"   Reset successful: {reset_results['success']}")
        print(f"   New revision: {reset_results['current_revision']}")

        # Verify reset
        table_info_after = migration_manager2.db_manager.get_table_info()
        print("   Data after reset:")
        for table_name, info in table_info_after.items():
            if "row_count" in info:
                print(f"     - {table_name}: {info['row_count']} rows")

        migration_manager2.close()

        print("\n🎉 Migration system demonstration complete!")
        print("\n📋 Key Features Demonstrated:")
        print("   ✅ Automatic database initialization")
        print("   ✅ Migration tracking and versioning")
        print("   ✅ Schema validation")
        print("   ✅ Database persistence across sessions")
        print("   ✅ Integration with MemoirAI core")
        print("   ✅ Database reset functionality")
        print("   ✅ Error handling and recovery")

    except DatabaseError as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up temporary database: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
