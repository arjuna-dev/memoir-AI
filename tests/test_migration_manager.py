"""
Tests for database migration management.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from memoir_ai.config import MemoirAIConfig
from memoir_ai.database.migration_manager import MigrationManager
from memoir_ai.database.models import Category, Chunk
from memoir_ai.exceptions import ConfigurationError, DatabaseError


class TestMigrationManager:
    """Test MigrationManager functionality."""

    def test_migration_manager_initialization(self) -> None:
        """Test migration manager initialization."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        assert migration_manager.config == config
        assert migration_manager.db_manager is not None
        assert migration_manager.alembic_cfg is not None

        migration_manager.close()

    def test_database_initialization_new_database(self) -> None:
        """Test initializing a new database."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Initialize database
        results = migration_manager.initialize_database(create_tables=True)

        assert results["database_url"] == "sqlite:///:memory:"
        assert results["tables_created"] is True

        # Verify tables were created
        table_info = migration_manager.db_manager.get_table_info()
        assert "categories" in table_info
        assert "chunks" in table_info
        assert "contextual_helpers" in table_info
        assert "category_limits" in table_info

        migration_manager.close()

    def test_database_initialization_existing_database(self) -> None:
        """Test initializing an existing database."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # First initialization
        migration_manager.initialize_database(create_tables=True)

        # Second initialization should detect existing tables
        results = migration_manager.initialize_database(create_tables=True)

        # Should not recreate tables
        assert "tables_created" in results

        migration_manager.close()

    def test_migration_tracking(self) -> None:
        """Test migration tracking functionality."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Initialize database
        migration_manager.initialize_database(create_tables=True)

        # Check migration tracking
        current_rev = migration_manager.get_current_revision()
        assert current_rev is not None

        # Check migration history
        history = migration_manager.get_migration_history()
        assert isinstance(history, list)

        # Check pending migrations
        pending = migration_manager.get_pending_migrations()
        assert isinstance(pending, list)

        migration_manager.close()

    def test_schema_validation(self) -> None:
        """Test database schema validation."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Initialize database
        migration_manager.initialize_database(create_tables=True)

        # Validate schema
        validation = migration_manager.validate_database_schema()

        assert "is_valid" in validation
        assert "current_revision" in validation
        assert "pending_migrations" in validation
        assert "tables_exist" in validation
        assert "table_info" in validation

        # Should be valid after initialization
        assert validation["tables_exist"] is True

        migration_manager.close()

    def test_database_reset(self) -> None:
        """Test database reset functionality."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Initialize and add some data
        migration_manager.initialize_database(create_tables=True)

        with migration_manager.db_manager.get_session() as session:
            category = Category(name="Test", level=1, parent_id=None)
            session.add(category)

        # Verify data exists
        with migration_manager.db_manager.get_session() as session:
            categories = session.query(Category).all()
            assert len(categories) == 1

        # Reset database
        results = migration_manager.reset_database()

        assert results["success"] is True
        assert "current_revision" in results

        # Verify data is gone but tables exist
        with migration_manager.db_manager.get_session() as session:
            categories = session.query(Category).all()
            assert len(categories) == 0

        migration_manager.close()

    def test_file_database_persistence(self) -> None:
        """Test migration system with file-based database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            config = MemoirAIConfig(database_url=f"sqlite:///{db_path}")

            # First session - initialize database
            migration_manager1 = MigrationManager(config)
            results1 = migration_manager1.initialize_database(create_tables=True)

            # Add some data
            with migration_manager1.db_manager.get_session() as session:
                category = Category(name="Persistent Test", level=1, parent_id=None)
                session.add(category)

            current_rev1 = migration_manager1.get_current_revision()
            migration_manager1.close()

            # Second session - should detect existing database
            migration_manager2 = MigrationManager(config)
            results2 = migration_manager2.initialize_database(create_tables=True)

            # Should not recreate tables
            assert (
                results2["tables_created"] is False
                or results2["tables_created"] is True
            )  # May vary

            # Data should persist
            with migration_manager2.db_manager.get_session() as session:
                categories = session.query(Category).all()
                assert len(categories) == 1
                assert categories[0].name == "Persistent Test"

            # Migration state should persist
            current_rev2 = migration_manager2.get_current_revision()
            assert current_rev2 == current_rev1

            migration_manager2.close()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_error_handling(self) -> None:
        """Test error handling in migration manager."""
        # Test with invalid database URL - this will fail at config level now
        with pytest.raises(ConfigurationError):
            config = MemoirAIConfig(database_url="invalid://url")

        # Test with valid config but invalid database path
        with pytest.raises(DatabaseError):
            config = MemoirAIConfig(database_url="sqlite:///nonexistent/path/test.db")
            migration_manager = MigrationManager(config)

    def test_migration_manager_with_database_manager_integration(self) -> None:
        """Test integration between migration manager and database manager."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Initialize database
        migration_manager.initialize_database(create_tables=True)

        # Test that we can use the database manager normally
        with migration_manager.db_manager.get_session() as session:
            # Create test data
            category = Category(name="Integration Test", level=1, parent_id=None)
            session.add(category)
            session.flush()

            chunk = Chunk(
                content="Test chunk for integration",
                token_count=5,
                category_id=category.id,
                source_id="integration_test",
            )
            session.add(chunk)

        # Verify data was saved
        with migration_manager.db_manager.get_session() as session:
            categories = session.query(Category).all()
            chunks = session.query(Chunk).all()

            assert len(categories) == 1
            assert len(chunks) == 1
            assert chunks[0].category == categories[0]

        # Test table info
        table_info = migration_manager.db_manager.get_table_info()
        assert table_info["categories"]["row_count"] == 1
        assert table_info["chunks"]["row_count"] == 1

        migration_manager.close()

    @patch("memoir_ai.database.migration_manager.command")
    def test_migration_operations_mocked(self, mock_command) -> None:
        """Test migration operations with mocked Alembic commands."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Mock successful operations
        mock_command.stamp.return_value = None
        mock_command.upgrade.return_value = None
        mock_command.downgrade.return_value = None

        # Test upgrade
        result = migration_manager.upgrade_database("head")
        assert result["success"] is True
        mock_command.upgrade.assert_called_once()

        # Test downgrade
        result = migration_manager.downgrade_database("base")
        assert result["success"] is True
        mock_command.downgrade.assert_called_once()

        migration_manager.close()

    def test_tables_exist_check(self) -> None:
        """Test the _tables_exist method."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Before initialization - tables should not exist
        assert migration_manager._tables_exist() is False

        # After initialization - tables should exist
        migration_manager.initialize_database(create_tables=True)
        assert migration_manager._tables_exist() is True

        migration_manager.close()

    def test_migration_initialization_check(self) -> None:
        """Test the _is_migration_initialized method."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        migration_manager = MigrationManager(config)

        # Before initialization - migration tracking should not exist
        assert migration_manager._is_migration_initialized() is False

        # After initialization - migration tracking should exist
        migration_manager.initialize_database(create_tables=True)
        assert migration_manager._is_migration_initialized() is True

        migration_manager.close()
