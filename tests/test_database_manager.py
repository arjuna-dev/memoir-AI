"""
Tests for database manager and engine functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from memoir_ai.config import MemoirAIConfig
from memoir_ai.database.engine import DatabaseManager
from memoir_ai.database.models import Base, Category
from memoir_ai.exceptions import DatabaseError


class TestDatabaseManager:
    """Test DatabaseManager functionality."""

    def test_sqlite_initialization(self) -> None:
        """Test SQLite database initialization."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            config = MemoirAIConfig(database_url=f"sqlite:///{db_path}")
            db_manager = DatabaseManager(config)

            assert db_manager.engine is not None
            assert db_manager.SessionLocal is not None

            # Test table creation
            db_manager.create_tables()

            # Test connection
            with db_manager.get_session() as session:
                # Should be able to create a category
                category = Category(name="Test", level=1, parent_id=None)
                session.add(category)
                session.commit()

                assert category.id is not None

            db_manager.close()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_in_memory_sqlite(self) -> None:
        """Test in-memory SQLite database."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)

        assert db_manager.engine is not None

        # Create tables
        db_manager.create_tables()

        # Test session usage
        with db_manager.get_session() as session:
            category = Category(name="Memory Test", level=1, parent_id=None)
            session.add(category)
            # Commit happens automatically when exiting context

        # Verify data persists within the same connection
        with db_manager.get_session() as session:
            categories = session.query(Category).all()
            assert len(categories) == 1
            assert categories[0].name == "Memory Test"

        db_manager.close()

    def test_session_error_handling(self) -> None:
        """Test session error handling and rollback."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        # Test that errors cause rollback
        with pytest.raises(DatabaseError):
            with db_manager.get_session() as session:
                # Create valid category
                category = Category(name="Test", level=1, parent_id=None)
                session.add(category)
                session.flush()  # This should work

                # Now cause an error by violating a constraint
                invalid_category = Category(
                    name="Test", level=1, parent_id=None
                )  # Same name
                session.add(invalid_category)
                # Error will occur on commit, causing rollback

        # Verify rollback occurred - no categories should exist
        with db_manager.get_session() as session:
            categories = session.query(Category).all()
            assert len(categories) == 0

        db_manager.close()

    def test_get_table_info(self) -> None:
        """Test getting table information."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        # Add some test data
        with db_manager.get_session() as session:
            category = Category(name="Test", level=1, parent_id=None)
            session.add(category)

        # Get table info
        info = db_manager.get_table_info()

        assert "categories" in info
        assert "chunks" in info
        assert "contextual_helpers" in info
        assert "project_metadata" in info
        assert "category_limits" in info

        assert info["categories"]["row_count"] == 1
        assert info["chunks"]["row_count"] == 0
        assert info["project_metadata"]["row_count"] == 0

        db_manager.close()

    def test_engine_kwargs_sqlite(self) -> None:
        """Test SQLite-specific engine configuration."""
        config = MemoirAIConfig(database_url="sqlite:///test.db")
        db_manager = DatabaseManager(config)

        kwargs = db_manager._get_engine_kwargs()

        assert "poolclass" in kwargs
        assert "connect_args" in kwargs
        assert kwargs["connect_args"]["check_same_thread"] is False
        assert "timeout" in kwargs["connect_args"]

        db_manager.close()

    def test_engine_kwargs_postgresql(self) -> None:
        """Test PostgreSQL-specific engine configuration."""
        config = MemoirAIConfig(database_url="postgresql://user:pass@localhost/db")
        db_manager = DatabaseManager(config)

        kwargs = db_manager._get_engine_kwargs()

        assert "pool_size" in kwargs
        assert "max_overflow" in kwargs
        assert kwargs["pool_pre_ping"] is True
        assert "connect_args" in kwargs
        assert "application_name" in kwargs["connect_args"]

        # Note: We don't actually connect to PostgreSQL in tests
        db_manager.engine = None  # Prevent connection attempt in close()

    def test_engine_kwargs_mysql(self) -> None:
        """Test MySQL-specific engine configuration."""
        config = MemoirAIConfig(database_url="mysql://user:pass@localhost/db")
        db_manager = DatabaseManager(config)

        kwargs = db_manager._get_engine_kwargs()

        assert "pool_size" in kwargs
        assert "max_overflow" in kwargs
        assert kwargs["pool_pre_ping"] is True
        assert "connect_args" in kwargs
        assert kwargs["connect_args"]["charset"] == "utf8mb4"

        # Note: We don't actually connect to MySQL in tests
        db_manager.engine = None  # Prevent connection attempt in close()

    def test_invalid_database_url(self) -> None:
        """Test handling of invalid database URL."""
        config = MemoirAIConfig(database_url="invalid://url")

        with pytest.raises(DatabaseError) as exc_info:
            DatabaseManager(config)

        assert "Failed to initialize database engine" in str(exc_info.value)

    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_execute_with_retry(self, mock_sleep) -> None:
        """Test retry logic for database operations."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)

        # Test successful operation (no retries needed)
        def successful_operation() -> None:
            return "success"

        result = db_manager.execute_with_retry(successful_operation)
        assert result == "success"
        assert mock_sleep.call_count == 0

        # Test operation that fails then succeeds
        call_count = 0

        def flaky_operation() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                from sqlalchemy.exc import OperationalError

                raise OperationalError("Connection failed", None, None)
            return "success_after_retry"

        result = db_manager.execute_with_retry(flaky_operation)
        assert result == "success_after_retry"
        assert mock_sleep.call_count == 1  # One retry

        # Reset for next test
        mock_sleep.reset_mock()

        # Test operation that always fails
        def always_fails() -> None:
            from sqlalchemy.exc import OperationalError

            raise OperationalError("Always fails", None, None)

        with pytest.raises(DatabaseError) as exc_info:
            db_manager.execute_with_retry(always_fails)

        assert "failed after" in str(exc_info.value)
        assert mock_sleep.call_count == 3  # max_retries attempts

        db_manager.close()

    def test_drop_tables(self) -> None:
        """Test dropping database tables."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)

        # Create tables
        db_manager.create_tables()

        # Add some data
        with db_manager.get_session() as session:
            category = Category(name="Test", level=1, parent_id=None)
            session.add(category)

        # Verify data exists
        info = db_manager.get_table_info()
        assert info["categories"]["row_count"] == 1

        # Drop tables
        db_manager.drop_tables()

        # Recreate tables (should be empty)
        db_manager.create_tables()
        info = db_manager.get_table_info()
        assert info["categories"]["row_count"] == 0

        db_manager.close()

    def test_database_manager_context_usage(self) -> None:
        """Test using database manager in various contexts."""
        config = MemoirAIConfig(database_url="sqlite:///:memory:")
        db_manager = DatabaseManager(config)
        db_manager.create_tables()

        # Test nested session usage (should work)
        with db_manager.get_session() as session1:
            category1 = Category(name="Category1", level=1, parent_id=None)
            session1.add(category1)
            session1.flush()  # Flush but don't commit yet

            # Use another session (should be independent)
            with db_manager.get_session() as session2:
                category2 = Category(name="Category2", level=1, parent_id=None)
                session2.add(category2)
                # session2 commits when exiting context

            # session1 should still be valid
            assert category1.name == "Category1"
            # session1 commits when exiting context

        # Verify both categories were saved
        with db_manager.get_session() as session:
            categories = session.query(Category).all()
            assert len(categories) == 2
            names = [c.name for c in categories]
            assert "Category1" in names
            assert "Category2" in names

        db_manager.close()
