"""
Database engine and connection management for MemoirAI.
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import StaticPool

from .models import Base
from ..exceptions import DatabaseError
from ..config import MemoirAIConfig


class DatabaseManager:
    """
    Database connection and session management.

    Handles database initialization, connection pooling, and error recovery.
    """

    def __init__(self, config: MemoirAIConfig):
        """Initialize database manager with configuration."""
        self.config = config
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None

        # Connection retry settings
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.backoff_multiplier = 2.0

        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with appropriate settings."""
        try:
            # Engine configuration based on database type
            engine_kwargs = self._get_engine_kwargs()

            self.engine = create_engine(self.config.database_url, **engine_kwargs)

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )

            # Test connection
            self._test_connection()

        except Exception as e:
            raise DatabaseError(
                f"Failed to initialize database engine: {str(e)}",
                operation="initialize_engine",
            )

    def _get_engine_kwargs(self) -> Dict[str, Any]:
        """Get engine configuration based on database type."""
        url_lower = self.config.database_url.lower()

        if url_lower.startswith("sqlite"):
            return {
                "echo": False,
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,  # Allow SQLite usage across threads
                    "timeout": 30,  # Connection timeout
                },
                "pool_pre_ping": True,  # Verify connections before use
            }

        elif url_lower.startswith("postgresql"):
            return {
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
                "pool_recycle": 3600,  # Recycle connections after 1 hour
                "connect_args": {
                    "connect_timeout": 30,
                    "application_name": "memoir_ai",
                },
            }

        elif url_lower.startswith("mysql"):
            return {
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
                "connect_args": {
                    "connect_timeout": 30,
                    "charset": "utf8mb4",
                },
            }

        else:
            # Default configuration
            return {
                "echo": False,
                "pool_pre_ping": True,
            }

    def _test_connection(self):
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise DatabaseError(
                f"Database connection test failed: {str(e)}",
                operation="test_connection",
            )

    def create_tables(self, use_migrations: bool = False):
        """
        Create all database tables and indexes.

        Args:
            use_migrations: If True, use migration system instead of direct creation
        """
        try:
            if use_migrations:
                # Import here to avoid circular imports
                from .migration_manager import MigrationManager

                migration_manager = MigrationManager(self.config)
                migration_manager.initialize_database(create_tables=True)
            else:
                Base.metadata.create_all(bind=self.engine)
        except Exception as e:
            raise DatabaseError(
                f"Failed to create database tables: {str(e)}", operation="create_tables"
            )

    def drop_tables(self):
        """Drop all database tables (for testing/cleanup)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
        except Exception as e:
            raise DatabaseError(
                f"Failed to drop database tables: {str(e)}", operation="drop_tables"
            )

    @contextmanager
    def get_session(self):
        """
        Get a database session with automatic cleanup and error handling.

        Usage:
            with db_manager.get_session() as session:
                # Use session here
                pass
        """
        if not self.SessionLocal:
            raise DatabaseError("Database not initialized", operation="get_session")

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"Database session error: {str(e)}", operation="session_operation"
            )
        finally:
            session.close()

    def execute_with_retry(self, operation_func, *args, **kwargs):
        """
        Execute database operation with retry logic and exponential backoff.

        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result of the operation function
        """
        last_exception = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return operation_func(*args, **kwargs)

            except OperationalError as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Log retry attempt (in production, use proper logging)
                    print(
                        f"Database operation failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {str(e)}"
                    )

                    time.sleep(delay)
                    delay *= self.backoff_multiplier
                else:
                    # Final attempt failed
                    break

            except Exception as e:
                # Non-retryable error
                raise DatabaseError(
                    f"Database operation failed: {str(e)}",
                    operation="execute_with_retry",
                )

        # All retries exhausted
        raise DatabaseError(
            f"Database operation failed after {self.max_retries + 1} attempts: {str(last_exception)}",
            operation="execute_with_retry",
        )

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about database tables and their row counts."""
        try:
            with self.get_session() as session:
                info = {}

                # Get table names from metadata
                for table_name in Base.metadata.tables.keys():
                    try:
                        result = session.execute(
                            text(f"SELECT COUNT(*) FROM {table_name}")
                        )
                        count = result.scalar()
                        info[table_name] = {"row_count": count}
                    except Exception as e:
                        info[table_name] = {"error": str(e)}

                return info

        except Exception as e:
            raise DatabaseError(
                f"Failed to get table information: {str(e)}", operation="get_table_info"
            )

    def close(self):
        """Close database connections and cleanup resources."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
