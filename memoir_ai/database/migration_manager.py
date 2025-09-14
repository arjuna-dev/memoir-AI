"""
Database migration management for MemoirAI.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from .engine import DatabaseManager
from ..config import MemoirAIConfig
from ..exceptions import DatabaseError


class MigrationManager:
    """
    Manages database migrations using Alembic.

    Provides automated schema versioning, migration generation,
    and database initialization with proper error handling.
    """

    def __init__(self, config: MemoirAIConfig):
        """Initialize migration manager with configuration."""
        self.config = config
        self.db_manager = DatabaseManager(config)

        # Set up Alembic configuration
        self.alembic_cfg = self._setup_alembic_config()

        # Migration tracking
        self.logger = logging.getLogger(__name__)

    def _setup_alembic_config(self) -> Config:
        """Set up Alembic configuration."""
        # Get the path to the migrations directory
        migrations_dir = Path(__file__).parent / "migrations"
        alembic_ini_path = migrations_dir / "alembic.ini"

        if not alembic_ini_path.exists():
            raise DatabaseError(
                f"Alembic configuration not found at {alembic_ini_path}",
                operation="setup_alembic",
            )

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))

        # Set the database URL
        alembic_cfg.set_main_option("sqlalchemy.url", self.config.database_url)

        # Set the script location to the migrations directory
        alembic_cfg.set_main_option("script_location", str(migrations_dir))

        return alembic_cfg

    def initialize_database(self, create_tables: bool = True) -> Dict[str, Any]:
        """
        Initialize database with proper schema and migration tracking.

        Args:
            create_tables: Whether to create tables if they don't exist

        Returns:
            Dictionary with initialization results
        """
        try:
            results = {
                "database_url": self.config.database_url,
                "tables_created": False,
                "migration_initialized": False,
                "current_revision": None,
                "pending_migrations": [],
            }

            # Test database connection
            self.db_manager._test_connection()
            self.logger.info(
                f"Database connection successful: {self.config.database_url}"
            )

            # Check if migration tracking is initialized
            if not self._is_migration_initialized():
                self.logger.info("Initializing migration tracking...")
                self._initialize_migration_tracking()
                results["migration_initialized"] = True

            # Get current migration state
            current_revision = self.get_current_revision()
            results["current_revision"] = current_revision

            # Check for pending migrations
            pending = self.get_pending_migrations()
            results["pending_migrations"] = pending

            # Create tables if requested and needed
            if create_tables:
                if not self._tables_exist() or pending:
                    if pending:
                        self.logger.info(
                            f"Applying {len(pending)} pending migrations..."
                        )
                        self.upgrade_database()
                    else:
                        self.logger.info("Creating database tables...")
                        self.db_manager.create_tables()
                        # Stamp the database with the current revision
                        self._stamp_database()

                    results["tables_created"] = True
                else:
                    self.logger.info("Database tables already exist and are up to date")

            return results

        except Exception as e:
            raise DatabaseError(
                f"Failed to initialize database: {str(e)}",
                operation="initialize_database",
            )

    def _is_migration_initialized(self) -> bool:
        """Check if Alembic migration tracking is initialized."""
        try:
            with self.db_manager.get_session() as session:
                # Try to query the alembic_version table
                result = session.execute(
                    text("SELECT version_num FROM alembic_version LIMIT 1")
                )
                result.fetchone()
                return True
        except Exception:
            return False

    def _initialize_migration_tracking(self):
        """Initialize Alembic migration tracking."""
        try:
            # Create the alembic_version table
            command.stamp(self.alembic_cfg, "head")
        except Exception as e:
            raise DatabaseError(
                f"Failed to initialize migration tracking: {str(e)}",
                operation="initialize_migration_tracking",
            )

    def _tables_exist(self) -> bool:
        """Check if main application tables exist."""
        try:
            with self.db_manager.get_session() as session:
                # Check for our main tables
                tables_to_check = [
                    "categories",
                    "chunks",
                    "contextual_helpers",
                    "category_limits",
                ]

                for table_name in tables_to_check:
                    try:
                        session.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
                    except Exception:
                        return False

                return True
        except Exception:
            return False

    def _stamp_database(self):
        """Stamp database with current migration revision."""
        try:
            command.stamp(self.alembic_cfg, "head")
        except Exception as e:
            raise DatabaseError(
                f"Failed to stamp database: {str(e)}", operation="stamp_database"
            )

    def get_current_revision(self) -> Optional[str]:
        """Get the current database revision."""
        try:
            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            self.logger.warning(f"Could not get current revision: {e}")
            return None

    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)

            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()

                # Get all revisions from current to head
                revisions = []
                for revision in script_dir.walk_revisions("head", current_rev):
                    if revision.revision != current_rev:
                        revisions.append(revision.revision)

                return revisions

        except Exception as e:
            self.logger.warning(f"Could not get pending migrations: {e}")
            return []

    def upgrade_database(self, revision: str = "head") -> Dict[str, Any]:
        """
        Upgrade database to specified revision.

        Args:
            revision: Target revision (default: "head")

        Returns:
            Dictionary with upgrade results
        """
        try:
            current_rev = self.get_current_revision()

            self.logger.info(f"Upgrading database from {current_rev} to {revision}")

            # Run the upgrade
            command.upgrade(self.alembic_cfg, revision)

            new_rev = self.get_current_revision()

            return {
                "success": True,
                "previous_revision": current_rev,
                "current_revision": new_rev,
                "target_revision": revision,
            }

        except Exception as e:
            raise DatabaseError(
                f"Failed to upgrade database: {str(e)}", operation="upgrade_database"
            )

    def downgrade_database(self, revision: str) -> Dict[str, Any]:
        """
        Downgrade database to specified revision.

        Args:
            revision: Target revision

        Returns:
            Dictionary with downgrade results
        """
        try:
            current_rev = self.get_current_revision()

            self.logger.info(f"Downgrading database from {current_rev} to {revision}")

            # Run the downgrade
            command.downgrade(self.alembic_cfg, revision)

            new_rev = self.get_current_revision()

            return {
                "success": True,
                "previous_revision": current_rev,
                "current_revision": new_rev,
                "target_revision": revision,
            }

        except Exception as e:
            raise DatabaseError(
                f"Failed to downgrade database: {str(e)}",
                operation="downgrade_database",
            )

    def generate_migration(self, message: str, autogenerate: bool = True) -> str:
        """
        Generate a new migration.

        Args:
            message: Migration message/description
            autogenerate: Whether to auto-generate migration from model changes

        Returns:
            Generated revision ID
        """
        try:
            self.logger.info(f"Generating migration: {message}")

            # Generate the migration
            if autogenerate:
                revision = command.revision(
                    self.alembic_cfg, message=message, autogenerate=True
                )
            else:
                revision = command.revision(self.alembic_cfg, message=message)

            return revision.revision

        except Exception as e:
            raise DatabaseError(
                f"Failed to generate migration: {str(e)}",
                operation="generate_migration",
            )

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)

            history = []
            for revision in script_dir.walk_revisions():
                history.append(
                    {
                        "revision": revision.revision,
                        "down_revision": revision.down_revision,
                        "message": revision.doc,
                        "branch_labels": revision.branch_labels,
                    }
                )

            return history

        except Exception as e:
            self.logger.warning(f"Could not get migration history: {e}")
            return []

    def validate_database_schema(self) -> Dict[str, Any]:
        """
        Validate that the database schema matches the current models.

        Returns:
            Dictionary with validation results
        """
        try:
            # Get current revision
            current_rev = self.get_current_revision()

            # Check for pending migrations
            pending = self.get_pending_migrations()

            # Check if tables exist
            tables_exist = self._tables_exist()

            # Get table info
            table_info = self.db_manager.get_table_info()

            is_valid = current_rev is not None and len(pending) == 0 and tables_exist

            return {
                "is_valid": is_valid,
                "current_revision": current_rev,
                "pending_migrations": pending,
                "tables_exist": tables_exist,
                "table_info": table_info,
            }

        except Exception as e:
            raise DatabaseError(
                f"Failed to validate database schema: {str(e)}",
                operation="validate_schema",
            )

    def reset_database(self) -> Dict[str, Any]:
        """
        Reset database by dropping all tables and recreating them.

        WARNING: This will destroy all data!

        Returns:
            Dictionary with reset results
        """
        try:
            self.logger.warning("Resetting database - all data will be lost!")

            # Drop all tables
            self.db_manager.drop_tables()

            # Recreate tables
            self.db_manager.create_tables()

            # Reinitialize migration tracking
            self._stamp_database()

            return {
                "success": True,
                "current_revision": self.get_current_revision(),
                "message": "Database reset successfully",
            }

        except Exception as e:
            raise DatabaseError(
                f"Failed to reset database: {str(e)}", operation="reset_database"
            )

    def close(self):
        """Close database connections."""
        if self.db_manager:
            self.db_manager.close()
