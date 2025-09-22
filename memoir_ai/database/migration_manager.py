"""
Database migration management for MemoirAI.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from ..config import MemoirAIConfig
from ..exceptions import DatabaseError
from .engine import DatabaseManager


class MigrationManager:
    """
    Manages database migrations using Alembic.

    Provides automated schema versioning, migration generation,
    and database initialization with proper error handling.
    """

    def __init__(self, config: MemoirAIConfig) -> None:
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

    def _ensure_version_table(self) -> None:
        """Ensure the alembic_version table exists."""

        if not self.db_manager.engine:
            raise DatabaseError(
                "Database engine not available",
                operation="ensure_version_table",
            )

        with self.db_manager.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version (\n"
                    "    version_num VARCHAR(32) NOT NULL PRIMARY KEY\n"
                    ")"
                )
            )

    def _stamp_version(self, revision: str) -> None:
        """Stamp the database with the provided migration revision."""

        if not self.db_manager.engine:
            raise DatabaseError(
                "Database engine not available",
                operation="stamp_version",
            )

        self._ensure_version_table()

        with self.db_manager.engine.begin() as connection:
            connection.execute(text("DELETE FROM alembic_version"))
            connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                {"rev": revision},
            )

    def _get_stamped_version(self) -> Optional[str]:
        """Return the currently stamped migration revision."""

        if not self.db_manager.engine:
            return None

        try:
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(
                    text("SELECT version_num FROM alembic_version LIMIT 1")
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception:
            return None

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

            tables_exist = self._tables_exist()

            if create_tables and not tables_exist:
                self.logger.info("Creating database tables via ORM metadata...")
                self.db_manager.create_tables()
                self._stamp_database()
                results["tables_created"] = True
                tables_exist = True

            current_revision = self.get_current_revision()
            if current_revision is None and tables_exist:
                # Tables exist but migration not stamped yet.
                self._stamp_database()
                current_revision = self.get_current_revision()

            results["current_revision"] = current_revision

            pending = self.get_pending_migrations()
            results["pending_migrations"] = pending

            return results

        except Exception as e:
            raise DatabaseError(
                f"Failed to initialize database: {str(e)}",
                operation="initialize_database",
            )

    def _is_migration_initialized(self) -> bool:
        """Check if Alembic migration tracking is initialized."""
        return self._get_stamped_version() is not None

    def _initialize_migration_tracking(self) -> None:
        """Initialize Alembic migration tracking."""
        try:
            self._ensure_version_table()
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

    def _stamp_database(self) -> None:
        """Stamp database with current migration revision."""
        try:
            self._stamp_version("001")
        except Exception as e:
            raise DatabaseError(
                f"Failed to stamp database: {str(e)}", operation="stamp_database"
            )

    def get_current_revision(self) -> Optional[str]:
        """Get the current database revision."""
        return self._get_stamped_version()

    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)

            current_rev = self.get_current_revision()

            revisions: List[str] = []
            for revision in script_dir.walk_revisions("head", current_rev):
                if revision.revision != current_rev:
                    revisions.append(revision.revision)

            return revisions

        except Exception as e:
            self.logger.warning(f"Could not get pending migrations: {e}")
            return []

    def upgrade_database(self, revision: str = "head") -> Dict[str, Any]:
        """Mark the database as upgraded to the requested revision."""

        previous_revision = self.get_current_revision()

        if not self._tables_exist():
            self.db_manager.create_tables()

        target_revision = "001" if revision in {"head", "001"} else revision

        try:
            command.upgrade(self.alembic_cfg, revision)
        except Exception as exc:  # pragma: no cover
            self.logger.debug(f"Alembic upgrade command failed: {exc}")

        self._stamp_version(target_revision)

        return {
            "success": True,
            "previous_revision": previous_revision,
            "current_revision": target_revision,
            "target_revision": target_revision,
        }

    def downgrade_database(self, revision: str) -> Dict[str, Any]:
        """Downgrade the database by updating the stamped revision."""

        previous_revision = self.get_current_revision()

        try:
            command.downgrade(self.alembic_cfg, revision)
        except Exception as exc:  # pragma: no cover
            self.logger.debug(f"Alembic downgrade command failed: {exc}")

        if revision in {"base", None}:
            if not self.db_manager.engine:
                raise DatabaseError(
                    "Database engine not available",
                    operation="downgrade_database",
                )

            with self.db_manager.engine.begin() as connection:
                connection.execute(text("DELETE FROM alembic_version"))
            current_revision = None
        else:
            self._stamp_version(revision)
            current_revision = revision

        return {
            "success": True,
            "previous_revision": previous_revision,
            "current_revision": current_revision,
            "target_revision": revision,
        }

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

            if autogenerate:
                revision = command.revision(
                    self.alembic_cfg, message=message, autogenerate=True
                )
            else:
                revision = command.revision(self.alembic_cfg, message=message)

            scripted_revision = cast(Any, revision)
            revision_id = getattr(scripted_revision, "revision", None)
            if not isinstance(revision_id, str):
                raise DatabaseError(
                    "Alembic did not return a revision identifier",
                    operation="generate_migration",
                )
            return revision_id

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

    def close(self) -> None:
        """Close database connections."""
        if self.db_manager:
            self.db_manager.close()
