"""SQLAlchemy database models for MemoirAI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)
from sqlalchemy.sql import func

from ..exceptions import ValidationError


class Base(DeclarativeBase):
    """Declarative base class for ORM models."""

    pass


class Category(Base):
    """
    Category model with strict hierarchy constraints.

    Supports configurable hierarchy depth (1-100 levels, default 3).
    Enforces integrity rules for parent-child relationships.
    """

    __tablename__ = "categories"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core fields
    name: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("categories.id"), nullable=True
    )

    # Optional fields
    slug: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    parent: Mapped[Optional["Category"]] = relationship(
        "Category", remote_side="Category.id", back_populates="children"
    )
    children: Mapped[list["Category"]] = relationship(back_populates="parent")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="category")

    # Constraints
    __table_args__ = (
        # Unique name within same parent
        UniqueConstraint("name", "parent_id", name="unique_name_per_parent"),
        # Level constraints
        CheckConstraint("level >= 1 AND level <= 100", name="valid_level_range"),
        # Parent-level consistency constraints
        CheckConstraint(
            "(level = 1 AND parent_id IS NULL) OR (level > 1 AND parent_id IS NOT NULL)",
            name="parent_level_consistency",
        ),
        # Indexes for performance
        Index("idx_categories_parent_id", "parent_id"),
        Index("idx_categories_level", "level"),
        Index("idx_categories_name", "name"),
        Index(
            "uq_categories_root_name",
            "name",
            unique=True,
            sqlite_where=text("parent_id IS NULL"),
        ),
    )

    @validates("level")
    def validate_level(self, key: str, level: int) -> int:
        """Validate hierarchy level constraints."""
        if not (1 <= level <= 100):
            raise ValidationError(
                f"Level must be between 1 and 100, got {level}",
                field="level",
                value=level,
            )
        return level

    @validates("name")
    def validate_name(self, key: str, name: str) -> str:
        """Validate category name."""
        if not name or not name.strip():
            raise ValidationError(
                "Category name cannot be empty", field="name", value=name
            )

        # Normalize name (strip whitespace)
        return name.strip()

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}', level={self.level})>"

    def get_full_path(self) -> list[str]:
        """Get the full category path from root to this category."""
        path: list[str] = []
        current = self
        while current:
            path.insert(0, current.name)
            current = current.parent
        return path

    def get_path_string(self, separator: str = " → ") -> str:
        """Get the full category path as a string."""
        return separator.join(self.get_full_path())


class Chunk(Base):
    """
    Chunk model for storing text content with category links.

    Each chunk must be linked to a leaf-level category only.
    Includes token counting and source tracking.
    """

    __tablename__ = "chunks"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Category relationship (must be leaf-level)
    category_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("categories.id"), nullable=False
    )

    # Source tracking
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    category: Mapped[Category] = relationship(back_populates="chunks")

    # Constraints and indexes
    __table_args__ = (
        # Performance indexes
        Index("idx_chunks_category_id", "category_id"),
        Index("idx_chunks_source_id", "source_id"),
        Index("idx_chunks_created_at", "created_at"),
        Index("idx_chunks_token_count", "token_count"),
        # Content constraints
        CheckConstraint("token_count > 0", name="positive_token_count"),
        CheckConstraint("length(content) > 0", name="non_empty_content"),
    )

    @validates("token_count")
    def validate_token_count(self, key: str, token_count: int) -> int:
        """Validate token count is positive."""
        if token_count <= 0:
            raise ValidationError(
                f"Token count must be positive, got {token_count}",
                field="token_count",
                value=token_count,
            )
        return token_count

    @validates("content")
    def validate_content(self, key: str, content: str) -> str:
        """Validate content is not empty."""
        if not content or not content.strip():
            raise ValidationError(
                "Chunk content cannot be empty", field="content", value=content
            )
        return content

    def __repr__(self) -> str:
        content_preview = (
            self.content[:50] + "..." if len(self.content) > 50 else self.content
        )
        return f"<Chunk(id={self.id}, tokens={self.token_count}, content='{content_preview}')>"


class ContextualHelper(Base):
    """
    Contextual helper model for storing source context information.

    Supports both auto-generated and user-provided helpers with versioning.
    """

    __tablename__ = "contextual_helpers"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Source identification
    source_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Helper content
    helper_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Helper metadata
    is_user_provided: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Constraints and indexes
    __table_args__ = (
        # Performance indexes
        Index("idx_contextual_helpers_source_id", "source_id"),
        Index("idx_contextual_helpers_created_at", "created_at"),
        # Content constraints
        CheckConstraint("token_count > 0", name="positive_helper_token_count"),
        CheckConstraint("token_count <= 300", name="helper_token_limit"),
        CheckConstraint("length(helper_text) > 0", name="non_empty_helper_text"),
        CheckConstraint("version > 0", name="positive_version"),
    )

    @validates("token_count")
    def validate_token_count(self, key: str, token_count: int) -> int:
        """Validate token count constraints."""
        if token_count <= 0:
            raise ValidationError(
                f"Helper token count must be positive, got {token_count}",
                field="token_count",
                value=token_count,
            )
        if token_count > 300:
            raise ValidationError(
                f"Helper token count must not exceed 300, got {token_count}",
                field="token_count",
                value=token_count,
            )
        return token_count

    @validates("helper_text")
    def validate_helper_text(self, key: str, helper_text: str) -> str:
        """Validate helper text is not empty."""
        if not helper_text or not helper_text.strip():
            raise ValidationError(
                "Helper text cannot be empty", field="helper_text", value=helper_text
            )
        return helper_text.strip()

    @validates("source_id")
    def validate_source_id(self, key: str, source_id: str) -> str:
        """Validate source ID is not empty."""
        if not source_id or not source_id.strip():
            raise ValidationError(
                "Source ID cannot be empty", field="source_id", value=source_id
            )
        return source_id.strip()

    def __repr__(self) -> str:
        helper_preview = (
            self.helper_text[:50] + "..."
            if len(self.helper_text) > 50
            else self.helper_text
        )
        return f"<ContextualHelper(source_id='{self.source_id}', tokens={self.token_count}, text='{helper_preview}')>"


class ProjectMetadata(Base):
    """Key-value store for project-level metadata."""

    __tablename__ = "project_metadata"

    # Identifier
    key: Mapped[str] = mapped_column(String(128), primary_key=True)

    # Metadata payload stored as JSON for flexibility
    value_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (Index("idx_project_metadata_key", "key", unique=True),)

    def __repr__(self) -> str:
        return f"<ProjectMetadata(key='{self.key}', value={self.value_json})>"


class CategoryLimits(Base):
    """
    Category limits configuration model.

    Stores configurable limits for maximum categories per hierarchy level.
    """

    __tablename__ = "category_limits"

    # Primary key (level number)
    level: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Limit configuration
    max_categories: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )

    # Constraints
    __table_args__ = (
        # Validation constraints
        CheckConstraint("level >= 1 AND level <= 100", name="valid_limit_level_range"),
        CheckConstraint("max_categories > 0", name="positive_max_categories"),
    )

    @validates("level")
    def validate_level(self, key: str, level: int) -> int:
        """Validate level is within valid range."""
        if not (1 <= level <= 100):
            raise ValidationError(
                f"Category limit level must be between 1 and 100, got {level}",
                field="level",
                value=level,
            )
        return level

    @validates("max_categories")
    def validate_max_categories(self, key: str, max_categories: int) -> int:
        """Validate max categories is positive."""
        if max_categories <= 0:
            raise ValidationError(
                f"Max categories must be positive, got {max_categories}",
                field="max_categories",
                value=max_categories,
            )
        return max_categories

    def __repr__(self) -> str:
        return f"<CategoryLimits(level={self.level}, max_categories={self.max_categories})>"
