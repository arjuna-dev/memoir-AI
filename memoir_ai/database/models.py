"""
SQLAlchemy database models for MemoirAI.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    SmallInteger,
    Boolean,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
    JSON,
    text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from ..exceptions import ValidationError

Base = declarative_base()


class Category(Base):
    """
    Category model with strict hierarchy constraints.

    Supports configurable hierarchy depth (1-100 levels, default 3).
    Enforces integrity rules for parent-child relationships.
    """

    __tablename__ = "categories"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core fields
    name = Column(String(255), nullable=False)
    level = Column(SmallInteger, nullable=False)
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)

    # Optional fields
    slug = Column(String(255), unique=True, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    parent = relationship("Category", remote_side=[id], backref="children")
    chunks = relationship("Chunk", back_populates="category")

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
    def validate_level(self, key, level):
        """Validate hierarchy level constraints."""
        if not (1 <= level <= 100):
            raise ValidationError(
                f"Level must be between 1 and 100, got {level}",
                field="level",
                value=level,
            )
        return level

    @validates("name")
    def validate_name(self, key, name):
        """Validate category name."""
        if not name or not name.strip():
            raise ValidationError(
                "Category name cannot be empty", field="name", value=name
            )

        # Normalize name (strip whitespace)
        return name.strip()

    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}', level={self.level})>"

    def get_full_path(self) -> List[str]:
        """Get the full category path from root to this category."""
        path = []
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
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core content
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    # Category relationship (must be leaf-level)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)

    # Source tracking
    source_id = Column(String(255), nullable=True)
    source_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    category = relationship("Category", back_populates="chunks")

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
    def validate_token_count(self, key, token_count):
        """Validate token count is positive."""
        if token_count <= 0:
            raise ValidationError(
                f"Token count must be positive, got {token_count}",
                field="token_count",
                value=token_count,
            )
        return token_count

    @validates("content")
    def validate_content(self, key, content):
        """Validate content is not empty."""
        if not content or not content.strip():
            raise ValidationError(
                "Chunk content cannot be empty", field="content", value=content
            )
        return content

    def __repr__(self):
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
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Source identification
    source_id = Column(String(255), unique=True, nullable=False)

    # Helper content
    helper_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    # Helper metadata
    is_user_provided = Column(Boolean, default=False, nullable=False)
    version = Column(Integer, default=1, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
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
    def validate_token_count(self, key, token_count):
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
    def validate_helper_text(self, key, helper_text):
        """Validate helper text is not empty."""
        if not helper_text or not helper_text.strip():
            raise ValidationError(
                "Helper text cannot be empty", field="helper_text", value=helper_text
            )
        return helper_text.strip()

    @validates("source_id")
    def validate_source_id(self, key, source_id):
        """Validate source ID is not empty."""
        if not source_id or not source_id.strip():
            raise ValidationError(
                "Source ID cannot be empty", field="source_id", value=source_id
            )
        return source_id.strip()

    def __repr__(self):
        helper_preview = (
            self.helper_text[:50] + "..."
            if len(self.helper_text) > 50
            else self.helper_text
        )
        return f"<ContextualHelper(source_id='{self.source_id}', tokens={self.token_count}, text='{helper_preview}')>"


class CategoryLimits(Base):
    """
    Category limits configuration model.

    Stores configurable limits for maximum categories per hierarchy level.
    """

    __tablename__ = "category_limits"

    # Primary key (level number)
    level = Column(Integer, primary_key=True)

    # Limit configuration
    max_categories = Column(Integer, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Constraints
    __table_args__ = (
        # Validation constraints
        CheckConstraint("level >= 1 AND level <= 100", name="valid_limit_level_range"),
        CheckConstraint("max_categories > 0", name="positive_max_categories"),
    )

    @validates("level")
    def validate_level(self, key, level):
        """Validate level is within valid range."""
        if not (1 <= level <= 100):
            raise ValidationError(
                f"Category limit level must be between 1 and 100, got {level}",
                field="level",
                value=level,
            )
        return level

    @validates("max_categories")
    def validate_max_categories(self, key, max_categories):
        """Validate max categories is positive."""
        if max_categories <= 0:
            raise ValidationError(
                f"Max categories must be positive, got {max_categories}",
                field="max_categories",
                value=max_categories,
            )
        return max_categories

    def __repr__(self):
        return f"<CategoryLimits(level={self.level}, max_categories={self.max_categories})>"
