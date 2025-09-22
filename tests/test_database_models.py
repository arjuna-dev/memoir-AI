"""
Tests for SQLAlchemy database models.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from memoir_ai.database.models import (
    Base,
    Category,
    CategoryLimits,
    Chunk,
    ContextualHelper,
)
from memoir_ai.exceptions import ValidationError


@pytest.fixture
def db_session() -> None:
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()


class TestCategoryModel:
    """Test Category model functionality."""

    def test_create_valid_category(self, db_session) -> None:
        """Test creating a valid category."""
        category = Category(name="Technology", level=1, parent_id=None)

        db_session.add(category)
        db_session.commit()

        assert category.id is not None
        assert category.name == "Technology"
        assert category.level == 1
        assert category.parent_id is None
        assert category.created_at is not None
        assert category.updated_at is not None

    def test_category_hierarchy(self, db_session) -> None:
        """Test category parent-child relationships."""
        # Create parent category
        parent = Category(name="Technology", level=1, parent_id=None)
        db_session.add(parent)
        db_session.commit()

        # Create child category
        child = Category(name="AI/ML", level=2, parent_id=parent.id)
        db_session.add(child)
        db_session.commit()

        # Test relationships
        assert child.parent == parent
        assert parent.children[0] == child

        # Test path methods
        assert child.get_full_path() == ["Technology", "AI/ML"]
        assert child.get_path_string() == "Technology → AI/ML"

    def test_category_level_validation(self, db_session) -> None:
        """Test category level validation."""
        # Test invalid level (too low)
        with pytest.raises(ValidationError) as exc_info:
            Category(name="Test", level=0, parent_id=None)
        assert "Level must be between 1 and 100" in str(exc_info.value)

        # Test invalid level (too high)
        with pytest.raises(ValidationError) as exc_info:
            Category(name="Test", level=101, parent_id=None)
        assert "Level must be between 1 and 100" in str(exc_info.value)

        # Test valid levels
        category1 = Category(name="Test1", level=1, parent_id=None)
        db_session.add(category1)
        db_session.commit()

        # For level 100, we need a parent (due to constraint)
        category100 = Category(name="Test100", level=100, parent_id=category1.id)
        db_session.add(category100)
        db_session.commit()

        assert category1.level == 1
        assert category100.level == 100

    def test_category_name_validation(self, db_session) -> None:
        """Test category name validation."""
        # Test empty name
        with pytest.raises(ValidationError) as exc_info:
            Category(name="", level=1, parent_id=None)
        assert "Category name cannot be empty" in str(exc_info.value)

        # Test whitespace-only name
        with pytest.raises(ValidationError) as exc_info:
            Category(name="   ", level=1, parent_id=None)
        assert "Category name cannot be empty" in str(exc_info.value)

        # Test name normalization (whitespace stripping)
        category = Category(name="  Technology  ", level=1, parent_id=None)
        assert category.name == "Technology"

    def test_unique_name_per_parent_constraint(self, db_session) -> None:
        """Test unique name constraint within same parent."""
        # Create parent
        parent = Category(name="Technology", level=1, parent_id=None)
        db_session.add(parent)
        db_session.commit()

        # Create first child
        child1 = Category(name="AI", level=2, parent_id=parent.id)
        db_session.add(child1)
        db_session.commit()

        # Try to create second child with same name - should fail
        child2 = Category(name="AI", level=2, parent_id=parent.id)
        db_session.add(child2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_parent_level_consistency_constraint(self, db_session) -> None:
        """Test parent-level consistency constraints."""
        # Create parent for testing
        parent = Category(name="Technology", level=1, parent_id=None)
        db_session.add(parent)
        db_session.commit()

        # Test: Level 1 with parent_id should fail (via database constraint)
        invalid_category = Category(name="Invalid", level=1, parent_id=parent.id)
        db_session.add(invalid_category)

        with pytest.raises(IntegrityError):
            db_session.commit()

        db_session.rollback()

        # Test: Level > 1 without parent_id should fail (via database constraint)
        invalid_category2 = Category(name="Invalid2", level=2, parent_id=None)
        db_session.add(invalid_category2)

        with pytest.raises(IntegrityError):
            db_session.commit()


class TestChunkModel:
    """Test Chunk model functionality."""

    def test_create_valid_chunk(self, db_session) -> None:
        """Test creating a valid chunk."""
        # Create category first
        category = Category(name="Technology", level=1, parent_id=None)
        db_session.add(category)
        db_session.commit()

        # Create chunk
        chunk = Chunk(
            content="This is a test chunk about technology.",
            token_count=8,
            category_id=category.id,
            source_id="test_source_1",
        )

        db_session.add(chunk)
        db_session.commit()

        assert chunk.id is not None
        assert chunk.content == "This is a test chunk about technology."
        assert chunk.token_count == 8
        assert chunk.category_id == category.id
        assert chunk.source_id == "test_source_1"
        assert chunk.created_at is not None
        assert chunk.category == category

    def test_chunk_content_validation(self, db_session) -> None:
        """Test chunk content validation."""
        # Create category first
        category = Category(name="Technology", level=1, parent_id=None)
        db_session.add(category)
        db_session.commit()

        # Test empty content
        with pytest.raises(ValidationError) as exc_info:
            Chunk(content="", token_count=1, category_id=category.id)
        assert "Chunk content cannot be empty" in str(exc_info.value)

        # Test whitespace-only content
        with pytest.raises(ValidationError) as exc_info:
            Chunk(content="   ", token_count=1, category_id=category.id)
        assert "Chunk content cannot be empty" in str(exc_info.value)

    def test_chunk_token_count_validation(self, db_session) -> None:
        """Test chunk token count validation."""
        # Create category first
        category = Category(name="Technology", level=1, parent_id=None)
        db_session.add(category)
        db_session.commit()

        # Test zero token count
        with pytest.raises(ValidationError) as exc_info:
            Chunk(content="Test content", token_count=0, category_id=category.id)
        assert "Token count must be positive" in str(exc_info.value)

        # Test negative token count
        with pytest.raises(ValidationError) as exc_info:
            Chunk(content="Test content", token_count=-1, category_id=category.id)
        assert "Token count must be positive" in str(exc_info.value)

    def test_chunk_category_relationship(self, db_session) -> None:
        """Test chunk-category relationship."""
        # Create category
        category = Category(name="Technology", level=1, parent_id=None)
        db_session.add(category)
        db_session.commit()

        # Create chunks
        chunk1 = Chunk(content="Chunk 1", token_count=2, category_id=category.id)
        chunk2 = Chunk(content="Chunk 2", token_count=2, category_id=category.id)

        db_session.add_all([chunk1, chunk2])
        db_session.commit()

        # Test relationships
        assert chunk1.category == category
        assert chunk2.category == category
        assert len(category.chunks) == 2
        assert chunk1 in category.chunks
        assert chunk2 in category.chunks


class TestContextualHelperModel:
    """Test ContextualHelper model functionality."""

    def test_create_valid_contextual_helper(self, db_session) -> None:
        """Test creating a valid contextual helper."""
        helper = ContextualHelper(
            source_id="test_source_1",
            helper_text="This document discusses artificial intelligence and machine learning concepts.",
            token_count=12,
            is_user_provided=False,
            version=1,
        )

        db_session.add(helper)
        db_session.commit()

        assert helper.id is not None
        assert helper.source_id == "test_source_1"
        assert helper.token_count == 12
        assert helper.is_user_provided is False
        assert helper.version == 1
        assert helper.created_at is not None

    def test_contextual_helper_validation(self, db_session) -> None:
        """Test contextual helper validation."""
        # Test empty source_id
        with pytest.raises(ValidationError) as exc_info:
            ContextualHelper(source_id="", helper_text="Test text", token_count=2)
        assert "Source ID cannot be empty" in str(exc_info.value)

        # Test empty helper_text
        with pytest.raises(ValidationError) as exc_info:
            ContextualHelper(source_id="test_source", helper_text="", token_count=1)
        assert "Helper text cannot be empty" in str(exc_info.value)

        # Test invalid token count (zero)
        with pytest.raises(ValidationError) as exc_info:
            ContextualHelper(
                source_id="test_source", helper_text="Test text", token_count=0
            )
        assert "Helper token count must be positive" in str(exc_info.value)

        # Test invalid token count (too high)
        with pytest.raises(ValidationError) as exc_info:
            ContextualHelper(
                source_id="test_source", helper_text="Test text", token_count=301
            )
        assert "Helper token count must not exceed 300" in str(exc_info.value)

    def test_contextual_helper_unique_source_id(self, db_session) -> None:
        """Test unique source_id constraint."""
        # Create first helper
        helper1 = ContextualHelper(
            source_id="test_source", helper_text="First helper", token_count=2
        )
        db_session.add(helper1)
        db_session.commit()

        # Try to create second helper with same source_id
        helper2 = ContextualHelper(
            source_id="test_source", helper_text="Second helper", token_count=2
        )
        db_session.add(helper2)

        with pytest.raises(IntegrityError):
            db_session.commit()


class TestCategoryLimitsModel:
    """Test CategoryLimits model functionality."""

    def test_create_valid_category_limits(self, db_session) -> None:
        """Test creating valid category limits."""
        limits = CategoryLimits(level=1, max_categories=50)

        db_session.add(limits)
        db_session.commit()

        assert limits.level == 1
        assert limits.max_categories == 50
        assert limits.created_at is not None

    def test_category_limits_validation(self, db_session) -> None:
        """Test category limits validation."""
        # Test invalid level (too low)
        with pytest.raises(ValidationError) as exc_info:
            CategoryLimits(level=0, max_categories=50)
        assert "Category limit level must be between 1 and 100" in str(exc_info.value)

        # Test invalid level (too high)
        with pytest.raises(ValidationError) as exc_info:
            CategoryLimits(level=101, max_categories=50)
        assert "Category limit level must be between 1 and 100" in str(exc_info.value)

        # Test invalid max_categories
        with pytest.raises(ValidationError) as exc_info:
            CategoryLimits(level=1, max_categories=0)
        assert "Max categories must be positive" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CategoryLimits(level=1, max_categories=-1)
        assert "Max categories must be positive" in str(exc_info.value)

    def test_category_limits_unique_level(self, db_session) -> None:
        """Test unique level constraint."""
        # Create first limit
        limits1 = CategoryLimits(level=1, max_categories=50)
        db_session.add(limits1)
        db_session.commit()

        # Try to create second limit for same level
        limits2 = CategoryLimits(level=1, max_categories=100)
        db_session.add(limits2)

        with pytest.raises(IntegrityError):
            db_session.commit()
