"""
Tests for category manager.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock

import pytest

from memoir_ai.classification.category_manager import (
    CategoryLimitConfig,
    CategoryManager,
    CategoryStats,
    create_category_manager,
    validate_hierarchy_depth,
)
from memoir_ai.database.models import Category
from memoir_ai.exceptions import DatabaseError, ValidationError


class TestCategoryLimitConfig:
    """Test CategoryLimitConfig functionality."""

    def test_category_limit_config_defaults(self) -> None:
        """Test CategoryLimitConfig with defaults."""
        config = CategoryLimitConfig()

        assert config.global_limit == 128
        assert config.per_level_limits is None

    def test_category_limit_config_custom(self) -> None:
        """Test CategoryLimitConfig with custom values."""
        config = CategoryLimitConfig(
            global_limit=50, per_level_limits={1: 10, 2: 20, 3: 30}
        )

        assert config.global_limit == 50
        assert config.per_level_limits == {1: 10, 2: 20, 3: 30}

    def test_get_limit_for_level_global(self) -> None:
        """Test getting limit using global configuration."""
        config = CategoryLimitConfig(global_limit=100)

        assert config.get_limit_for_level(1) == 100
        assert config.get_limit_for_level(2) == 100
        assert config.get_limit_for_level(5) == 100

    def test_get_limit_for_level_per_level(self) -> None:
        """Test getting limit using per-level configuration."""
        config = CategoryLimitConfig(global_limit=100, per_level_limits={1: 10, 3: 30})

        assert config.get_limit_for_level(1) == 10  # Per-level override
        assert config.get_limit_for_level(2) == 100  # Global default
        assert config.get_limit_for_level(3) == 30  # Per-level override


class TestCategoryStats:
    """Test CategoryStats data class."""

    def test_category_stats_creation(self) -> None:
        """Test creating CategoryStats."""
        stats = CategoryStats(
            total_categories=25,
            categories_by_level={1: 5, 2: 10, 3: 10},
            max_depth=3,
            leaf_categories=10,
            categories_at_limit=[2, 3],
        )

        assert stats.total_categories == 25
        assert stats.categories_by_level == {1: 5, 2: 10, 3: 10}
        assert stats.max_depth == 3
        assert stats.leaf_categories == 10
        assert stats.categories_at_limit == [2, 3]


class TestCategoryManager:
    """Test CategoryManager functionality."""

    def create_mock_session(self) -> None:
        """Create a mock database session."""
        session = Mock()
        session.query.return_value = session
        session.filter.return_value = session
        session.order_by.return_value = session
        session.limit.return_value = session
        session.all.return_value = []
        session.first.return_value = None
        session.scalar.return_value = 0
        return session

    def test_category_manager_initialization_defaults(self) -> None:
        """Test CategoryManager initialization with defaults."""
        session = self.create_mock_session()

        manager = CategoryManager(session)

        assert manager.db_session == session
        assert manager.hierarchy_depth == 3
        assert manager.limits.global_limit == 128
        assert manager.limits.per_level_limits is None

    def test_category_manager_initialization_custom(self) -> None:
        """Test CategoryManager initialization with custom parameters."""
        session = self.create_mock_session()

        manager = CategoryManager(
            session, hierarchy_depth=5, category_limits={1: 10, 2: 20}
        )

        assert manager.hierarchy_depth == 5
        assert manager.limits.per_level_limits == {1: 10, 2: 20}

    def test_category_manager_initialization_with_config(self) -> None:
        """Test CategoryManager initialization with CategoryLimitConfig."""
        session = self.create_mock_session()
        config = CategoryLimitConfig(global_limit=50, per_level_limits={1: 5})

        manager = CategoryManager(session, category_limits=config)

        assert manager.limits.global_limit == 50
        assert manager.limits.per_level_limits == {1: 5}

    def test_category_manager_initialization_validation(self) -> None:
        """Test CategoryManager initialization validation."""
        session = self.create_mock_session()

        # Test invalid hierarchy depth
        with pytest.raises(ValidationError) as exc_info:
            CategoryManager(session, hierarchy_depth=0)
        assert "hierarchy_depth must be between 1 and 100" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CategoryManager(session, hierarchy_depth=101)
        assert "hierarchy_depth must be between 1 and 100" in str(exc_info.value)

        # Test invalid global limit
        with pytest.raises(ValidationError) as exc_info:
            CategoryManager(session, category_limits=0)
        assert "global_limit must be positive" in str(exc_info.value)

    def test_get_existing_categories_level_1(self) -> None:
        """Test getting existing categories at level 1."""
        session = self.create_mock_session()

        # Mock categories
        categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="Science", level=1, parent_id=None),
        ]
        session.all.return_value = categories

        manager = CategoryManager(session)
        result = manager.get_existing_categories(1)

        assert len(result) == 2
        assert result[0].name == "Technology"
        assert result[1].name == "Science"

    def test_get_existing_categories_level_2(self) -> None:
        """Test getting existing categories at level 2."""
        session = self.create_mock_session()

        categories = [
            Category(id=3, name="AI", level=2, parent_id=1),
            Category(id=4, name="ML", level=2, parent_id=1),
        ]
        session.all.return_value = categories

        manager = CategoryManager(session)
        result = manager.get_existing_categories(2, parent_id=1)

        assert len(result) == 2
        assert result[0].name == "AI"
        assert result[1].name == "ML"

    def test_get_existing_categories_level_2_no_parent(self) -> None:
        """Test getting categories at level 2 without parent raises error."""
        session = self.create_mock_session()
        manager = CategoryManager(session)

        with pytest.raises(DatabaseError) as exc_info:
            manager.get_existing_categories(2)
        assert "parent_id required for level 2" in str(exc_info.value)

    def test_can_create_category_under_limit(self) -> None:
        """Test can_create_category when under limit."""
        session = self.create_mock_session()
        session.scalar.return_value = 5  # 5 existing categories

        manager = CategoryManager(session, category_limits=10)

        assert manager.can_create_category(1) is True

    def test_can_create_category_at_limit(self) -> None:
        """Test can_create_category when at limit."""
        session = self.create_mock_session()
        session.scalar.return_value = 10  # 10 existing categories

        manager = CategoryManager(session, category_limits=10)

        assert manager.can_create_category(1) is False

    def test_create_category_success(self) -> None:
        """Test successful category creation."""
        session = self.create_mock_session()
        session.scalar.return_value = 5  # Under limit

        # Mock no existing category with same name
        session.first.return_value = None

        manager = CategoryManager(session, category_limits=10)

        category = manager.create_category("Technology", 1)

        assert category.name == "Technology"
        assert category.level == 1
        assert category.parent_id is None
        session.add.assert_called_once()
        session.flush.assert_called_once()

    def test_create_category_at_limit(self) -> None:
        """Test category creation when at limit."""
        session = self.create_mock_session()
        session.scalar.return_value = 10  # At limit

        manager = CategoryManager(session, category_limits=10)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Technology", 1)
        assert "Category limit (10) reached" in str(exc_info.value)

    def test_create_category_duplicate_name(self) -> None:
        """Test category creation with duplicate name."""
        session = self.create_mock_session()
        session.scalar.return_value = 5  # Under limit

        # Mock existing category with same name
        existing_category = Category(id=1, name="Technology", level=1, parent_id=None)
        session.first.return_value = existing_category

        manager = CategoryManager(session, category_limits=10)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Technology", 1)
        assert "already exists" in str(exc_info.value)

    def test_create_category_validation_empty_name(self) -> None:
        """Test category creation validation with empty name."""
        session = self.create_mock_session()
        manager = CategoryManager(session)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("", 1)
        assert "Category name cannot be empty" in str(exc_info.value)

    def test_create_category_validation_long_name(self) -> None:
        """Test category creation validation with long name."""
        session = self.create_mock_session()
        manager = CategoryManager(session)

        long_name = "x" * 256  # Exceeds 255 character limit

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category(long_name, 1)
        assert "cannot exceed 255 characters" in str(exc_info.value)

    def test_create_category_validation_invalid_level(self) -> None:
        """Test category creation validation with invalid level."""
        session = self.create_mock_session()
        manager = CategoryManager(session, hierarchy_depth=3)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 0)
        assert "Level must be between 1 and 3" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 4)
        assert "Level must be between 1 and 3" in str(exc_info.value)

    def test_create_category_validation_level_1_with_parent(self) -> None:
        """Test category creation validation for level 1 with parent."""
        session = self.create_mock_session()
        manager = CategoryManager(session)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 1, parent_id=1)
        assert "Level 1 categories cannot have a parent" in str(exc_info.value)

    def test_create_category_validation_level_2_without_parent(self) -> None:
        """Test category creation validation for level 2 without parent."""
        session = self.create_mock_session()
        manager = CategoryManager(session)

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 2)
        assert "Level 2 categories must have a parent" in str(exc_info.value)

    def test_create_category_validation_invalid_parent(self) -> None:
        """Test category creation validation with invalid parent."""
        session = self.create_mock_session()

        # Mock get_category_by_id to return None (parent not found)
        def mock_get_category_by_id(category_id) -> None:
            return None

        manager = CategoryManager(session)
        manager.get_category_by_id = mock_get_category_by_id

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 2, parent_id=999)
        assert "Parent category 999 not found" in str(exc_info.value)

    def test_create_category_validation_wrong_parent_level(self) -> None:
        """Test category creation validation with wrong parent level."""
        session = self.create_mock_session()

        # Mock parent at wrong level
        parent = Category(id=1, name="Parent", level=2, parent_id=None)

        def mock_get_category_by_id(category_id) -> None:
            return parent if category_id == 1 else None

        manager = CategoryManager(session)
        manager.get_category_by_id = mock_get_category_by_id

        with pytest.raises(ValidationError) as exc_info:
            manager.create_category("Test", 2, parent_id=1)
        assert "Parent category must be at level 1" in str(exc_info.value)

    def test_get_category_by_id_found(self) -> None:
        """Test getting category by ID when found."""
        session = self.create_mock_session()

        category = Category(id=1, name="Technology", level=1, parent_id=None)
        session.first.return_value = category

        manager = CategoryManager(session)
        result = manager.get_category_by_id(1)

        assert result == category

    def test_get_category_by_id_not_found(self) -> None:
        """Test getting category by ID when not found."""
        session = self.create_mock_session()
        session.first.return_value = None

        manager = CategoryManager(session)
        result = manager.get_category_by_id(999)

        assert result is None

    def test_get_category_path(self) -> None:
        """Test getting category path from root to target."""
        session = self.create_mock_session()

        # Create category hierarchy
        root = Category(id=1, name="Technology", level=1, parent_id=None)
        child = Category(id=2, name="AI", level=2, parent_id=1)
        grandchild = Category(id=3, name="ML", level=3, parent_id=2)

        def mock_get_category_by_id(category_id) -> None:
            categories = {1: root, 2: child, 3: grandchild}
            return categories.get(category_id)

        manager = CategoryManager(session)
        manager.get_category_by_id = mock_get_category_by_id

        path = manager.get_category_path(grandchild)

        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child
        assert path[2] == grandchild

    def test_is_leaf_category(self) -> None:
        """Test checking if category is a leaf."""
        session = self.create_mock_session()
        manager = CategoryManager(session, hierarchy_depth=3)

        level_1_cat = Category(id=1, name="Tech", level=1, parent_id=None)
        level_3_cat = Category(id=3, name="ML", level=3, parent_id=2)

        assert manager.is_leaf_category(level_1_cat) is False
        assert manager.is_leaf_category(level_3_cat) is True

    def test_get_category_limit(self) -> None:
        """Test getting category limit for level."""
        session = self.create_mock_session()

        manager = CategoryManager(session, category_limits={1: 10, 2: 20, 3: 30})

        assert manager.get_category_limit(1) == 10
        assert manager.get_category_limit(2) == 20
        assert manager.get_category_limit(4) == 128  # Global default

    def test_get_categories_for_llm_prompt(self) -> None:
        """Test getting categories formatted for LLM prompts."""
        session = self.create_mock_session()

        categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="Science", level=1, parent_id=None),
        ]
        session.all.return_value = categories
        session.scalar.return_value = 2  # Under limit of 10

        manager = CategoryManager(session, category_limits=10)

        result_categories, can_create_new = manager.get_categories_for_llm_prompt(1)

        assert len(result_categories) == 2
        assert can_create_new is True

    def test_validate_category_hierarchy_valid(self) -> None:
        """Test validating a valid category hierarchy."""
        session = self.create_mock_session()

        categories = [
            Category(id=1, name="Technology", level=1, parent_id=None),
            Category(id=2, name="AI", level=2, parent_id=1),
            Category(id=3, name="ML", level=3, parent_id=2),
        ]
        session.all.return_value = categories

        def mock_get_category_by_id(category_id) -> None:
            categories_dict = {1: categories[0], 2: categories[1], 3: categories[2]}
            return categories_dict.get(category_id)

        manager = CategoryManager(session, hierarchy_depth=3)
        manager.get_category_by_id = mock_get_category_by_id

        errors = manager.validate_category_hierarchy()

        assert errors == []

    def test_validate_category_hierarchy_invalid_level(self) -> None:
        """Test validating hierarchy with invalid level."""
        session = self.create_mock_session()

        categories = [
            Category(id=1, name="Technology", level=5, parent_id=None),  # Invalid level
        ]
        session.all.return_value = categories

        manager = CategoryManager(session, hierarchy_depth=3)

        errors = manager.validate_category_hierarchy()

        assert len(errors) > 0
        assert "invalid level 5" in errors[0]

    def test_validate_category_hierarchy_invalid_parent(self) -> None:
        """Test validating hierarchy with invalid parent relationship."""
        session = self.create_mock_session()

        categories = [
            Category(
                id=1, name="Technology", level=1, parent_id=999
            ),  # Level 1 with parent
        ]
        session.all.return_value = categories

        manager = CategoryManager(session, hierarchy_depth=3)

        errors = manager.validate_category_hierarchy()

        assert len(errors) > 0
        assert "has parent_id 999" in errors[0]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_category_manager(self) -> None:
        """Test category manager creation utility."""
        session = Mock()

        manager = create_category_manager(
            session, hierarchy_depth=5, category_limits=50
        )

        assert isinstance(manager, CategoryManager)
        assert manager.hierarchy_depth == 5
        assert manager.limits.global_limit == 50

    def test_validate_hierarchy_depth(self) -> None:
        """Test hierarchy depth validation utility."""
        assert validate_hierarchy_depth(1) is True
        assert validate_hierarchy_depth(3) is True
        assert validate_hierarchy_depth(100) is True

        assert validate_hierarchy_depth(0) is False
        assert validate_hierarchy_depth(101) is False
        assert validate_hierarchy_depth(-1) is False
        assert validate_hierarchy_depth("3") is False
        assert validate_hierarchy_depth(None) is False
