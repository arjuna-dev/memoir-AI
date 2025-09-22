"""
Category hierarchy management for MemoirAI.

This module provides category hierarchy management capabilities including
category creation, validation, limit enforcement, and retrieval operations.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from ..database.models import Category
from ..exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class CategoryLimitConfig:
    """Configuration for category limits."""

    global_limit: int = 128
    per_level_limits: Optional[Dict[int, int]] = None

    def get_limit_for_level(self, level: int) -> int:
        """Get the category limit for a specific level."""
        if self.per_level_limits and level in self.per_level_limits:
            return self.per_level_limits[level]
        return self.global_limit


@dataclass
class CategoryStats:
    """Statistics for category hierarchy."""

    total_categories: int
    categories_by_level: Dict[int, int]
    max_depth: int
    leaf_categories: int
    categories_at_limit: List[int]  # Levels that have reached their limits


class CategoryManager:
    """
    Manages category hierarchy operations including creation, validation, and limits.

    Features:
    - Configurable hierarchy depth (1-100 levels, default 3)
    - Global and per-level category limits
    - Category creation validation and constraint enforcement
    - Efficient category retrieval and traversal
    - Limit enforcement with existing category selection
    """

    def __init__(
        self,
        db_session: Session,
        hierarchy_depth: int = 3,
        category_limits: Optional[
            Union[int, Dict[int, int], CategoryLimitConfig]
        ] = None,
    ) -> None:
        """
        Initialize category manager.

        Args:
            db_session: Database session for operations
            hierarchy_depth: Maximum hierarchy depth (1-100, default 3)
            category_limits: Category limits configuration
        """
        self.db_session = db_session
        self.hierarchy_depth = hierarchy_depth

        # Configure category limits
        if isinstance(category_limits, CategoryLimitConfig):
            self.limits = category_limits
        elif isinstance(category_limits, dict):
            self.limits = CategoryLimitConfig(per_level_limits=category_limits)
        elif isinstance(category_limits, int):
            self.limits = CategoryLimitConfig(global_limit=category_limits)
        else:
            self.limits = CategoryLimitConfig()  # Use defaults

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate category manager configuration."""
        if (
            not isinstance(self.hierarchy_depth, int)
            or self.hierarchy_depth < 1
            or self.hierarchy_depth > 100
        ):
            raise ValidationError(
                "hierarchy_depth must be between 1 and 100",
                field="hierarchy_depth",
                value=self.hierarchy_depth,
            )

        if self.limits.global_limit <= 0:
            raise ValidationError(
                "global_limit must be positive",
                field="global_limit",
                value=self.limits.global_limit,
            )

        if self.limits.per_level_limits:
            for level, limit in self.limits.per_level_limits.items():
                if (
                    not isinstance(level, int)
                    or level < 1
                    or level > self.hierarchy_depth
                ):
                    raise ValidationError(
                        f"per_level_limits level {level} must be between 1 and {self.hierarchy_depth}",
                        field="per_level_limits",
                        value=level,
                    )

                if not isinstance(limit, int) or limit <= 0:
                    raise ValidationError(
                        f"per_level_limits limit for level {level} must be positive",
                        field="per_level_limits",
                        value=limit,
                    )

    def get_existing_categories(
        self, level: int, parent_id: Optional[int] = None, limit: Optional[int] = None
    ) -> List[Category]:
        """
        Retrieve existing categories at specified level and parent.

        Args:
            level: Hierarchy level (1-based)
            parent_id: Parent category ID (None for level 1)
            limit: Maximum number of categories to return

        Returns:
            List of existing categories
        """
        try:
            query = self.db_session.query(Category).filter(Category.level == level)

            if level == 1:
                query = query.filter(Category.parent_id.is_(None))
            else:
                if parent_id is None:
                    raise ValidationError(
                        f"parent_id required for level {level}",
                        field="parent_id",
                        value=parent_id,
                    )
                query = query.filter(Category.parent_id == parent_id)

            # Order by name for consistent presentation
            query = query.order_by(Category.name)

            if limit:
                query = query.limit(limit)

            categories = cast(List[Category], query.all())

            logger.debug(
                f"Retrieved {len(categories)} categories at level {level}, parent {parent_id}"
            )
            return categories

        except Exception as e:
            logger.error(f"Failed to retrieve categories: {e}")
            raise DatabaseError(
                f"Failed to retrieve categories at level {level}: {str(e)}",
                operation="select",
                table="categories",
            )

    def can_create_category(self, level: int, parent_id: Optional[int] = None) -> bool:
        """
        Check if a new category can be created at the specified level.

        Args:
            level: Hierarchy level (1-based)
            parent_id: Parent category ID (None for level 1)

        Returns:
            True if category can be created, False if limit reached
        """
        try:
            existing_count = self._count_categories_at_level(level, parent_id)
            limit = self.limits.get_limit_for_level(level)

            can_create = existing_count < limit

            logger.debug(
                f"Level {level} (parent {parent_id}): {existing_count}/{limit} categories, "
                f"can_create={can_create}"
            )

            return can_create

        except Exception as e:
            logger.error(f"Failed to check category creation limit: {e}")
            return False

    def create_category(
        self,
        name: str,
        level: int,
        parent_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Category:
        """
        Create a new category with validation and constraint enforcement.

        Args:
            name: Category name
            level: Hierarchy level (1-based)
            parent_id: Parent category ID (None for level 1)
            metadata: Optional metadata dictionary

        Returns:
            Created category

        Raises:
            ValidationError: If validation fails
            DatabaseError: If database operation fails
        """
        # Validate inputs
        self._validate_category_creation(name, level, parent_id)

        # Check if category can be created (limit enforcement)
        if not self.can_create_category(level, parent_id):
            limit = self.limits.get_limit_for_level(level)
            raise ValidationError(
                f"Category limit ({limit}) reached at level {level}. "
                f"Must select from existing categories.",
                field="category_limit",
                value={"level": level, "limit": limit},
            )

        # Check for duplicate names at same level/parent
        if self._category_name_exists(name, level, parent_id):
            raise ValidationError(
                f"Category '{name}' already exists at level {level} with parent {parent_id}",
                field="name",
                value=name,
            )

        try:
            # Create category
            category = Category(
                name=name.strip(),
                level=level,
                parent_id=parent_id,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            self.db_session.add(category)
            self.db_session.flush()  # Get the ID without committing

            logger.info(
                f"Created category '{name}' at level {level} (ID: {category.id})"
            )
            return category

        except Exception as e:
            logger.error(f"Failed to create category '{name}': {e}")
            self.db_session.rollback()
            raise DatabaseError(
                f"Failed to create category '{name}': {str(e)}",
                operation="insert",
                table="categories",
            )

    def get_category_by_id(self, category_id: int) -> Optional[Category]:
        """
        Retrieve category by ID.

        Args:
            category_id: Category ID

        Returns:
            Category if found, None otherwise
        """
        try:
            category = cast(
                Optional[Category],
                self.db_session.query(Category)
                .filter(Category.id == category_id)
                .first(),
            )
            return category
        except Exception as e:
            logger.error(f"Failed to retrieve category {category_id}: {e}")
            return None

    def get_category_path(self, category: Category) -> List[Category]:
        """
        Get the full path from root to the specified category.

        Args:
            category: Target category

        Returns:
            List of categories from root to target
        """
        path: List[Category] = []
        current: Optional[Category] = category

        while current:
            path.insert(0, current)  # Insert at beginning to build path from root

            if current.parent_id:
                current = self.get_category_by_id(current.parent_id)
            else:
                current = None

        return path

    def get_leaf_categories(self, parent_id: Optional[int] = None) -> List[Category]:
        """
        Get all leaf categories (categories at maximum depth).

        Args:
            parent_id: Optional parent to filter by

        Returns:
            List of leaf categories
        """
        try:
            query = self.db_session.query(Category).filter(
                Category.level == self.hierarchy_depth
            )

            if parent_id:
                # Get all descendants of the parent
                query = query.join(
                    Category, Category.parent_id == parent_id, isouter=True
                )

            categories = cast(List[Category], query.all())
            return categories

        except Exception as e:
            logger.error(f"Failed to retrieve leaf categories: {e}")
            return []

    def is_leaf_category(self, category: Category) -> bool:
        """
        Check if category is a leaf (at maximum configured depth).

        Args:
            category: Category to check

        Returns:
            True if category is at leaf level
        """
        level = cast(int, category.level)
        return level == self.hierarchy_depth

    def get_category_stats(self) -> CategoryStats:
        """
        Get statistics about the category hierarchy.

        Returns:
            CategoryStats with hierarchy information
        """
        try:
            # Total categories
            total_categories = (
                self.db_session.query(func.count(Category.id)).scalar() or 0
            )

            # Categories by level
            level_counts: Dict[int, int] = {}
            for level in range(1, self.hierarchy_depth + 1):
                count = (
                    self.db_session.query(func.count(Category.id))
                    .filter(Category.level == level)
                    .scalar()
                    or 0
                )
                level_counts[level] = count

            # Max depth with categories
            max_depth = 0
            for level in range(self.hierarchy_depth, 0, -1):
                if level_counts.get(level, 0) > 0:
                    max_depth = level
                    break

            # Leaf categories
            leaf_categories = level_counts.get(self.hierarchy_depth, 0)

            # Levels at limit
            categories_at_limit: List[int] = []
            for level in range(1, self.hierarchy_depth + 1):
                limit = self.limits.get_limit_for_level(level)
                if level_counts.get(level, 0) >= limit:
                    categories_at_limit.append(level)

            return CategoryStats(
                total_categories=total_categories,
                categories_by_level=level_counts,
                max_depth=max_depth,
                leaf_categories=leaf_categories,
                categories_at_limit=categories_at_limit,
            )

        except Exception as e:
            logger.error(f"Failed to get category stats: {e}")
            return CategoryStats(
                total_categories=0,
                categories_by_level={},
                max_depth=0,
                leaf_categories=0,
                categories_at_limit=[],
            )

    def validate_category_hierarchy(self) -> List[str]:
        """
        Validate the entire category hierarchy for consistency.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []

        try:
            # Check all categories
            categories = cast(List[Category], self.db_session.query(Category).all())

            for category in categories:
                # Check level constraints
                if category.level < 1 or category.level > self.hierarchy_depth:
                    errors.append(
                        f"Category {category.id} has invalid level {category.level}"
                    )

                # Check parent relationships
                if category.level == 1:
                    if category.parent_id is not None:
                        errors.append(
                            f"Level 1 category {category.id} has parent_id {category.parent_id}"
                        )
                else:
                    if category.parent_id is None:
                        errors.append(
                            f"Level {category.level} category {category.id} missing parent_id"
                        )
                    else:
                        parent = self.get_category_by_id(category.parent_id)
                        if not parent:
                            errors.append(
                                f"Category {category.id} has invalid parent_id {category.parent_id}"
                            )
                        elif parent.level != category.level - 1:
                            errors.append(
                                f"Category {category.id} (level {category.level}) "
                                f"has parent {parent.id} (level {parent.level})"
                            )

            return errors

        except Exception as e:
            logger.error(f"Failed to validate hierarchy: {e}")
            return [f"Validation failed: {str(e)}"]

    def _validate_category_creation(
        self, name: str, level: int, parent_id: Optional[int]
    ) -> None:
        """Validate category creation parameters."""
        if not name or not name.strip():
            raise ValidationError(
                "Category name cannot be empty", field="name", value=name
            )

        if len(name.strip()) > 255:
            raise ValidationError(
                "Category name cannot exceed 255 characters", field="name", value=name
            )

        if level < 1 or level > self.hierarchy_depth:
            raise ValidationError(
                f"Level must be between 1 and {self.hierarchy_depth}",
                field="level",
                value=level,
            )

        if level == 1 and parent_id is not None:
            raise ValidationError(
                "Level 1 categories cannot have a parent",
                field="parent_id",
                value=parent_id,
            )

        if level > 1 and parent_id is None:
            raise ValidationError(
                f"Level {level} categories must have a parent",
                field="parent_id",
                value=parent_id,
            )

        # Validate parent exists and is at correct level
        if parent_id is not None:
            parent = self.get_category_by_id(parent_id)
            if not parent:
                raise ValidationError(
                    f"Parent category {parent_id} not found",
                    field="parent_id",
                    value=parent_id,
                )

            if parent.level != level - 1:
                raise ValidationError(
                    f"Parent category must be at level {level - 1}, got level {parent.level}",
                    field="parent_id",
                    value=parent_id,
                )

    def _count_categories_at_level(
        self, level: int, parent_id: Optional[int] = None
    ) -> int:
        """Count existing categories at specified level and parent."""
        try:
            query = self.db_session.query(func.count(Category.id)).filter(
                Category.level == level
            )

            if level == 1:
                query = query.filter(Category.parent_id.is_(None))
            else:
                query = query.filter(Category.parent_id == parent_id)

            count = query.scalar() or 0
            return count

        except Exception as e:
            logger.error(f"Failed to count categories: {e}")
            return 0

    def _category_name_exists(
        self, name: str, level: int, parent_id: Optional[int] = None
    ) -> bool:
        """Check if category name already exists at the same level/parent."""
        try:
            query = self.db_session.query(Category).filter(
                and_(Category.name == name.strip(), Category.level == level)
            )

            if level == 1:
                query = query.filter(Category.parent_id.is_(None))
            else:
                query = query.filter(Category.parent_id == parent_id)

            existing = query.first()
            return existing is not None

        except Exception as e:
            logger.error(f"Failed to check category name existence: {e}")
            return False

    def get_category_limit(self, level: int) -> int:
        """
        Get the category limit for a specific level.

        Args:
            level: Hierarchy level

        Returns:
            Category limit for the level
        """
        return self.limits.get_limit_for_level(level)

    def get_categories_for_llm_prompt(
        self,
        level: int,
        parent_id: Optional[int] = None,
        max_categories: Optional[int] = None,
    ) -> Tuple[List[Category], bool]:
        """
        Get categories formatted for LLM prompts with limit information.

        Args:
            level: Hierarchy level
            parent_id: Parent category ID
            max_categories: Maximum categories to return for prompt

        Returns:
            Tuple of (categories, can_create_new)
        """
        categories = self.get_existing_categories(level, parent_id, max_categories)
        can_create_new = self.can_create_category(level, parent_id)

        return categories, can_create_new


# Utility functions
def create_category_manager(
    db_session: Session, hierarchy_depth: int = 3, **kwargs: Any
) -> CategoryManager:
    """
    Create a category manager with default configuration.

    Args:
        db_session: Database session
        hierarchy_depth: Maximum hierarchy depth
        **kwargs: Additional configuration options

    Returns:
        Configured CategoryManager
    """
    return CategoryManager(
        db_session=db_session, hierarchy_depth=hierarchy_depth, **kwargs
    )


def validate_hierarchy_depth(depth: int) -> bool:
    """
    Validate hierarchy depth parameter.

    Args:
        depth: Hierarchy depth to validate

    Returns:
        True if valid, False otherwise
    """
    return isinstance(depth, int) and 1 <= depth <= 100
