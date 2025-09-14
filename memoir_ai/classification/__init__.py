"""
Classification module for MemoirAI.

This module provides classification capabilities for text chunks,
including batch processing, category management, and iterative workflows.
"""

from .batch_classifier import (
    BatchCategoryClassifier,
    ClassificationResult,
    BatchClassificationMetrics,
    create_batch_classifier,
    validate_batch_size,
)

from .category_manager import (
    CategoryManager,
    CategoryLimitConfig,
    CategoryStats,
    create_category_manager,
    validate_hierarchy_depth,
)

from .iterative_classifier import (
    IterativeClassificationWorkflow,
    IterativeClassificationResult,
    ClassificationWorkflowMetrics,
    create_iterative_classifier,
)

__all__ = [
    # Batch classification
    "BatchCategoryClassifier",
    "ClassificationResult",
    "BatchClassificationMetrics",
    "create_batch_classifier",
    "validate_batch_size",
    # Category management
    "CategoryManager",
    "CategoryLimitConfig",
    "CategoryStats",
    "create_category_manager",
    "validate_hierarchy_depth",
    # Iterative classification
    "IterativeClassificationWorkflow",
    "IterativeClassificationResult",
    "ClassificationWorkflowMetrics",
    "create_iterative_classifier",
]
