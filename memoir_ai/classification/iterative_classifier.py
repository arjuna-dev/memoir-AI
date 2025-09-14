"""
Iterative classification workflow for MemoirAI.

This module provides the iterative classification workflow that processes
chunks through hierarchy levels sequentially, integrating contextual helpers
and category management.
"""

import logging
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from ..text_processing.chunker import TextChunk
from ..database.models import Category, Chunk
from ..llm.schemas import CategorySelection
from ..llm.agents import create_classification_agent
from .batch_classifier import BatchCategoryClassifier, ClassificationResult
from .category_manager import CategoryManager
from ..exceptions import ClassificationError, ValidationError, DatabaseError


logger = logging.getLogger(__name__)


@dataclass
class IterativeClassificationResult:
    """Result of iterative classification for a single chunk."""

    chunk: TextChunk
    category_path: List[Category]
    final_category: Category
    success: bool
    total_latency_ms: int
    llm_calls: int
    levels_processed: int
    error: Optional[str] = None
    level_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class ClassificationWorkflowMetrics:
    """Metrics for classification workflow operations."""

    workflow_id: str
    chunks_processed: int
    chunks_successful: int
    chunks_failed: int
    total_latency_ms: int
    total_llm_calls: int
    average_levels_per_chunk: float
    timestamp: datetime
    model_name: str


class IterativeClassificationWorkflow:
    """
    Iterative classification workflow that processes chunks through hierarchy levels.

    Features:
    - Sequential processing through hierarchy levels (1 to max depth)
    - Integration with contextual helpers for improved classification
    - Category reuse logic with existing category presentation
    - Batch processing support for efficiency
    - Storage of chunks linked to leaf-level categories only
    - Comprehensive metrics and error handling
    """

    def __init__(
        self,
        db_session: Session,
        category_manager: CategoryManager,
        model_name: str = "openai:gpt-4",
        use_batch_processing: bool = True,
        batch_size: int = 5,
        max_retries: int = 3,
        temperature: float = 0.0,
    ):
        """
        Initialize iterative classification workflow.

        Args:
            db_session: Database session for operations
            category_manager: Category manager for hierarchy operations
            model_name: LLM model to use for classification
            use_batch_processing: Whether to use batch processing when possible
            batch_size: Batch size for batch processing
            max_retries: Maximum retry attempts for failed classifications
            temperature: Temperature for LLM generation
        """
        self.db_session = db_session
        self.category_manager = category_manager
        self.model_name = model_name
        self.use_batch_processing = use_batch_processing
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.temperature = temperature

        # Create classifiers
        if use_batch_processing:
            self.batch_classifier = BatchCategoryClassifier(
                model_name=model_name,
                batch_size=batch_size,
                max_retries=max_retries,
                hierarchy_depth=category_manager.hierarchy_depth,
                max_categories_per_level=category_manager.limits.global_limit,
                temperature=temperature,
            )

        self.single_classifier = create_classification_agent(model_name)

        # Metrics tracking
        self.metrics_history: List[ClassificationWorkflowMetrics] = []

    async def classify_chunks(
        self,
        chunks: List[TextChunk],
        contextual_helper: str,
        source_id: Optional[str] = None,
    ) -> List[IterativeClassificationResult]:
        """
        Classify multiple chunks through the complete hierarchy.

        Args:
            chunks: List of text chunks to classify
            contextual_helper: Contextual information about the source
            source_id: Optional source identifier for tracking

        Returns:
            List of classification results for all chunks
        """
        if not chunks:
            return []

        workflow_id = f"workflow_{int(time.time())}_{len(chunks)}"
        start_time = time.time()

        logger.info(
            f"Starting iterative classification workflow {workflow_id} for {len(chunks)} chunks"
        )

        try:
            results = []
            total_llm_calls = 0

            for chunk in chunks:
                chunk_start_time = time.time()

                try:
                    result = await self._classify_single_chunk(
                        chunk, contextual_helper, workflow_id
                    )
                    results.append(result)
                    total_llm_calls += result.llm_calls

                    # Store successful classification
                    if result.success:
                        await self._store_chunk_classification(
                            chunk, result.final_category, source_id
                        )

                except Exception as e:
                    logger.error(f"Failed to classify chunk in {workflow_id}: {e}")

                    # Create failed result
                    failed_result = IterativeClassificationResult(
                        chunk=chunk,
                        category_path=[],
                        final_category=None,
                        success=False,
                        total_latency_ms=int((time.time() - chunk_start_time) * 1000),
                        llm_calls=0,
                        levels_processed=0,
                        error=str(e),
                    )
                    results.append(failed_result)

            # Record workflow metrics
            total_latency_ms = int((time.time() - start_time) * 1000)
            self._record_workflow_metrics(
                workflow_id, chunks, results, total_latency_ms, total_llm_calls
            )

            logger.info(
                f"Completed workflow {workflow_id}: {len([r for r in results if r.success])}/{len(results)} successful"
            )

            return results

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise ClassificationError(
                f"Classification workflow failed: {str(e)}", retry_count=0
            )

    async def _classify_single_chunk(
        self,
        chunk: TextChunk,
        contextual_helper: str,
        workflow_id: str,
    ) -> IterativeClassificationResult:
        """
        Classify a single chunk through all hierarchy levels.

        Args:
            chunk: Text chunk to classify
            contextual_helper: Contextual information
            workflow_id: Workflow identifier for tracking

        Returns:
            Classification result for the chunk
        """
        start_time = time.time()
        category_path = []
        level_results = []
        total_llm_calls = 0
        current_parent_id = None

        try:
            # Process each level sequentially
            for level in range(1, self.category_manager.hierarchy_depth + 1):
                level_start_time = time.time()

                # Get existing categories at this level
                existing_categories, can_create_new = (
                    self.category_manager.get_categories_for_llm_prompt(
                        level, current_parent_id
                    )
                )

                # Classify at this level
                category_name = await self._classify_at_level(
                    chunk, level, existing_categories, contextual_helper, can_create_new
                )

                total_llm_calls += 1
                level_latency = int((time.time() - level_start_time) * 1000)

                # Find or create category
                category = await self._get_or_create_category(
                    category_name, level, current_parent_id, can_create_new
                )

                category_path.append(category)
                current_parent_id = category.id

                # Record level result
                level_results.append(
                    {
                        "level": level,
                        "category_name": category_name,
                        "category_id": category.id,
                        "existing_categories": len(existing_categories),
                        "can_create_new": can_create_new,
                        "latency_ms": level_latency,
                    }
                )

                logger.debug(
                    f"Level {level} classification: '{category_name}' "
                    f"(ID: {category.id}, {level_latency}ms)"
                )

            # Create successful result
            total_latency_ms = int((time.time() - start_time) * 1000)

            result = IterativeClassificationResult(
                chunk=chunk,
                category_path=category_path,
                final_category=category_path[-1] if category_path else None,
                success=True,
                total_latency_ms=total_latency_ms,
                llm_calls=total_llm_calls,
                levels_processed=len(category_path),
                level_results=level_results,
            )

            logger.debug(
                f"Successfully classified chunk through {len(category_path)} levels "
                f"in {total_latency_ms}ms with {total_llm_calls} LLM calls"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to classify chunk: {e}")

            total_latency_ms = int((time.time() - start_time) * 1000)

            return IterativeClassificationResult(
                chunk=chunk,
                category_path=category_path,
                final_category=category_path[-1] if category_path else None,
                success=False,
                total_latency_ms=total_latency_ms,
                llm_calls=total_llm_calls,
                levels_processed=len(category_path),
                error=str(e),
                level_results=level_results,
            )

    async def _classify_at_level(
        self,
        chunk: TextChunk,
        level: int,
        existing_categories: List[Category],
        contextual_helper: str,
        can_create_new: bool,
    ) -> str:
        """
        Classify chunk at a specific hierarchy level.

        Args:
            chunk: Text chunk to classify
            level: Current hierarchy level
            existing_categories: Existing categories at this level
            contextual_helper: Contextual information
            can_create_new: Whether new categories can be created

        Returns:
            Selected category name
        """
        # Create classification prompt
        prompt = self._create_level_prompt(
            chunk, level, existing_categories, contextual_helper, can_create_new
        )

        # Call LLM
        try:
            response = await self.single_classifier.run_async(prompt)

            if not response.data or not response.data.category:
                raise ClassificationError(
                    f"Empty category response at level {level}", retry_count=0
                )

            category_name = response.data.category.strip()

            # Validate category selection
            if not can_create_new:
                # Must select from existing categories
                existing_names = [cat.name for cat in existing_categories]
                if category_name not in existing_names:
                    # Try to find closest match or use first existing
                    if existing_categories:
                        category_name = existing_categories[0].name
                        logger.warning(
                            f"LLM selected non-existing category, using '{category_name}' instead"
                        )
                    else:
                        raise ClassificationError(
                            f"No existing categories available at level {level}",
                            retry_count=0,
                        )

            return category_name

        except Exception as e:
            logger.error(f"Classification failed at level {level}: {e}")
            raise ClassificationError(
                f"Classification failed at level {level}: {str(e)}", retry_count=0
            )

    def _create_level_prompt(
        self,
        chunk: TextChunk,
        level: int,
        existing_categories: List[Category],
        contextual_helper: str,
        can_create_new: bool,
    ) -> str:
        """Create classification prompt for a specific level."""
        context_parts = []

        if contextual_helper:
            context_parts.append(f"Document Context: {contextual_helper}")

        context_parts.append(f"Classification Level: {level}")

        if existing_categories:
            category_names = [cat.name for cat in existing_categories]
            context_parts.append(f"Existing Categories: {', '.join(category_names)}")

            if can_create_new:
                context_parts.append(
                    "Please select from existing categories when possible to avoid duplicates, "
                    "or create a new category if none fit well."
                )
            else:
                limit = self.category_manager.get_category_limit(level)
                context_parts.append(
                    f"Category limit ({limit}) reached. You MUST select from existing categories only."
                )
        else:
            context_parts.append(
                "No existing categories at this level. You may create a new category."
            )

        context_section = "\n".join(context_parts)

        prompt = f"""{context_section}

Please classify the following text into the most appropriate category for level {level}:

Text: {chunk.content}

Provide only the category name."""

        return prompt

    async def _get_or_create_category(
        self,
        category_name: str,
        level: int,
        parent_id: Optional[int],
        can_create_new: bool,
    ) -> Category:
        """
        Get existing category or create new one if allowed.

        Args:
            category_name: Name of the category
            level: Hierarchy level
            parent_id: Parent category ID
            can_create_new: Whether new categories can be created

        Returns:
            Category (existing or newly created)
        """
        try:
            # First try to find existing category
            existing_categories = self.category_manager.get_existing_categories(
                level, parent_id
            )

            for category in existing_categories:
                if category.name.lower() == category_name.lower():
                    logger.debug(
                        f"Using existing category '{category.name}' (ID: {category.id})"
                    )
                    return category

            # Category doesn't exist - create if allowed
            if can_create_new:
                category = self.category_manager.create_category(
                    name=category_name, level=level, parent_id=parent_id
                )
                logger.info(
                    f"Created new category '{category_name}' at level {level} (ID: {category.id})"
                )
                return category
            else:
                # Must use existing category - use first available
                if existing_categories:
                    category = existing_categories[0]
                    logger.warning(
                        f"Cannot create '{category_name}', using existing '{category.name}' instead"
                    )
                    return category
                else:
                    raise ClassificationError(
                        f"No categories available at level {level} and cannot create new",
                        retry_count=0,
                    )

        except Exception as e:
            logger.error(f"Failed to get/create category '{category_name}': {e}")
            raise ClassificationError(
                f"Failed to get/create category '{category_name}': {str(e)}",
                retry_count=0,
            )

    async def _store_chunk_classification(
        self,
        chunk: TextChunk,
        final_category: Category,
        source_id: Optional[str] = None,
    ):
        """
        Store chunk classification result in database.

        Args:
            chunk: Classified text chunk
            final_category: Final leaf-level category
            source_id: Optional source identifier
        """
        try:
            # Verify this is a leaf category
            if not self.category_manager.is_leaf_category(final_category):
                raise ValidationError(
                    f"Can only link chunks to leaf categories. "
                    f"Category {final_category.id} is at level {final_category.level}, "
                    f"but max depth is {self.category_manager.hierarchy_depth}",
                    field="category_level",
                    value=final_category.level,
                )

            # Create chunk record
            chunk_record = Chunk(
                content=chunk.content,
                token_count=chunk.token_count,
                category_id=final_category.id,
                source_id=source_id,
                source_metadata=chunk.metadata or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            self.db_session.add(chunk_record)
            self.db_session.flush()  # Get ID without committing

            logger.debug(
                f"Stored chunk {chunk_record.id} in category {final_category.id}"
            )

        except Exception as e:
            logger.error(f"Failed to store chunk classification: {e}")
            self.db_session.rollback()
            raise DatabaseError(
                f"Failed to store chunk classification: {str(e)}",
                operation="insert",
                table="chunks",
            )

    def _record_workflow_metrics(
        self,
        workflow_id: str,
        chunks: List[TextChunk],
        results: List[IterativeClassificationResult],
        total_latency_ms: int,
        total_llm_calls: int,
    ):
        """Record metrics for workflow execution."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        # Calculate average levels processed
        if successful_results:
            avg_levels = sum(r.levels_processed for r in successful_results) / len(
                successful_results
            )
        else:
            avg_levels = 0.0

        metrics = ClassificationWorkflowMetrics(
            workflow_id=workflow_id,
            chunks_processed=len(chunks),
            chunks_successful=len(successful_results),
            chunks_failed=len(failed_results),
            total_latency_ms=total_latency_ms,
            total_llm_calls=total_llm_calls,
            average_levels_per_chunk=avg_levels,
            timestamp=datetime.now(),
            model_name=self.model_name,
        )

        self.metrics_history.append(metrics)

        logger.info(
            f"Workflow {workflow_id}: {len(successful_results)}/{len(chunks)} successful, "
            f"{total_latency_ms}ms, {total_llm_calls} LLM calls, "
            f"{avg_levels:.1f} avg levels"
        )

    def get_workflow_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of workflow metrics."""
        if not self.metrics_history:
            return {
                "total_workflows": 0,
                "total_chunks": 0,
                "success_rate": 0.0,
                "average_latency_ms": 0.0,
                "total_llm_calls": 0,
                "average_levels_per_chunk": 0.0,
            }

        total_chunks = sum(m.chunks_processed for m in self.metrics_history)
        total_successful = sum(m.chunks_successful for m in self.metrics_history)
        total_latency = sum(m.total_latency_ms for m in self.metrics_history)
        total_llm_calls = sum(m.total_llm_calls for m in self.metrics_history)

        # Weighted average of levels per chunk
        total_levels = sum(
            m.chunks_successful * m.average_levels_per_chunk
            for m in self.metrics_history
        )
        avg_levels = total_levels / total_successful if total_successful > 0 else 0.0

        return {
            "total_workflows": len(self.metrics_history),
            "total_chunks": total_chunks,
            "total_successful": total_successful,
            "total_failed": total_chunks - total_successful,
            "success_rate": (
                total_successful / total_chunks if total_chunks > 0 else 0.0
            ),
            "average_latency_ms": total_latency / len(self.metrics_history),
            "total_llm_calls": total_llm_calls,
            "average_llm_calls_per_chunk": (
                total_llm_calls / total_chunks if total_chunks > 0 else 0.0
            ),
            "average_levels_per_chunk": avg_levels,
        }

    def clear_metrics(self):
        """Clear workflow metrics history."""
        self.metrics_history.clear()
        logger.info("Cleared workflow metrics history")


# Utility functions
def create_iterative_classifier(
    db_session: Session,
    category_manager: CategoryManager,
    model_name: str = "openai:gpt-4",
    **kwargs,
) -> IterativeClassificationWorkflow:
    """
    Create an iterative classification workflow with default configuration.

    Args:
        db_session: Database session
        category_manager: Category manager instance
        model_name: LLM model to use
        **kwargs: Additional configuration options

    Returns:
        Configured IterativeClassificationWorkflow
    """
    return IterativeClassificationWorkflow(
        db_session=db_session,
        category_manager=category_manager,
        model_name=model_name,
        **kwargs,
    )
