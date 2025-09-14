"""
Batch classification system for MemoirAI.

This module provides batch classification capabilities that allow processing
multiple text chunks in a single LLM call for improved performance and cost efficiency.
"""

import logging
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai import Agent

from ..text_processing.chunker import TextChunk
from ..database.models import Category
from ..llm.schemas import (
    BatchClassificationResponse,
    ChunkClassificationRequest,
    CategorySelection,
    LLMResponseMetadata,
    ValidationResult,
)
from ..llm.agents import create_batch_classification_agent, create_classification_agent
from ..exceptions import ClassificationError, ValidationError, LLMError


logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of classifying a single chunk."""

    chunk_id: int
    chunk: TextChunk
    category: str
    success: bool
    error: Optional[str] = None
    retry_count: int = 0
    latency_ms: Optional[int] = None


@dataclass
class BatchClassificationMetrics:
    """Metrics for batch classification operations."""

    batch_id: str
    chunks_sent: int
    chunks_successful: int
    chunks_failed: int
    chunks_retried: int
    total_latency_ms: int
    llm_calls: int
    timestamp: datetime
    model_name: str


class BatchCategoryClassifier:
    """
    Batch category classifier for processing multiple chunks in single LLM calls.

    Features:
    - Batch processing with configurable batch size
    - Structured prompts with chunk IDs and existing categories
    - Individual chunk retry logic for failures
    - Comprehensive logging and metrics
    - Validation and error handling
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4",
        batch_size: int = 5,
        max_retries: int = 3,
        hierarchy_depth: int = 3,
        max_categories_per_level: Union[int, Dict[int, int]] = 128,
        temperature: float = 0.0,
    ):
        """
        Initialize batch classifier.

        Args:
            model_name: LLM model to use for classification
            batch_size: Number of chunks to process per batch (default 5)
            max_retries: Maximum retry attempts for failed chunks
            hierarchy_depth: Maximum hierarchy depth for categories
            max_categories_per_level: Maximum categories per level (global or per-level)
            temperature: Temperature for LLM generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.hierarchy_depth = hierarchy_depth
        self.max_categories_per_level = max_categories_per_level
        self.temperature = temperature

        # Create agents
        self.batch_agent = create_batch_classification_agent(model_name)
        self.single_agent = create_classification_agent(model_name)

        # Metrics tracking
        self.metrics_history: List[BatchClassificationMetrics] = []

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """Validate classifier configuration."""
        if self.batch_size <= 0:
            raise ValidationError(
                "batch_size must be positive", field="batch_size", value=self.batch_size
            )

        if self.batch_size > 50:
            raise ValidationError(
                "batch_size cannot exceed 50 for performance reasons",
                field="batch_size",
                value=self.batch_size,
            )

        if self.max_retries < 0:
            raise ValidationError(
                "max_retries cannot be negative",
                field="max_retries",
                value=self.max_retries,
            )

        if self.hierarchy_depth < 1 or self.hierarchy_depth > 100:
            raise ValidationError(
                "hierarchy_depth must be between 1 and 100",
                field="hierarchy_depth",
                value=self.hierarchy_depth,
            )

    async def classify_chunks_batch(
        self,
        chunks: List[TextChunk],
        level: int,
        parent_category: Optional[Category],
        existing_categories: List[Category],
        contextual_helper: str,
    ) -> List[ClassificationResult]:
        """
        Classify multiple chunks in batches.

        Args:
            chunks: List of text chunks to classify
            level: Current hierarchy level (1-based)
            parent_category: Parent category (None for level 1)
            existing_categories: Existing categories at this level
            contextual_helper: Contextual information about the source

        Returns:
            List of classification results for all chunks
        """
        if not chunks:
            return []

        logger.info(
            f"Starting batch classification of {len(chunks)} chunks at level {level}"
        )

        # Split chunks into batches
        batches = self._create_batches(chunks)
        all_results = []

        for batch_idx, batch_chunks in enumerate(batches):
            batch_id = f"batch_{int(time.time())}_{batch_idx}"
            logger.debug(f"Processing {batch_id} with {len(batch_chunks)} chunks")

            try:
                batch_results = await self._process_batch(
                    batch_chunks,
                    level,
                    parent_category,
                    existing_categories,
                    contextual_helper,
                    batch_id,
                )
                all_results.extend(batch_results)

            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")
                # Create failed results for all chunks in batch
                failed_results = [
                    ClassificationResult(
                        chunk_id=chunk.chunk_id if hasattr(chunk, "chunk_id") else i,
                        chunk=chunk,
                        category="",
                        success=False,
                        error=str(e),
                    )
                    for i, chunk in enumerate(batch_chunks)
                ]
                all_results.extend(failed_results)

        logger.info(f"Completed batch classification: {len(all_results)} results")
        return all_results

    def _create_batches(self, chunks: List[TextChunk]) -> List[List[TextChunk]]:
        """Split chunks into batches of configured size."""
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batches.append(batch)

        logger.debug(f"Created {len(batches)} batches from {len(chunks)} chunks")
        return batches

    async def _process_batch(
        self,
        chunks: List[TextChunk],
        level: int,
        parent_category: Optional[Category],
        existing_categories: List[Category],
        contextual_helper: str,
        batch_id: str,
    ) -> List[ClassificationResult]:
        """Process a single batch of chunks."""
        start_time = time.time()

        try:
            # Create batch prompt
            prompt = self._create_batch_prompt(
                chunks, existing_categories, contextual_helper, level
            )

            # Call LLM
            logger.debug(f"Sending batch prompt for {batch_id}")
            response = await self.batch_agent.run_async(prompt)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Validate response
            if not self._validate_batch_response(response.data, len(chunks)):
                raise ClassificationError(
                    f"Invalid batch response format for {batch_id}", retry_count=0
                )

            # Process successful results
            results = []
            failed_chunks = []

            for i, chunk in enumerate(chunks):
                chunk_id = i + 1  # 1-based IDs as per requirements

                # Find corresponding response
                chunk_response = None
                for resp_chunk in response.data.chunks:
                    if resp_chunk.chunk_id == chunk_id:
                        chunk_response = resp_chunk
                        break

                if chunk_response and chunk_response.category:
                    # Successful classification
                    results.append(
                        ClassificationResult(
                            chunk_id=chunk_id,
                            chunk=chunk,
                            category=chunk_response.category,
                            success=True,
                            latency_ms=latency_ms,
                        )
                    )
                else:
                    # Failed classification - add to retry list
                    failed_chunks.append((chunk, chunk_id))

            # Retry failed chunks individually
            if failed_chunks:
                logger.warning(
                    f"Retrying {len(failed_chunks)} failed chunks from {batch_id}"
                )
                retry_results = await self._retry_failed_chunks(
                    failed_chunks, level, existing_categories, contextual_helper
                )
                results.extend(retry_results)

            # Record metrics
            self._record_batch_metrics(
                batch_id, chunks, results, latency_ms, 1 + len(failed_chunks)
            )

            return results

        except Exception as e:
            logger.error(f"Batch processing failed for {batch_id}: {e}")
            raise ClassificationError(
                f"Batch classification failed: {str(e)}", retry_count=0
            )

    def _create_batch_prompt(
        self,
        chunks: List[TextChunk],
        existing_categories: List[Category],
        contextual_helper: str,
        level: int,
    ) -> str:
        """
        Create structured batch prompt according to requirements.

        Format per Requirement 2A.2:
        Chunk 1:
        \"\"\"
        <text content>
        \"\"\"
        Chunk 2:
        \"\"\"
        <text content>
        \"\"\"
        """
        # Build contextual information
        context_parts = []

        if contextual_helper:
            context_parts.append(f"Document Context: {contextual_helper}")

        context_parts.append(f"Classification Level: {level}")

        if existing_categories:
            category_names = [cat.name for cat in existing_categories]
            context_parts.append(f"Existing Categories: {', '.join(category_names)}")
            context_parts.append(
                "Please select from existing categories when possible to avoid duplicates."
            )
        else:
            context_parts.append(
                "No existing categories at this level. You may create new categories."
            )

        # Add category limits information
        if isinstance(self.max_categories_per_level, dict):
            limit = self.max_categories_per_level.get(level, 128)
        else:
            limit = self.max_categories_per_level

        if existing_categories and len(existing_categories) >= limit:
            context_parts.append(
                f"Category limit ({limit}) reached. You MUST select from existing categories only."
            )

        context_section = "\n".join(context_parts)

        # Build chunks section per requirements
        chunks_section = []
        for i, chunk in enumerate(chunks, 1):
            chunks_section.append(f"Chunk {i}:")
            chunks_section.append('"""')
            chunks_section.append(chunk.content)
            chunks_section.append('"""')

        chunks_text = "\n".join(chunks_section)

        # Build complete prompt
        prompt = f"""{context_section}

Please classify each chunk into the most appropriate category. For each chunk, provide:
1. The chunk ID (1, 2, 3, etc.)
2. The category name (select existing or create new if allowed)

{chunks_text}

Respond with JSON containing the category for each chunk. Do not echo the chunk content."""

        return prompt

    def _validate_batch_response(
        self, response: BatchClassificationResponse, expected_count: int
    ) -> bool:
        """Validate batch classification response."""
        if not response or not response.chunks:
            logger.error("Empty or missing response chunks")
            return False

        if len(response.chunks) != expected_count:
            logger.error(
                f"Response chunk count mismatch: expected {expected_count}, got {len(response.chunks)}"
            )
            return False

        # Check for required fields and valid IDs
        seen_ids = set()
        for chunk_resp in response.chunks:
            if not hasattr(chunk_resp, "chunk_id") or not hasattr(
                chunk_resp, "category"
            ):
                logger.error("Response chunk missing required fields")
                return False

            if chunk_resp.chunk_id in seen_ids:
                logger.error(f"Duplicate chunk ID: {chunk_resp.chunk_id}")
                return False

            if chunk_resp.chunk_id < 1 or chunk_resp.chunk_id > expected_count:
                logger.error(f"Invalid chunk ID: {chunk_resp.chunk_id}")
                return False

            if not chunk_resp.category or not chunk_resp.category.strip():
                logger.error(f"Empty category for chunk {chunk_resp.chunk_id}")
                return False

            seen_ids.add(chunk_resp.chunk_id)

        return True

    async def _retry_failed_chunks(
        self,
        failed_chunks: List[Tuple[TextChunk, int]],
        level: int,
        existing_categories: List[Category],
        contextual_helper: str,
    ) -> List[ClassificationResult]:
        """Retry classification for failed chunks individually."""
        results = []

        for chunk, original_chunk_id in failed_chunks:
            retry_count = 0
            success = False

            while retry_count < self.max_retries and not success:
                try:
                    # Create single chunk prompt
                    prompt = self._create_single_chunk_prompt(
                        chunk, existing_categories, contextual_helper, level
                    )

                    start_time = time.time()
                    response = await self.single_agent.run_async(prompt)
                    latency_ms = int((time.time() - start_time) * 1000)

                    if response.data and response.data.category:
                        results.append(
                            ClassificationResult(
                                chunk_id=original_chunk_id,
                                chunk=chunk,
                                category=response.data.category,
                                success=True,
                                retry_count=retry_count + 1,
                                latency_ms=latency_ms,
                            )
                        )
                        success = True
                        logger.debug(f"Retry successful for chunk {original_chunk_id}")
                    else:
                        retry_count += 1
                        logger.warning(
                            f"Retry {retry_count} failed for chunk {original_chunk_id}"
                        )

                except Exception as e:
                    retry_count += 1
                    logger.error(
                        f"Retry {retry_count} error for chunk {original_chunk_id}: {e}"
                    )

            if not success:
                # All retries failed
                results.append(
                    ClassificationResult(
                        chunk_id=original_chunk_id,
                        chunk=chunk,
                        category="",
                        success=False,
                        error=f"Failed after {self.max_retries} retries",
                        retry_count=retry_count,
                    )
                )
                logger.error(f"All retries failed for chunk {original_chunk_id}")

        return results

    def _create_single_chunk_prompt(
        self,
        chunk: TextChunk,
        existing_categories: List[Category],
        contextual_helper: str,
        level: int,
    ) -> str:
        """Create prompt for single chunk classification (retry)."""
        context_parts = []

        if contextual_helper:
            context_parts.append(f"Document Context: {contextual_helper}")

        context_parts.append(f"Classification Level: {level}")

        if existing_categories:
            category_names = [cat.name for cat in existing_categories]
            context_parts.append(f"Existing Categories: {', '.join(category_names)}")
            context_parts.append(
                "Please select from existing categories when possible to avoid duplicates."
            )

        context_section = "\n".join(context_parts)

        prompt = f"""{context_section}

Please classify the following text chunk into the most appropriate category:

Text: {chunk.content}

Provide the category name only."""

        return prompt

    def _record_batch_metrics(
        self,
        batch_id: str,
        chunks: List[TextChunk],
        results: List[ClassificationResult],
        latency_ms: int,
        llm_calls: int,
    ):
        """Record metrics for batch processing."""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        retried = sum(1 for r in results if r.retry_count > 0)

        metrics = BatchClassificationMetrics(
            batch_id=batch_id,
            chunks_sent=len(chunks),
            chunks_successful=successful,
            chunks_failed=failed,
            chunks_retried=retried,
            total_latency_ms=latency_ms,
            llm_calls=llm_calls,
            timestamp=datetime.now(),
            model_name=self.model_name,
        )

        self.metrics_history.append(metrics)

        logger.info(
            f"Batch {batch_id}: {successful}/{len(chunks)} successful, "
            f"{retried} retried, {latency_ms}ms, {llm_calls} LLM calls"
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of classification metrics."""
        if not self.metrics_history:
            return {
                "total_batches": 0,
                "total_chunks": 0,
                "success_rate": 0.0,
                "average_latency_ms": 0.0,
                "total_llm_calls": 0,
            }

        total_chunks = sum(m.chunks_sent for m in self.metrics_history)
        total_successful = sum(m.chunks_successful for m in self.metrics_history)
        total_latency = sum(m.total_latency_ms for m in self.metrics_history)
        total_llm_calls = sum(m.llm_calls for m in self.metrics_history)

        return {
            "total_batches": len(self.metrics_history),
            "total_chunks": total_chunks,
            "total_successful": total_successful,
            "total_failed": total_chunks - total_successful,
            "success_rate": (
                total_successful / total_chunks if total_chunks > 0 else 0.0
            ),
            "average_latency_ms": total_latency / len(self.metrics_history),
            "total_llm_calls": total_llm_calls,
            "average_chunks_per_batch": total_chunks / len(self.metrics_history),
        }

    def clear_metrics(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Cleared classification metrics history")


# Utility functions for batch processing
def create_batch_classifier(
    model_name: str = "openai:gpt-4", batch_size: int = 5, **kwargs
) -> BatchCategoryClassifier:
    """
    Create a batch category classifier with default configuration.

    Args:
        model_name: LLM model to use
        batch_size: Batch size for processing
        **kwargs: Additional configuration options

    Returns:
        Configured BatchCategoryClassifier
    """
    return BatchCategoryClassifier(
        model_name=model_name, batch_size=batch_size, **kwargs
    )


def validate_batch_size(batch_size: int) -> bool:
    """
    Validate batch size parameter.

    Args:
        batch_size: Batch size to validate

    Returns:
        True if valid, False otherwise
    """
    return 1 <= batch_size <= 50
