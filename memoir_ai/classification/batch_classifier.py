"""
Batch classification system for MemoirAI.

This module provides batch classification capabilities that allow processing
multiple text chunks in a single LLM call for improved performance and cost efficiency.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm import token_counter
from pydantic_ai import Agent

from ..database.models import Category
from ..exceptions import ClassificationError, LLMError, ValidationError
from ..llm.agents import create_batch_classification_agent, create_classification_agent
from ..llm.llm_models import Model, Models
from ..llm.schemas import (
    BatchClassificationResponse,
    CategorySelection,
    ChunkClassificationRequest,
    LLMResponseMetadata,
    ValidationResult,
)
from ..text_processing.chunker import TextChunk

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
        model: Model = Models.openai_gpt_4_1_mini,
        batch_size: int = 5,
        max_retries: int = 3,
        hierarchy_depth: int = 3,
        max_categories_per_level: Union[int, Dict[int, int]] = 128,
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize batch classifier.

        Args:
            model: LLM model to use for classification
            batch_size: Number of chunks to process per batch (default 5)
            max_retries: Maximum retry attempts for failed chunks
            hierarchy_depth: Maximum hierarchy depth for categories
            max_categories_per_level: Maximum categories per level (global or per-level)
            temperature: Temperature for LLM generation
        """
        self.model_name = model.name
        self.context_length = model.context_length
        self.litellm_name = model.litellm_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.hierarchy_depth = hierarchy_depth
        self.max_categories_per_level = max_categories_per_level
        self.temperature = temperature

        # Create agents
        self.batch_agent = create_batch_classification_agent(model.name)
        self.single_agent = create_classification_agent(model.name)

        # Metrics tracking
        self.metrics_history: List[BatchClassificationMetrics] = []

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate classifier configuration."""
        if self.batch_size <= 0:
            raise ValidationError(
                "batch_size must be positive", field="batch_size", value=self.batch_size
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

    def _validate_chunk_token_count(self, chunks: List[TextChunk]) -> None:
        """
        Validate that the combined token count of all chunks does not exceed half of the context window.
        """

        max_tokens = self.context_length // 2
        total_tokens = sum(chunk.token_count for chunk in chunks)
        if total_tokens > max_tokens:
            raise ValidationError(
                f"Total token count ({total_tokens}) exceeds half of the context window ({max_tokens})",
                field="chunks",
                value=total_tokens,
            )

    def _validate_final_prompt_length(self, prompt: str) -> None:
        """
        Validate that the final prompt does not exceed half of the context window.
        """

        max_tokens = self.context_length // 2
        messages = [{"user": "role", "content": prompt}]
        token_count = token_counter(model=self.litellm_name, messages=messages)
        if token_count > max_tokens:
            raise ValidationError(
                f"Final prompt token count ({token_count}) exceeds half of the context window ({max_tokens})",
                field="prompt",
                value=token_count,
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

        # Consolidate chunks into batches
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
        """Consolidate chunks into batches of configured size."""
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
            prompt = self._create_batch_prompt_all_levels(chunks, contextual_helper)

            # Call LLM
            logger.debug(f"Sending batch prompt for {batch_id}")
            response = await self.batch_agent.run_async(prompt)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Validate response structure (allowing blank categories for retries)
            if not self._is_structurally_valid_response(response.data, len(chunks)):
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

    def _create_batch_prompt_level_by_level(
        self,
        chunks: List[TextChunk],
        existing_categories: List[Category],
        contextual_helper: str,
        level: int,
    ) -> str:
        """
        Create structured batch prompt according to requirements.
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

    def _create_batch_prompt_all_levels(
        self,
        chunks: List[TextChunk],
        contextual_helper: str,
    ) -> str:
        """
        Create structured batch prompt
        """
        # Validate chunk token count
        self._validate_chunk_token_count(chunks)

        context_section = f"Document Context: {contextual_helper}"

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

Your job is to classify each of the {len(chunks)} chunks of the text into {self.hierarchy_depth} nested categories. Take in account that an LLM will use those categories to try to retrieve relevant texts for a human query so the category names should be descriptive enough that they will point to the relevant content.

Since we know the context of the text already (title, author, date, etc.) the nested categories should be under it, drilling down into more specifics.

For each chunk, provide:
1. The chunk ID (1, 2, 3, etc.)
2. The nested categories with name and child category if any

Example input context:
\"\"\"
title="UN Delegates Walk Out on Netanyahu Speech",
author="Jane Doe",
date="2025-10-09",
topic="News Article About Palestine-Israel Conflict",
source_type="news_article",
description= "UN delegates lead a mass walkout as Netanyahu insists Israel must 'finish the job' in Gaza. Delegates stormed out during Benjamin Netanyahu's speech as he lambasted nations for 'caving' to Hamas"
\"\"\"

Example input text chunks:
---
Chunk 1:
\"\"\"

UN delegates lead a mass walkout as Netanyahu insists Israel must 'finish the job' in Gaza

Delegates stormed out during Benjamin Netanyahu's speech as he lambasted nations for 'caving' to Hamas

Delegates staged a mass walkout as Israeli prime minister Benjamin Netanyahu gave his UN address, shortly before he railed against nations that have “waged a political and legal war” against his country.

There were boos as he decried the growing global recognition of a Palestinian state and vowed Israel would continue to fight in Gaza and “finish the job” of eliminating Hamas. The remarks fly in the face of international pressure on Netanyahu to end the war.
\"\"\"

Chunk 2:
\"\"\"
Addressing rows of empty seats at the UN General Assembly on Friday, he firmly rejected giving the Palestinians a state, and told world leaders who have recognised such a state this week that “we will not allow you to shove a terror state down our throats”.

Dozens of delegates walked out of the chamber as Netanyahu, who is wanted for war crimes by the International Criminal Court, took to the podium.

He accused world leaders of buckling “when the going got tough” for Israel.
\"\"\"

Chunk 3:
\"\"\"
“When the going got tough, you caved,” he said to the mostly empty chamber.

“And here is the shameless result of that collapse. For much of the past two years, Israel has had to fight a seven-front war against barbarism with many of your nations opposing us.

Benjamin Netanyahu told UN General Assembly delegates they 'appease evil'
open image in gallery
Benjamin Netanyahu told UN General Assembly delegates they 'appease evil' (Reuters)
“Astoundingly, as we fight the terrorists who murdered many of your citizens, you are fighting us. You condemn us, you embargo us, and you wage political and legal warfare – it's called lawfare – against us.

“I say to the representatives of those nations, this is not an indictment of Israel; it's an indictment of you. It's an indictment of weak leaders who appease evil rather than support a nation whose brave soldiers guard you from the barbarians at the gate.”
\"\"\"
---

Example response format for 3 nested categories (for real use cases expect larger chunks):
---
"classifications": [
    {{
      "chunk_id": 1,
      "category_tree": {{
        "name": "UNGA Walkout & Audience Reaction (Boos, Empty Seats)",
        "child": {{
          "name": "Speech Theme: 'Finish the Job' Pledge",
          "child": {{
            "name": "Opposition to Palestinian State Recognition (Within Gaza War Context)"
          }}
        }}
      }}
    }},
    {{
      "chunk_id": 2,
      "category_tree": {{
        "name": "Policy Stance at UNGA",
        "child": {{
          "name": "Rejection of Palestinian Statehood",
          "child": {{
            "name": "International Law & Accountability (ICC Warrant Reference)"
          }}
        }}
      }}
    }},
    {{
      "chunk_id": 3,
      "category_tree": {{
        "name": "Political Communication & Rhetoric",
        "child": {{
          "name": "Accusations of 'Lawfare' and International Appeasement",
          "child": {{
            "name": "Security Narrative: 'Seven-Front War' & Condemnation of Embargoes"
          }}
        }}
      }}
    }}
  ]
---

Text Chunks to Classify:

{chunks_text}

"""

        # Validate final prompt length
        self._validate_final_prompt_length(prompt)

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

    def _is_structurally_valid_response(
        self, response: BatchClassificationResponse, expected_count: int
    ) -> bool:
        """Validate response structure while allowing empty categories."""

        if not response or not response.chunks:
            return False

        if len(response.chunks) != expected_count:
            return False

        seen_ids = set()
        for chunk_resp in response.chunks:
            if not hasattr(chunk_resp, "chunk_id") or not hasattr(
                chunk_resp, "category"
            ):
                return False

            if chunk_resp.chunk_id in seen_ids:
                return False

            if chunk_resp.chunk_id < 1 or chunk_resp.chunk_id > expected_count:
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
    ) -> None:
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

    def clear_metrics(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Cleared classification metrics history")


# Utility functions for batch processing
def create_batch_classifier(
    model: Model = Models.openai_gpt_4_1_mini, batch_size: int = 5, **kwargs: Any
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
    return BatchCategoryClassifier(model=model, batch_size=batch_size, **kwargs)


def validate_batch_size(batch_size: int) -> bool:
    """
    Validate batch size parameter.

    Args:
        batch_size: Batch size to validate

    Returns:
        True if valid, False otherwise
    """
    return 1 <= batch_size <= 50
