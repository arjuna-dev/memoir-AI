"""Utility helpers for building prompts and executing LLM interactions."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic_ai import Agent

from ..exceptions import LLMError
from .agents import (
    create_classification_agent,
    create_hierarchical_batch_classification_agent,
    create_query_classification_agent,
    create_query_multi_selection_agent,
)
from .schemas import (
    CategorySelection,
    HierarchicalBatchClassificationResponse,
    QueryCategorySelection,
    QueryCategorySelectionList,
)


def _extract_category_names(categories: Sequence[Any]) -> list[str]:
    """Normalize a sequence of categories into plain names."""

    names: list[str] = []
    for category in categories:
        if isinstance(category, str):
            if category.strip():
                names.append(category.strip())
            continue

        name = getattr(category, "name", None)
        if isinstance(name, str) and name.strip():
            names.append(name.strip())

    return names


def build_chunk_classification_prompt(
    *,
    chunk_content: str,
    level: int,
    existing_categories: Sequence[Any],
    contextual_helper: str,
    can_create_new: bool,
    category_limit: Optional[int] = None,
    parent_categories: Optional[Sequence[Any]] = None,
) -> str:
    """Construct the prompt used for single chunk classification."""

    context_parts: list[str] = []

    if contextual_helper:
        context_parts.append(f"Document Context (root category): {contextual_helper}")
        if level == 1:
            context_parts.append(
                "Level 1 categories must be direct subcategories of this document context."
            )
        else:
            context_parts.append(
                "All categories must remain consistent with the document context and parent selections."
            )

    context_parts.append(f"Classification Level: {level}")

    parent_names: list[str] = []
    if level > 1 and parent_categories:
        parent_names = _extract_category_names(parent_categories)
        if parent_names:
            context_parts.append("Parent Category Path: " + " > ".join(parent_names))
            context_parts.append(
                "Select a sub-category that is more specific than the parent path."
            )
            context_parts.append(
                "Avoid repeating the exact parent category name unless it is the only suitable option."
            )

    category_names = _extract_category_names(existing_categories)
    if category_names:
        categories_text = ", ".join(category_names)
        context_parts.append(f"Existing Categories: {categories_text}")
        if can_create_new:
            context_parts.append(
                "Please select from existing categories when possible to avoid duplicates, "
                "or create a new category if none fit well."
            )
        else:
            if category_limit is not None:
                context_parts.append(
                    f"Category limit ({category_limit}) reached. You MUST select from existing categories only."
                )
            else:
                context_parts.append("You MUST select from existing categories only.")
    else:
        context_parts.append(
            "No existing categories at this level. You may create a new category."
        )

    context_section = "\n".join(context_parts)

    prompt = f"""{context_section}

Please classify the following text into the most appropriate category for level {level}:

Text: {chunk_content}

Provide only the category name."""

    return prompt


def build_query_category_prompt(
    *,
    query_text: str,
    level: int,
    available_categories: Sequence[Any],
    contextual_helper: str,
    selection_count: int = 1,
) -> str:
    """Create the prompt used when selecting categories for a user query.

    Raises:
        LLMError: If no categories are available at the specified level
    """

    context_parts: list[str] = [f"Classification Level: {level}"]

    category_names = _extract_category_names(available_categories)

    if level == 1:
        raise LLMError(
            f"Level should be at least 2 and not {level} since level 1 is for contextual helpers only."
        )

    context_parts.append(f"Additional Context: {contextual_helper}")
    context_parts.append("Available Categories: " + ", ".join(category_names))
    if selection_count == 1:
        context_parts.append(
            "Select the single most relevant category for the user query and provide a "
            "ranked relevance score from 1 (least relevant) to 10 (most relevant)."
        )
    else:
        context_parts.append(
            f"Select the top {selection_count} categories for the user query. Return them from most to least relevant and provide a "
            "ranked relevance score from 1 (least relevant) to 10 (most relevant) for each."
        )
        context_parts.append(
            "Do not create new categories; every selection must come from the provided list. Return it matching the format exactly, including punctuation and capitalization."
        )

    context_section = "\n".join(context_parts)

    prompt = f"""{context_section}

User Query: {query_text}

Return the category name(s) and relevance score(s) only."""

    return prompt


def build_query_contextual_helper_prompt(
    *,
    query_text: str,
    contextual_helpers: Sequence[Any],
) -> str:
    """Create the prompt used when selecting a contextual helper at level 1."""

    helper_entries: List[str] = []
    for idx, helper in enumerate(contextual_helpers, start=1):
        name = getattr(helper, "name", None)
        metadata = getattr(helper, "metadata_json", {}) or {}
        helper_text = metadata.get("helper_text")

        parts = [f"Option {idx}:"]
        if name:
            parts.append(f"Name: {name}")
        if helper_text:
            parts.append(f"Context: {helper_text}")
        helper_entries.append(" ".join(parts))

    if not helper_entries:
        raise LLMError("No contextual helpers available for selection at level 1.")

    options_section = "\n".join(helper_entries)

    prompt = f"""You are selecting the most relevant contextual helper (root category) for a user query.

Available Contextual Helpers:
{options_section}

User Query: {query_text}

Return the helper name and a relevance score from 1 (least relevant) to 10 (most relevant)."""

    return prompt


async def classify_chunk_with_llm(
    *,
    chunk_content: str,
    level: int,
    existing_categories: Sequence[Any],
    contextual_helper: str,
    can_create_new: bool,
    category_limit: Optional[int] = None,
    agent: Optional[Agent] = None,
    model_name: Optional[str] = None,
    chunk_identifier: Optional[str] = None,
    parent_categories: Optional[Sequence[Any]] = None,
) -> Tuple[CategorySelection, Dict[str, Any]]:
    """Execute a single chunk classification call using Pydantic AI."""

    prompt = build_chunk_classification_prompt(
        chunk_content=chunk_content,
        level=level,
        existing_categories=existing_categories,
        contextual_helper=contextual_helper,
        can_create_new=can_create_new,
        category_limit=category_limit,
        parent_categories=parent_categories,
    )

    logger.info(f"LLM Prompt for level {level}: {prompt}")

    classification_agent = agent or create_classification_agent(model_name)

    start_time = time.perf_counter()
    try:
        response = await classification_agent.run_async(prompt)
        logger.info(f"LLM Response for level {level}: {response}")
    except Exception as exc:  # pragma: no cover - network/runtime specific
        raise LLMError(
            f"interactions.py | Failed to classify chunk at level {level}: {exc}"
        ) from exc

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "output", None)
    if not isinstance(data, CategorySelection) or not data.category.strip():
        raise LLMError(f"Empty category response at level {level}")

    metadata = {
        "prompt": prompt,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC),
        "model_name": model_name or getattr(classification_agent, "model", None),
        "chunk_identifier": chunk_identifier,
        "level": level,
        "can_create_new": can_create_new,
        "existing_category_names": _extract_category_names(existing_categories),
        "parent_category_path": (
            _extract_category_names(parent_categories) if parent_categories else None
        ),
    }

    return data, metadata


async def select_category_for_query(
    *,
    query_text: str,
    level: int,
    available_categories: Sequence[Any],
    contextual_helper: str,
    agent: Optional[Agent] = None,
    model_name: Optional[str] = None,
) -> Tuple[QueryCategorySelection, Dict[str, Any]]:
    """Ask an LLM to select the best category for a natural language query."""

    prompt = build_query_category_prompt(
        query_text=query_text,
        level=level,
        available_categories=available_categories,
        contextual_helper=contextual_helper,
        selection_count=1,
    )

    query_agent = agent or create_query_classification_agent(model_name)

    start_time = time.perf_counter()
    try:
        response = await query_agent.run_async(prompt)
    except Exception as exc:  # pragma: no cover - network/runtime specific
        raise LLMError(
            f"Failed to select category for query at level {level}: {exc}"
        ) from exc

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "output", None)

    if isinstance(data, QueryCategorySelection):
        if not data.category.strip():
            raise LLMError(f"Empty category response at level {level}")
        selection = data
    elif isinstance(data, CategorySelection):
        if not data.category.strip():
            raise LLMError(f"Empty category response at level {level}")
        selection = QueryCategorySelection(
            category=data.category, ranked_relevance=data.ranked_relevance
        )
    else:
        raise LLMError(
            "Query classification agent returned an unexpected response type."
        )

    metadata = {
        "prompt": prompt,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC),
        "model_name": model_name or getattr(query_agent, "model", None),
        "level": level,
        "available_category_names": _extract_category_names(available_categories),
        "selection_count": 1,
    }

    return selection, metadata


async def select_categories_for_query(
    *,
    query_text: str,
    level: int,
    available_categories: Sequence[Any],
    contextual_helper: str,
    selection_count: int,
    agent: Optional[Agent] = None,
    model_name: Optional[str] = None,
) -> Tuple[List[QueryCategorySelection], Dict[str, Any]]:
    """Ask an LLM to select multiple categories for a natural language query."""

    if selection_count <= 0:
        raise ValueError("selection_count must be positive")

    available_names = _extract_category_names(available_categories)
    if not available_names:
        if level == 1:
            raise LLMError(
                "No contextual helpers (Level 1 categories) available for query selection. "
                "This indicates that no documents have been ingested or the database is empty."
            )
        raise LLMError(
            f"No categories available at level {level} for query selection. "
            f"This indicates a database inconsistency or incomplete category hierarchy."
        )

    effective_count = min(selection_count, len(available_names))

    prompt = build_query_category_prompt(
        query_text=query_text,
        level=level,
        available_categories=available_categories,
        contextual_helper=contextual_helper,
        selection_count=effective_count,
    )

    query_agent = agent or create_query_multi_selection_agent(model_name)

    start_time = time.perf_counter()
    try:
        response = await query_agent.run_async(prompt)
    except Exception as exc:  # pragma: no cover - network/runtime specific
        raise LLMError(
            f"Failed to select categories for query at level {level}: {exc}"
        ) from exc

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "output", None)

    if isinstance(data, QueryCategorySelectionList):
        selections = data.selections
    elif isinstance(data, QueryCategorySelection):
        selections = [data]
    else:
        raise LLMError(
            "Query multi-selection agent returned an unexpected response type."
        )

    if not selections:
        raise LLMError("Query multi-selection agent returned no selections.")

    normalized: List[QueryCategorySelection] = []
    seen: set[str] = set()
    for selection in selections:
        name = selection.category.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(selection)
        if len(normalized) >= effective_count:
            break

    if not normalized:
        raise LLMError("Query multi-selection agent did not return valid categories.")

    metadata = {
        "prompt": prompt,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC),
        "model_name": model_name or getattr(query_agent, "model", None),
        "level": level,
        "available_category_names": available_names,
        "selection_count": len(normalized),
    }

    return normalized, metadata


async def select_contextual_helper_for_query(
    *,
    query_text: str,
    contextual_helpers: Sequence[Any],
    agent: Optional[Agent] = None,
    model_name: Optional[str] = None,
) -> Tuple[QueryCategorySelection, Dict[str, Any]]:
    """Ask an LLM to select the most relevant contextual helper (level 1)."""

    prompt = build_query_contextual_helper_prompt(
        query_text=query_text, contextual_helpers=contextual_helpers
    )

    helper_agent = agent or create_query_classification_agent(model_name)

    start_time = time.perf_counter()
    try:
        response = await helper_agent.run_async(prompt)
    except Exception as exc:  # pragma: no cover - network/runtime specific
        raise LLMError(f"Failed to select contextual helper for query: {exc}") from exc

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "output", None)

    if not isinstance(data, QueryCategorySelection):
        raise LLMError("Contextual helper selection agent returned invalid response")

    if not data.category.strip():
        raise LLMError("Contextual helper selection returned empty category")

    metadata = {
        "prompt": prompt,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC),
        "model_name": model_name or getattr(helper_agent, "model", None),
        "available_contextual_helpers": [
            getattr(helper, "name", None) or "" for helper in contextual_helpers
        ],
    }

    return data, metadata


async def classify_all_chunks_with_llm(
    *,
    chunks: list[Any],
    contextual_helper: str,
    hierarchy_depth: int = 3,
    agent: Optional[Agent] = None,
    model_name: Optional[str] = None,
) -> Tuple[HierarchicalBatchClassificationResponse, Dict[str, Any]]:
    """Execute a hierarchical batch classification call for all chunks using Pydantic AI."""

    from ..classification.batch_classifier import BatchCategoryClassifier
    from .llm_models import Models

    # Create a temporary batch classifier to use its prompt generation method
    temp_classifier = BatchCategoryClassifier(
        model=Models.test, hierarchy_depth=hierarchy_depth
    )

    # Generate the prompt using the existing method
    prompt = temp_classifier._create_batch_prompt_all_levels(
        chunks=chunks, contextual_helper=contextual_helper
    )

    classification_agent = agent or create_hierarchical_batch_classification_agent(
        model_name
    )

    start_time = time.perf_counter()
    try:
        response = await classification_agent.run_async(prompt)
        logger.info(f"LLM Hierarchical Batch Classification Response: {response}")
    except Exception as exc:  # pragma: no cover - network/runtime specific
        raise LLMError(
            f"interactions.py | Failed to classify all chunks with hierarchical batch classification: {exc}"
        ) from exc

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    data = getattr(response, "data", None)
    if data is None:
        data = getattr(response, "output", None)

    if not isinstance(data, HierarchicalBatchClassificationResponse):
        raise LLMError("Invalid hierarchical batch classification response format")

    if not data.classifications:
        raise LLMError("Empty classifications response")

    metadata = {
        "prompt": prompt,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(UTC),
        "model_name": model_name or getattr(classification_agent, "model", None),
        "chunks_count": len(chunks),
        "hierarchy_depth": hierarchy_depth,
        "contextual_helper": contextual_helper,
    }

    return data, metadata
