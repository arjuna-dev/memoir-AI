"""Utility helpers for building prompts and executing LLM interactions."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pydantic_ai import Agent

from ..exceptions import LLMError
from .agents import create_classification_agent, create_query_classification_agent
from .schemas import CategorySelection, QueryCategorySelection


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
        context_parts.append(f"Document Context: {contextual_helper}")

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
) -> str:
    """Create the prompt used when selecting categories for a user query."""

    context_parts: list[str] = [f"Classification Level: {level}"]

    if contextual_helper:
        context_parts.append(f"Additional Context: {contextual_helper}")

    category_names = _extract_category_names(available_categories)
    if category_names:
        context_parts.append("Available Categories: " + ", ".join(category_names))
    else:
        context_parts.append(
            "No explicit categories provided. You must still return a best guess."
        )

    context_parts.append(
        "Select the single most relevant category for the user query and provide a "
        "ranked relevance score from 1 (least relevant) to 10 (most relevant)."
    )

    context_section = "\n".join(context_parts)

    prompt = f"""{context_section}

User Query: {query_text}

Return the category name and relevance score only."""

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
    }

    return selection, metadata
