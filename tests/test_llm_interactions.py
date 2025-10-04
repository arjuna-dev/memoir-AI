"""Tests for LLM interaction helpers."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from memoir_ai.exceptions import LLMError
from memoir_ai.llm.interactions import (
    build_chunk_classification_prompt,
    build_query_category_prompt,
    build_query_contextual_helper_prompt,
    classify_chunk_with_llm,
    select_categories_for_query,
    select_category_for_query,
    select_contextual_helper_for_query,
)
from memoir_ai.llm.schemas import (
    CategorySelection,
    QueryCategorySelection,
    QueryCategorySelectionList,
)


class DummyCategory:
    """Simple helper class providing a name attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_build_chunk_classification_prompt_with_categories() -> None:
    """Ensure prompts include expected context when categories exist."""

    prompt = build_chunk_classification_prompt(
        chunk_content="AI research advances",
        level=2,
        existing_categories=[DummyCategory("Technology"), DummyCategory("Science")],
        contextual_helper="Academic paper",
        can_create_new=True,
        category_limit=10,
        parent_categories=[DummyCategory("Research")],
    )

    assert "Document Context (root category): Academic paper" in prompt
    assert "remain consistent with the document context" in prompt
    assert "Classification Level: 2" in prompt
    assert "Parent Category Path: Research" in prompt
    assert "more specific" in prompt
    assert "Existing Categories: Technology, Science" in prompt
    assert "AI research advances" in prompt


def test_build_chunk_classification_prompt_without_categories() -> None:
    """When no categories are provided the prompt should allow creation."""

    prompt = build_chunk_classification_prompt(
        chunk_content="General news",
        level=1,
        existing_categories=[],
        contextual_helper=None,
        can_create_new=True,
    )

    assert "No existing categories at this level" in prompt
    assert "General news" in prompt


def test_build_query_category_prompt() -> None:
    """Query prompt should list options and instructions."""

    prompt = build_query_category_prompt(
        query_text="Explain the latest in quantum computing",
        level=2,
        available_categories=["Technology", "Science"],
        contextual_helper="Looking for research topics",
    )

    assert "Available Categories: Technology, Science" in prompt
    assert "User Query: Explain the latest in quantum computing" in prompt
    assert "ranked relevance score" in prompt


def test_build_query_contextual_helper_prompt() -> None:
    """Contextual helper prompt should list all helper options."""

    helpers = [
        DummyCategory("Tech Insights"),
        DummyCategory("Policy Overview"),
    ]
    # Attach metadata for first helper
    helpers[0].metadata_json = {"helper_text": "Technical deep dive"}

    prompt = build_query_contextual_helper_prompt(
        query_text="Explain the policy response to AI advancements",
        contextual_helpers=helpers,
    )

    assert "Option 1" in prompt
    assert "Tech Insights" in prompt
    assert "Technical deep dive" in prompt
    assert "Option 2" in prompt
    assert "User Query: Explain the policy response to AI advancements" in prompt


@pytest.mark.asyncio
async def test_classify_chunk_with_llm_success() -> None:
    """Successful classification should return the model output and metadata."""

    agent = Mock()
    agent.run_async = AsyncMock(
        return_value=SimpleNamespace(
            data=CategorySelection(category="Technology", ranked_relevance=5)
        )
    )

    selection, metadata = await classify_chunk_with_llm(
        chunk_content="AI breakthroughs",
        level=1,
        existing_categories=["Technology"],
        contextual_helper="Tech blog",
        can_create_new=False,
        category_limit=5,
        agent=agent,
        chunk_identifier="chunk-1",
        parent_categories=[DummyCategory("Computing")],
    )

    agent.run_async.assert_awaited()
    assert selection.category == "Technology"
    assert metadata["chunk_identifier"] == "chunk-1"
    assert "latency_ms" in metadata
    assert metadata["parent_category_path"] == ["Computing"]


@pytest.mark.asyncio
async def test_classify_chunk_with_llm_invalid_response() -> None:
    """Non CategorySelection responses should raise an error."""

    agent = Mock()
    agent.run_async = AsyncMock(return_value=SimpleNamespace(data=None))

    with pytest.raises(LLMError):
        await classify_chunk_with_llm(
            chunk_content="AI breakthroughs",
            level=1,
            existing_categories=["Technology"],
            contextual_helper="Tech blog",
            can_create_new=False,
            agent=agent,
        )


@pytest.mark.asyncio
async def test_select_category_for_query_success() -> None:
    """Successful query selection should return schema output."""

    agent = Mock()
    agent.run_async = AsyncMock(
        return_value=SimpleNamespace(
            data=QueryCategorySelection(category="AI", ranked_relevance=7)
        )
    )

    selection, metadata = await select_category_for_query(
        query_text="deep learning",
        level=2,
        available_categories=[DummyCategory("AI"), DummyCategory("ML")],
        contextual_helper="Research",
        agent=agent,
    )

    agent.run_async.assert_awaited()
    assert selection.category == "AI"
    assert metadata["available_category_names"] == ["AI", "ML"]
    assert metadata["level"] == 2
    assert metadata["selection_count"] == 1


@pytest.mark.asyncio
async def test_select_category_for_query_invalid_response() -> None:
    """Unexpected responses should trigger an LLMError."""

    agent = Mock()
    agent.run_async = AsyncMock(return_value=SimpleNamespace(data="invalid"))

    with pytest.raises(LLMError):
        await select_category_for_query(
            query_text="deep learning",
            level=2,
            available_categories=["AI"],
            contextual_helper=None,
            agent=agent,
        )


@pytest.mark.asyncio
async def test_select_categories_for_query_multi_success() -> None:
    """Multi-selection should return ordered results and metadata."""

    agent = Mock()
    agent.run_async = AsyncMock(
        return_value=SimpleNamespace(
            data=QueryCategorySelectionList(
                selections=[
                    QueryCategorySelection(category="AI", ranked_relevance=9),
                    QueryCategorySelection(category="ML", ranked_relevance=8),
                    QueryCategorySelection(category="AI", ranked_relevance=7),
                ]
            )
        )
    )

    selections, metadata = await select_categories_for_query(
        query_text="deep learning",
        level=2,
        available_categories=[
            DummyCategory("AI"),
            DummyCategory("ML"),
            DummyCategory("Web"),
        ],
        contextual_helper="Research",
        selection_count=2,
        agent=agent,
    )

    agent.run_async.assert_awaited()
    assert [selection.category for selection in selections] == ["AI", "ML"]
    assert metadata["selection_count"] == 2
    assert metadata["available_category_names"] == ["AI", "ML", "Web"]


@pytest.mark.asyncio
async def test_select_categories_for_query_invalid_response() -> None:
    """Invalid multi-selection responses should raise errors."""

    agent = Mock()
    agent.run_async = AsyncMock(return_value=SimpleNamespace(data=None))

    with pytest.raises(LLMError):
        await select_categories_for_query(
            query_text="deep learning",
            level=2,
            available_categories=[DummyCategory("AI")],
            contextual_helper="Research",
            selection_count=2,
            agent=agent,
        )


@pytest.mark.asyncio
async def test_select_contextual_helper_for_query_success() -> None:
    """Contextual helper selection should return chosen helper and metadata."""

    helpers = [DummyCategory("Tech"), DummyCategory("Policy")]
    helpers[0].metadata_json = {"helper_text": "Technical details"}
    helpers[1].metadata_json = {"helper_text": "Policy analysis"}

    agent = Mock()
    agent.run_async = AsyncMock(
        return_value=SimpleNamespace(
            data=QueryCategorySelection(category="Policy", ranked_relevance=8)
        )
    )

    selection, metadata = await select_contextual_helper_for_query(
        query_text="How are governments regulating AI?",
        contextual_helpers=helpers,
        agent=agent,
    )

    agent.run_async.assert_awaited()
    assert selection.category == "Policy"
    assert metadata["available_contextual_helpers"] == ["Tech", "Policy"]


@pytest.mark.asyncio
async def test_select_contextual_helper_for_query_invalid_response() -> None:
    """Invalid contextual helper responses should raise an error."""

    helpers = [DummyCategory("Tech")]
    agent = Mock()
    agent.run_async = AsyncMock(return_value=SimpleNamespace(data=None))

    with pytest.raises(LLMError):
        await select_contextual_helper_for_query(
            query_text="Explain AI infrastructure",
            contextual_helpers=helpers,
            agent=agent,
        )
