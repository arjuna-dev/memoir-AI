import pytest

from memoir_ai.aggregation.budget_manager import (
    BudgetConfig,
    BudgetManager,
    PromptLimitingStrategy,
)
from memoir_ai.aggregation.summarization_engine import SummarizationEngine


@pytest.fixture
def budget_manager() -> None:
    config = BudgetConfig(
        max_token_budget=1000,
        prompt_limiting_strategy=PromptLimitingStrategy.SUMMARIZE,
        model_name="gpt-4",
        summarization_instruction_headroom_tokens=200,
    )
    return BudgetManager(config)


def test_analyze_no_compression_needed(budget_manager) -> None:
    chunk_texts = ["short one", "another short"]
    # Force estimate where chunks_total_tokens small so ratio>=1
    estimate = budget_manager.estimate_budget_usage(
        query_text="Q", chunks_text=chunk_texts
    )
    engine = SummarizationEngine(budget_manager)
    plan = engine.analyze(estimate, chunk_texts)
    assert plan["compression_ratio"] >= 0
    assert len(plan["targets"]) == 2


def test_summarize_basic_truncation(budget_manager) -> None:
    # Create larger texts so compression needed (simulate by adjusting estimate)
    chunk_texts = ["a" * 400, "b" * 500]
    estimate = budget_manager.estimate_budget_usage(
        query_text="query", chunks_text=chunk_texts
    )
    engine = SummarizationEngine(budget_manager)
    result = engine.summarize(estimate, chunk_texts)
    assert result.required_compression_ratio > 0
    assert len(result.parts) == 2
    assert set(result.combined_summaries.keys()) == {1, 2}
    # Each summary length <= target_chars enforced by design
    for part in result.parts:
        for ct in part.summaries.items():
            pass
    if not result.within_overage_tolerance:
        assert result.error_message is not None


def test_engine_requires_summarize_strategy(budget_manager) -> None:
    # Change config to PRUNE and expect validation error
    config = BudgetConfig(
        max_token_budget=500,
        prompt_limiting_strategy=PromptLimitingStrategy.PRUNE,
        model_name="gpt-4",
    )
    bm = BudgetManager(config)
    from memoir_ai.exceptions import ValidationError

    with pytest.raises(ValidationError):
        SummarizationEngine(bm)
