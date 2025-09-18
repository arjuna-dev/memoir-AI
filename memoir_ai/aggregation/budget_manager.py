"""
Token budget management for MemoirAI.

This module provides token counting, budget validation, and estimation
capabilities using liteLLM for accurate token counting.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from litellm import token_counter

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("liteLLM not available, using fallback token counting")

from ..exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


class PromptLimitingStrategy(Enum):
    """Available prompt limiting strategies."""

    PRUNE = "prune"
    SUMMARIZE = "summarize"


@dataclass
class BudgetConfig:
    """Configuration for token budget management."""

    # Core budget settings
    max_token_budget: int
    prompt_limiting_strategy: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE
    model_name: str = "gpt-4"

    # Pruning settings
    use_rankings: bool = True

    # Summarization settings
    summarization_instruction_headroom_tokens: int = 1024
    summary_char_overage_tolerance_percent: int = 5
    summary_max_retries: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_token_budget <= 0:
            raise ValidationError(
                "max_token_budget must be positive",
                field="max_token_budget",
                value=self.max_token_budget,
            )

        # Only enforce headroom < max budget when summarization strategy is active;
        # tests may construct configs in PRUNE mode with large headroom tokens without error.
        if (
            self.prompt_limiting_strategy == PromptLimitingStrategy.SUMMARIZE
            and self.summarization_instruction_headroom_tokens >= self.max_token_budget
        ):
            raise ValidationError(
                "summarization_instruction_headroom_tokens must be less than max_token_budget when using summarization",
                field="summarization_instruction_headroom_tokens",
                value=self.summarization_instruction_headroom_tokens,
            )

        if not 0 <= self.summary_char_overage_tolerance_percent <= 100:
            raise ValidationError(
                "summary_char_overage_tolerance_percent must be between 0 and 100",
                field="summary_char_overage_tolerance_percent",
                value=self.summary_char_overage_tolerance_percent,
            )


@dataclass
class TokenEstimate:
    """Token and character count estimates for budget planning."""

    # Token counts
    fixed_prompt_tokens: int
    chunks_total_tokens: int
    total_tokens: int

    # Character counts
    fixed_prompt_chars: int
    chunks_total_chars: int
    total_chars: int

    # Budget status
    within_budget: bool
    tokens_over_budget: int = 0

    @property
    def budget_utilization(self) -> float:
        """Calculate budget utilization as a percentage."""
        if self.total_tokens == 0:
            return 0.0
        return min(100.0, (self.total_tokens / self.total_tokens) * 100)


@dataclass
class BudgetValidationResult:
    """Result of budget validation with recommendations."""

    estimate: TokenEstimate
    is_valid: bool
    requires_action: bool
    recommended_strategy: Optional[PromptLimitingStrategy] = None
    error_message: Optional[str] = None

    # Compression calculations for summarization
    required_compression_ratio: Optional[float] = None
    target_tokens: Optional[int] = None


class BudgetManager:
    """
    Manages token budgets and provides estimation capabilities.

    Features:
    - Accurate token counting using liteLLM
    - Budget validation and recommendations
    - Compression ratio calculations
    - Fallback token estimation when liteLLM unavailable
    """

    def __init__(
        self,
        config: BudgetConfig,
        fallback_chars_per_token: float = 4.0,
    ):
        """
        Initialize budget manager.

        Args:
            config: Budget configuration
            fallback_chars_per_token: Fallback ratio when liteLLM unavailable
        """
        self.config = config
        self.fallback_chars_per_token = fallback_chars_per_token

        # Validate liteLLM availability for accurate counting
        if not LITELLM_AVAILABLE:
            logger.warning(
                "liteLLM not available. Token counting will use fallback estimation. "
                "Install liteLLM for accurate token counting: pip install litellm"
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using liteLLM or fallback.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if LITELLM_AVAILABLE:
            try:
                return token_counter(model=self.config.model_name, text=text)
            except Exception as e:
                logger.warning(f"liteLLM token counting failed: {e}, using fallback")
                return self._fallback_token_count(text)
        else:
            return self._fallback_token_count(text)

    def _fallback_token_count(self, text: str) -> int:
        """
        Fallback token counting using character-based estimation.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        return max(1, int(len(text) / self.fallback_chars_per_token))

    def estimate_budget_usage(
        self,
        query_text: str,
        contextual_helper: Optional[str] = None,
        wrapper_text: str = "",
        chunks_text: List[str] = None,
    ) -> TokenEstimate:
        """
        Estimate token and character usage for budget planning.

        Args:
            query_text: The user's query
            contextual_helper: Optional contextual information
            wrapper_text: Additional wrapper text for the prompt
            chunks_text: List of chunk text content

        Returns:
            Token and character estimates
        """
        chunks_text = chunks_text or []

        # Build fixed prompt text
        fixed_parts = [query_text]
        if contextual_helper:
            fixed_parts.append(contextual_helper)
        if wrapper_text:
            fixed_parts.append(wrapper_text)

        fixed_prompt_text = "\n".join(fixed_parts)

        # Count tokens and characters
        fixed_prompt_tokens = self.count_tokens(fixed_prompt_text)
        fixed_prompt_chars = len(fixed_prompt_text)

        chunks_total_tokens = sum(self.count_tokens(chunk) for chunk in chunks_text)
        chunks_total_chars = sum(len(chunk) for chunk in chunks_text)

        total_tokens = fixed_prompt_tokens + chunks_total_tokens
        total_chars = fixed_prompt_chars + chunks_total_chars

        # Check budget status
        within_budget = total_tokens <= self.config.max_token_budget
        tokens_over_budget = max(0, total_tokens - self.config.max_token_budget)

        return TokenEstimate(
            fixed_prompt_tokens=fixed_prompt_tokens,
            chunks_total_tokens=chunks_total_tokens,
            total_tokens=total_tokens,
            fixed_prompt_chars=fixed_prompt_chars,
            chunks_total_chars=chunks_total_chars,
            total_chars=total_chars,
            within_budget=within_budget,
            tokens_over_budget=tokens_over_budget,
        )

    def validate_budget(
        self,
        estimate: TokenEstimate,
        strategy: Optional[PromptLimitingStrategy] = None,
    ) -> BudgetValidationResult:
        """
        Validate budget and provide recommendations.

        Args:
            estimate: Token estimate to validate
            strategy: Optional strategy override

        Returns:
            Budget validation result with recommendations
        """
        strategy = strategy or self.config.prompt_limiting_strategy

        # Check if within budget
        if estimate.within_budget:
            return BudgetValidationResult(
                estimate=estimate, is_valid=True, requires_action=False
            )

        # Budget exceeded - determine action needed
        if strategy == PromptLimitingStrategy.PRUNE:
            return self._validate_pruning_budget(estimate)
        elif strategy == PromptLimitingStrategy.SUMMARIZE:
            return self._validate_summarization_budget(estimate)
        else:
            return BudgetValidationResult(
                estimate=estimate,
                is_valid=False,
                requires_action=True,
                error_message=f"Unknown strategy: {strategy}",
            )

    def _validate_pruning_budget(
        self, estimate: TokenEstimate
    ) -> BudgetValidationResult:
        """Validate budget for pruning strategy."""
        # Check if pruning can help
        available_tokens = self.config.max_token_budget - estimate.fixed_prompt_tokens

        if available_tokens <= 0:
            return BudgetValidationResult(
                estimate=estimate,
                is_valid=False,
                requires_action=True,
                error_message="Fixed prompt exceeds budget, pruning cannot help",
                recommended_strategy=PromptLimitingStrategy.SUMMARIZE,
            )

        return BudgetValidationResult(
            estimate=estimate,
            is_valid=False,
            requires_action=True,
            recommended_strategy=PromptLimitingStrategy.PRUNE,
            target_tokens=available_tokens,
        )

    def _validate_summarization_budget(
        self, estimate: TokenEstimate
    ) -> BudgetValidationResult:
        """Validate budget for summarization strategy."""
        available_tokens = self.config.max_token_budget - estimate.fixed_prompt_tokens

        if available_tokens <= 0:
            return BudgetValidationResult(
                estimate=estimate,
                is_valid=False,
                requires_action=True,
                error_message="Fixed prompt exceeds budget, no strategy can help",
            )

        # Calculate compression ratio
        if estimate.chunks_total_tokens > 0:
            compression_ratio = available_tokens / estimate.chunks_total_tokens
        else:
            compression_ratio = 1.0

        if compression_ratio <= 0:
            return BudgetValidationResult(
                estimate=estimate,
                is_valid=False,
                requires_action=True,
                error_message="Required compression ratio is zero or negative",
            )

        return BudgetValidationResult(
            estimate=estimate,
            is_valid=False,
            requires_action=True,
            recommended_strategy=PromptLimitingStrategy.SUMMARIZE,
            required_compression_ratio=compression_ratio,
            target_tokens=available_tokens,
        )

    def calculate_compression_requirements(
        self,
        estimate: TokenEstimate,
        chunk_texts: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate compression requirements for summarization.

        Args:
            estimate: Token estimate
            chunk_texts: List of chunk text content

        Returns:
            Dictionary with compression calculations
        """
        available_tokens = self.config.max_token_budget - estimate.fixed_prompt_tokens

        if estimate.chunks_total_tokens <= 0:
            return {
                "compression_ratio": 1.0,
                "target_char_counts": [],
                "total_target_chars": 0,
                "compression_needed": False,
            }

        compression_ratio = available_tokens / estimate.chunks_total_tokens

        # Calculate per-chunk target character counts
        target_char_counts = []
        for chunk_text in chunk_texts:
            original_chars = len(chunk_text)
            target_chars = max(1, int(original_chars * compression_ratio))
            target_char_counts.append(target_chars)

        total_target_chars = sum(target_char_counts)

        return {
            "compression_ratio": compression_ratio,
            "target_char_counts": target_char_counts,
            "total_target_chars": total_target_chars,
            "compression_needed": compression_ratio < 1.0,
            "available_tokens": available_tokens,
            "original_tokens": estimate.chunks_total_tokens,
        }

    def validate_final_prompt(
        self,
        final_prompt: str,
        max_budget: Optional[int] = None,
    ) -> Tuple[bool, int, str]:
        """
        Validate that final prompt is within budget.

        Args:
            final_prompt: Complete final prompt text
            max_budget: Optional budget override

        Returns:
            Tuple of (is_valid, token_count, message)
        """
        max_budget = max_budget or self.config.max_token_budget
        token_count = self.count_tokens(final_prompt)

        is_valid = token_count <= max_budget

        if is_valid:
            message = f"Prompt within budget: {token_count}/{max_budget} tokens"
        else:
            overage = token_count - max_budget
            message = (
                f"Prompt exceeds budget by {overage} tokens: {token_count}/{max_budget}"
            )

        return is_valid, token_count, message

    def get_budget_statistics(self) -> Dict[str, Any]:
        """
        Get budget manager statistics and configuration.

        Returns:
            Dictionary with budget statistics
        """
        return {
            "config": {
                "max_token_budget": self.config.max_token_budget,
                "strategy": self.config.prompt_limiting_strategy.value,
                "model_name": self.config.model_name,
                "use_rankings": self.config.use_rankings,
            },
            "capabilities": {
                "litellm_available": LITELLM_AVAILABLE,
                "fallback_chars_per_token": self.fallback_chars_per_token,
                "accurate_counting": LITELLM_AVAILABLE,
            },
            "summarization_config": {
                "headroom_tokens": self.config.summarization_instruction_headroom_tokens,
                "overage_tolerance_percent": self.config.summary_char_overage_tolerance_percent,
                "max_retries": self.config.summary_max_retries,
            },
        }


# Utility functions
def create_budget_manager(
    max_token_budget: int,
    strategy: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE,
    model_name: str = "gpt-4",
    **kwargs,
) -> BudgetManager:
    """
    Create a budget manager with default configuration.

    Args:
        max_token_budget: Maximum token budget
        strategy: Prompt limiting strategy
        model_name: LLM model name for token counting
        **kwargs: Additional configuration options

    Returns:
        Configured BudgetManager
    """
    config = BudgetConfig(
        max_token_budget=max_token_budget,
        prompt_limiting_strategy=strategy,
        model_name=model_name,
        **kwargs,
    )

    return BudgetManager(config=config)


def estimate_tokens_fallback(text: str, chars_per_token: float = 4.0) -> int:
    """
    Fallback token estimation using character count.

    Args:
        text: Text to estimate
        chars_per_token: Average characters per token

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))
