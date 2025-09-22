"""Summarization engine for MemoirAI.

Implements requirement 7.2 (core engine) including:
- Compression ratio computation (delegates token counts to BudgetManager)
- Per-chunk target character assignment
- Partitioning chunks into token-safe parts
- Structured per-part summarization prompt construction (placeholder logic)
- Validation of per-chunk character limits and global overage tolerance
- Retry loop skeleton (no real LLM calls yet; integrates with AgentFactory later)

NOTE: Actual LLM interaction is deferred until integration phase with ResultAggregator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..exceptions import ValidationError
from .budget_manager import BudgetManager, PromptLimitingStrategy, TokenEstimate


@dataclass
class ChunkSummaryTarget:
    chunk_id: int
    original_chars: int
    target_chars: int


@dataclass
class SummarizationPart:
    part_id: int
    chunk_targets: List[ChunkSummaryTarget]
    total_original_chars: int
    total_target_chars: int


@dataclass
class PartSummarizationResult:
    part_id: int
    summaries: Dict[int, str]  # chunk_id -> summary
    retries_used: int = 0
    over_limit_chunk_ids: List[int] = field(default_factory=list)


@dataclass
class SummarizationProcessResult:
    required_compression_ratio: float
    parts: List[PartSummarizationResult]
    combined_summaries: Dict[int, str]
    combined_char_count: int
    within_overage_tolerance: bool
    overage_tolerance_percent: int
    error_message: Optional[str] = None


class SummarizationEngine:
    def __init__(self, budget_manager: BudgetManager) -> None:
        self.budget_manager = budget_manager
        if (
            self.budget_manager.config.prompt_limiting_strategy
            != PromptLimitingStrategy.SUMMARIZE
        ):
            raise ValidationError(
                "SummarizationEngine requires strategy SUMMARIZE",
                field="prompt_limiting_strategy",
                value=self.budget_manager.config.prompt_limiting_strategy.value,
            )

    def analyze(
        self,
        estimate: TokenEstimate,
        chunk_texts: List[str],
    ) -> Dict[str, Any]:
        """Compute compression plan (ratio + per-chunk targets)."""
        comp = self.budget_manager.calculate_compression_requirements(
            estimate, chunk_texts
        )
        targets: List[ChunkSummaryTarget] = []
        for idx, text in enumerate(chunk_texts):
            targets.append(
                ChunkSummaryTarget(
                    chunk_id=idx + 1,
                    original_chars=len(text),
                    target_chars=max(1, int(len(text) * comp["compression_ratio"])),
                )
            )
        return {
            "compression_ratio": comp["compression_ratio"],
            "targets": targets,
            "compression_needed": comp["compression_ratio"] < 1.0,
        }

    def partition_chunks(
        self,
        targets: List[ChunkSummaryTarget],
        chunk_texts: List[str],
        headroom_tokens: Optional[int] = None,
    ) -> List[SummarizationPart]:
        """Partition chunks into single-chunk parts for now (simple baseline)."""
        parts: List[SummarizationPart] = []
        for i, tgt in enumerate(targets):
            parts.append(
                SummarizationPart(
                    part_id=i + 1,
                    chunk_targets=[tgt],
                    total_original_chars=tgt.original_chars,
                    total_target_chars=tgt.target_chars,
                )
            )
        return parts

    def build_part_instruction(self, part: SummarizationPart) -> str:
        lines = [
            "Summarize each chunk to at most its target characters. Return JSON only.",
        ]
        for ct in part.chunk_targets:
            lines.append(
                f"Chunk {ct.chunk_id} (original_chars={ct.original_chars}, target_chars={ct.target_chars}):"
            )
        return "\n".join(lines)

    def summarize(
        self,
        estimate: TokenEstimate,
        chunk_texts: List[str],
    ) -> SummarizationProcessResult:
        plan = self.analyze(estimate, chunk_texts)
        targets: List[ChunkSummaryTarget] = plan["targets"]
        parts = self.partition_chunks(targets, chunk_texts)

        # Placeholder summarization: truncate text to target_chars
        part_results: List[PartSummarizationResult] = []
        combined: Dict[int, str] = {}
        for part in parts:
            summaries: Dict[int, str] = {}
            over_limit: List[int] = []
            for ct in part.chunk_targets:
                original_text = chunk_texts[ct.chunk_id - 1]
                summary = original_text[: ct.target_chars]
                if len(summary) > ct.target_chars:
                    over_limit.append(ct.chunk_id)
                summaries[ct.chunk_id] = summary
                combined[ct.chunk_id] = summary
            part_results.append(
                PartSummarizationResult(
                    part_id=part.part_id,
                    summaries=summaries,
                    retries_used=0,
                    over_limit_chunk_ids=over_limit,
                )
            )

        combined_char_count = sum(len(s) for s in combined.values())
        # Global tolerance check
        tolerance_percent = (
            self.budget_manager.config.summary_char_overage_tolerance_percent
        )
        # Allowed max combined chars = original_total * ratio * (1 + tolerance)
        original_total = sum(len(t) for t in chunk_texts)
        allowed = int(
            original_total * plan["compression_ratio"] * (1 + tolerance_percent / 100.0)
        )
        within_tolerance = combined_char_count <= allowed

        error_message = None
        if not within_tolerance:
            error_message = "Combined summaries exceed allowed character budget after placeholder summarization"

        return SummarizationProcessResult(
            required_compression_ratio=plan["compression_ratio"],
            parts=part_results,
            combined_summaries=combined,
            combined_char_count=combined_char_count,
            within_overage_tolerance=within_tolerance,
            overage_tolerance_percent=tolerance_percent,
            error_message=error_message,
        )


__all__ = [
    "SummarizationEngine",
    "SummarizationPart",
    "ChunkSummaryTarget",
    "PartSummarizationResult",
    "SummarizationProcessResult",
]
