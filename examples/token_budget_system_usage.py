"""
Token budget estimation and pruning system example for MemoirAI.

This example demonstrates how to use the token budget management system
to handle LLM prompt limits through pruning strategies.
"""

import asyncio
from datetime import datetime, timedelta

from memoir_ai.aggregation import (
    BudgetConfig,
    BudgetManager,
    PromptLimitingStrategy,
    PruningEngine,
    PruningStrategy,
    ResultAggregator,
    create_budget_manager,
    create_result_aggregator,
)
from memoir_ai.query.chunk_retrieval import ChunkResult, QueryResult
from memoir_ai.query.query_strategy_engine import (
    LLMCallResponse,
    QueryClassificationResult,
)


def create_sample_chunks():
    """Create sample chunks for demonstration."""
    base_time = datetime.now()

    return [
        ChunkResult(
            chunk_id=1,
            text_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
            category_path="Technology > AI > Machine Learning",
            category_id_path="1/2/3",
            ranked_relevance=5,
            created_at=base_time,
            source_id="ml_intro.pdf",
        ),
        ChunkResult(
            chunk_id=2,
            text_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            category_path="Technology > AI > Deep Learning",
            category_id_path="1/2/4",
            ranked_relevance=4,
            created_at=base_time + timedelta(minutes=1),
            source_id="dl_guide.pdf",
        ),
        ChunkResult(
            chunk_id=3,
            text_content="Natural language processing enables computers to understand, interpret, and generate human language.",
            category_path="Technology > AI > NLP",
            category_id_path="1/2/5",
            ranked_relevance=4,
            created_at=base_time + timedelta(minutes=2),
            source_id="nlp_basics.pdf",
        ),
        ChunkResult(
            chunk_id=4,
            text_content="Computer vision allows machines to interpret and understand visual information from the world.",
            category_path="Technology > AI > Computer Vision",
            category_id_path="1/2/6",
            ranked_relevance=3,
            created_at=base_time + timedelta(minutes=3),
            source_id="cv_overview.pdf",
        ),
        ChunkResult(
            chunk_id=5,
            text_content="Web development involves creating websites and web applications using various technologies and frameworks.",
            category_path="Technology > Web Development",
            category_id_path="1/7",
            ranked_relevance=2,
            created_at=base_time + timedelta(minutes=4),
            source_id="web_dev.pdf",
        ),
        ChunkResult(
            chunk_id=6,
            text_content="Database management systems store, organize, and retrieve data efficiently for applications.",
            category_path="Technology > Databases",
            category_id_path="1/8",
            ranked_relevance=1,
            created_at=base_time + timedelta(minutes=5),
            source_id="db_systems.pdf",
        ),
    ]


def demonstrate_budget_manager():
    """Demonstrate BudgetManager functionality."""
    print("=== Budget Manager Example ===\n")

    # Create budget manager with different configurations
    configs = [
        ("Small Budget", BudgetConfig(max_token_budget=200, model_name="gpt-4")),
        ("Medium Budget", BudgetConfig(max_token_budget=500, model_name="gpt-4")),
        ("Large Budget", BudgetConfig(max_token_budget=2000, model_name="gpt-4")),
    ]

    query_text = "What are the main applications of artificial intelligence?"
    contextual_helper = "Focus on practical applications and real-world use cases"
    chunks = create_sample_chunks()
    chunk_texts = [chunk.text_content for chunk in chunks]

    for name, config in configs:
        print(f"--- {name} ({config.max_token_budget} tokens) ---")

        manager = BudgetManager(config=config)

        # Estimate budget usage
        estimate = manager.estimate_budget_usage(
            query_text=query_text,
            contextual_helper=contextual_helper,
            chunks_text=chunk_texts,
        )

        print(f"Fixed prompt tokens: {estimate.fixed_prompt_tokens}")
        print(f"Chunks total tokens: {estimate.chunks_total_tokens}")
        print(f"Total tokens: {estimate.total_tokens}")
        print(f"Within budget: {estimate.within_budget}")

        if not estimate.within_budget:
            print(f"Tokens over budget: {estimate.tokens_over_budget}")

            # Validate budget and get recommendations
            validation = manager.validate_budget(estimate)
            print(f"Recommended strategy: {validation.recommended_strategy}")
            if validation.target_tokens:
                print(f"Target tokens for chunks: {validation.target_tokens}")

        print()


def demonstrate_pruning_engine():
    """Demonstrate PruningEngine functionality."""
    print("=== Pruning Engine Example ===\n")

    chunks = create_sample_chunks()

    # Create pruning engine
    def simple_token_counter(text):
        return len(text) // 4  # Rough estimate: 4 chars per token

    engine = PruningEngine(token_counter_func=simple_token_counter)

    # Test different pruning strategies
    strategies = [
        ("Ranking-Based Pruning", PruningStrategy.RANKING_BASED, True),
        ("Deterministic Order Pruning", PruningStrategy.DETERMINISTIC_ORDER, False),
    ]

    target_tokens = 100  # Small budget to force pruning

    for name, strategy, use_rankings in strategies:
        print(f"--- {name} ---")

        result = engine.prune_chunks(
            chunks=chunks,
            target_tokens=target_tokens,
            strategy=strategy,
            use_rankings=use_rankings,
        )

        print(f"Original chunks: {result.original_count}")
        print(f"Kept chunks: {result.kept_count}")
        print(f"Dropped chunks: {result.dropped_count}")
        print(f"Pruning ratio: {result.pruning_ratio:.2%}")
        print(f"Token reduction: {result.token_reduction_ratio:.2%}")
        print(f"Dropped paths: {result.dropped_paths}")

        if result.kept_chunks:
            print("Kept chunks:")
            for chunk in result.kept_chunks:
                print(
                    f"  - ID {chunk.chunk_id}: {chunk.text_content[:50]}... (relevance: {chunk.ranked_relevance})"
                )

        print()


async def demonstrate_result_aggregator():
    """Demonstrate ResultAggregator functionality."""
    print("=== Result Aggregator Example ===\n")

    chunks = create_sample_chunks()

    # Create query result
    query_result = QueryResult(
        chunks=chunks,
        responses=[
            LLMCallResponse(
                llm_output=QueryClassificationResult(category="AI", ranked_relevance=5),
                timestamp=datetime.now(),
                latency_ms=150,
            )
        ],
        total_latency_ms=300,
        total_chunks=len(chunks),
        successful_paths=4,
        failed_paths=0,
    )

    # Test different budget scenarios
    scenarios = [
        ("Large Budget (No Pruning)", 2000),
        ("Medium Budget (Some Pruning)", 400),
        ("Small Budget (Heavy Pruning)", 150),
    ]

    query_text = "Explain the different areas of artificial intelligence"
    contextual_helper = "Provide technical details and examples"

    for name, budget in scenarios:
        print(f"--- {name} ---")

        # Create aggregator with specific budget
        aggregator = create_result_aggregator(
            max_token_budget=budget,
            strategy=PromptLimitingStrategy.PRUNE,
            use_rankings=True,
        )

        # Aggregate results
        result = await aggregator.aggregate_results(
            query_result=query_result,
            query_text=query_text,
            contextual_helper=contextual_helper,
        )

        print(f"Budget: {budget} tokens")
        print(f"Within budget: {result.within_budget}")
        print(f"Final chunks: {len(result.final_chunks)}")
        print(f"Processing time: {result.processing_latency_ms}ms")

        if result.pruning_result:
            print(
                f"Pruning applied: {result.pruning_result.pruning_ratio:.2%} of chunks dropped"
            )
            print(f"Dropped paths: {result.dropped_paths}")

        if result.error_message:
            print(f"Error: {result.error_message}")

        if result.warnings:
            print(f"Warnings: {result.warnings}")

        print()


def demonstrate_budget_analysis():
    """Demonstrate budget analysis capabilities."""
    print("=== Budget Analysis Example ===\n")

    chunks = create_sample_chunks()

    # Create aggregator
    aggregator = create_result_aggregator(
        max_token_budget=300, strategy=PromptLimitingStrategy.PRUNE, use_rankings=True
    )

    # Analyze requirements without performing aggregation
    analysis = aggregator.analyze_aggregation_requirements(
        chunks=chunks,
        query_text="What are the applications of AI?",
        contextual_helper="Focus on practical examples",
    )

    print("Budget Analysis Results:")
    print(f"Total chunks: {analysis['chunk_analysis']['total_chunks']}")
    print(f"Unique paths: {analysis['chunk_analysis']['unique_paths']}")
    print(f"Total tokens: {analysis['token_estimate']['total_tokens']}")
    print(f"Within budget: {analysis['token_estimate']['within_budget']}")
    print(f"Action required: {analysis['action_required']}")

    if analysis["pruning_analysis"]:
        pruning = analysis["pruning_analysis"]
        print(f"Estimated pruning ratio: {pruning['estimated_pruning_ratio']:.2%}")
        print(f"Estimated kept chunks: {pruning['estimated_kept_chunks']}")
        print(f"Paths affected: {pruning['paths_affected']}")

    print(f"Ranking distribution: {analysis['chunk_analysis']['ranking_distribution']}")
    print()


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    print("=== Configuration Options ===\n")

    # Different budget configurations
    configs = [
        BudgetConfig(
            max_token_budget=40000,
            prompt_limiting_strategy=PromptLimitingStrategy.PRUNE,
            use_rankings=True,
            model_name="gpt-4",
        ),
        BudgetConfig(
            max_token_budget=40000,
            prompt_limiting_strategy=PromptLimitingStrategy.PRUNE,
            use_rankings=False,  # Deterministic order only
            model_name="gpt-3.5-turbo",
        ),
        BudgetConfig(
            max_token_budget=40000,
            prompt_limiting_strategy=PromptLimitingStrategy.SUMMARIZE,
            summarization_instruction_headroom_tokens=500,
            summary_char_overage_tolerance_percent=10,
            summary_max_retries=2,
            model_name="gpt-4",
        ),
    ]

    for i, config in enumerate(configs, 1):
        print(f"Configuration {i}:")
        print(f"  Strategy: {config.prompt_limiting_strategy.value}")
        print(f"  Use rankings: {config.use_rankings}")
        print(f"  Model: {config.model_name}")
        print(f"  Max budget: {config.max_token_budget}")

        if config.prompt_limiting_strategy == PromptLimitingStrategy.SUMMARIZE:
            print(
                f"  Headroom tokens: {config.summarization_instruction_headroom_tokens}"
            )
            print(
                f"  Overage tolerance: {config.summary_char_overage_tolerance_percent}%"
            )
            print(f"  Max retries: {config.summary_max_retries}")

        # Create manager and show statistics
        manager = BudgetManager(config=config)
        stats = manager.get_budget_statistics()
        print(f"  LiteLLM available: {stats['capabilities']['litellm_available']}")
        print()


async def main():
    """Run all demonstrations."""
    print("=== MemoirAI Token Budget System Examples ===\n")

    # Run demonstrations
    demonstrate_budget_manager()
    demonstrate_pruning_engine()
    await demonstrate_result_aggregator()
    demonstrate_budget_analysis()
    demonstrate_configuration_options()

    print("=== Key Features Demonstrated ===")
    print("✓ Token counting with liteLLM integration")
    print("✓ Budget validation and recommendations")
    print("✓ Ranking-based and deterministic pruning strategies")
    print("✓ Path diversity preservation")
    print("✓ Comprehensive result aggregation")
    print("✓ Performance tracking and metrics")
    print("✓ Error handling and fallback mechanisms")
    print("✓ Configurable budget management")
    print()

    print("This system provides robust token budget management for LLM applications,")
    print(
        "ensuring prompts stay within limits while preserving the most relevant content."
    )


if __name__ == "__main__":
    print("Running token budget system examples...\n")
    asyncio.run(main())
    print("Examples completed!")
