"""End-to-end example using real LLM calls via Pydantic AI."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from memoir_ai.core import MemoirAI
from memoir_ai.query.query_strategy_engine import QueryStrategy
from memoir_ai.text_processing.contextual_helper import ContextualHelperGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)


async def run_demo() -> None:
    """Execute ingestion and querying with live LLM interactions."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "\n⚠️  Please set the OPENAI_API_KEY environment variable before running this demo.\n"
        )
        return

    database_path = Path("./memoir_llm_demo.db")

    if database_path.exists():
        database_path.unlink()

    memoir = MemoirAI(
        database_url=f"sqlite:///{database_path}",
        model_name="openai:gpt-4o-mini",
        hierarchy_depth=4,
        chunk_min_tokens=80,
        chunk_max_tokens=220,
    )

    contextual_helper = ContextualHelperGenerator()
    user_provided_context = contextual_helper.create_user_provided_helper(
        title="UN Delegates Walk Out on Netanyahu Speech",
        author="Jane Doe",
        date="2025-10-09",
        topic="News Article About Palestine-Israel Conflict",
        source_type="news_article",
        description=(
            "UN delegates lead a mass walkout as Netanyahu insists Israel must ‘finish the job’ in Gaza"
            "Delegates stormed out during Benjamin Netanyahu’s speech as he lambasted nations for ‘caving’ to Hamas"
        ),
    )

    from sample_data_set.news_26_09_2025_0 import news_article

    sample_text = news_article
    print("\nIngesting sample document using live classification...\n")
    ingestion_result = await memoir.ingest_text(
        content=sample_text.strip(),
        source_id="demo-doc",
        contextual_helper=user_provided_context,
    )

    if not ingestion_result.success:
        print("Ingestion failed:", ingestion_result.error_message)
        return

    print(
        f"Stored {ingestion_result.chunks_stored} chunks across "
        f"{ingestion_result.categories_created} new categories."
    )
    for detail in ingestion_result.chunk_details:
        print(
            f"  - Chunk {detail['chunk_index']} classified into path "
            f"{detail['category_path']} (latency: {detail['classification_latency_ms']} ms)"
        )

    print("\nRunning a query that triggers LLM category selection...\n")
    query_result = await memoir.query_processor.process_query(
        query_text="What happened with Netanyahu?",
        # strategy=QueryStrategy.ZOOM_IN,
        strategy=QueryStrategy.ONE_SHOT,
        chunk_limit_per_path=5,
    )

    for response in query_result.responses:
        llm_output = response.llm_output
        if llm_output:
            print(
                f"LLM selected path with relevance {llm_output.ranked_relevance}: "
                f"{response.llm_output.category}"
            )

    print(f"\nRetrieved {query_result.total_chunks} chunk(s) for inspection.\n")
    for chunk in query_result.chunks:
        preview = chunk.text_content[:200].replace("\n", " ")
        print(f"Category Path: {chunk.category_path[1:]}")
        print(f"Preview: {preview}...")
        print("-")

    memoir.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
