"""End-to-end example using real LLM calls via Pydantic AI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from memoir_ai.core import MemoirAI
from memoir_ai.query.query_strategy_engine import QueryStrategy


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
        hierarchy_depth=3,
        chunk_min_tokens=80,
        chunk_max_tokens=220,
    )

    sample_text = """
MemoirAI is an experimental knowledge management system. It focuses on
creating rich hierarchical categories that capture the nuance of long-form
content. The project combines automated text chunking with LLM powered
classification to organize articles, research papers, meeting notes, and more.
"""

    print("\nIngesting sample document using live classification...\n")
    ingestion_result = await memoir.ingest_text(
        content=sample_text.strip(),
        source_id="demo-doc",
        contextual_helper="Internal README for the MemoirAI library",
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
        query_text="How does MemoirAI organize documents?",
        strategy=QueryStrategy.ONE_SHOT,
        contextual_helper="User wants a brief explanation of the system",
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
        preview = chunk.text_content[:120].replace("\n", " ")
        print(f"Category Path: {chunk.category_path}")
        print(f"Preview: {preview}...")
        print("-")

    memoir.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
