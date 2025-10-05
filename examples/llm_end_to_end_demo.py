"""End-to-end example using real LLM calls via Pydantic AI."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
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
        query_text="What has been said about the 'Seven-Front War'?",
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

    # Check if there's only one level 1 category in the database to determine if we should skip it
    level_one_categories = memoir.category_manager.get_existing_categories(level=1)
    skip_level_1 = len(level_one_categories) == 1

    # Write chunks to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chunk_results/chunk_results_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Chunk Results - Generated at {datetime.now().isoformat()}\n")
        f.write(f"Query: What has been said about the 'Seven-Front War'?\n")
        f.write(f"Total chunks: {query_result.total_chunks}\n")
        f.write("=" * 80 + "\n\n")

        for i, chunk in enumerate(query_result.chunks, 1):
            # Format category path (skip level 1 if there's only one)
            if chunk.category_path:
                # Try different separators that might be used
                if " → " in chunk.category_path:
                    path_parts = chunk.category_path.split(" → ")
                    separator = " → "
                elif " > " in chunk.category_path:
                    path_parts = chunk.category_path.split(" > ")
                    separator = " > "
                else:
                    path_parts = [chunk.category_path]
                    separator = " → "

                if skip_level_1 and len(path_parts) > 1:
                    display_path = separator.join(path_parts[1:])
                else:
                    display_path = chunk.category_path
            else:
                display_path = "No category path"

            f.write(f"CHUNK {i}\n")
            f.write(f"Category Path: {display_path}\n")
            f.write("-" * 40 + "\n")
            f.write(chunk.text_content)
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f"Chunk results written to: {filename}")

    # Also display preview in console (with same path formatting as file)
    for chunk in query_result.chunks:
        # Format category path (skip level 1 if there's only one)
        if chunk.category_path:
            # Try different separators that might be used
            if " → " in chunk.category_path:
                path_parts = chunk.category_path.split(" → ")
                separator = " → "
            elif " > " in chunk.category_path:
                path_parts = chunk.category_path.split(" > ")
                separator = " > "
            else:
                path_parts = [chunk.category_path]
                separator = " → "

            if skip_level_1 and len(path_parts) > 1:
                display_path = separator.join(path_parts[1:])
            else:
                display_path = chunk.category_path
        else:
            display_path = "No category path"

        preview = chunk.text_content[:200].replace("\n", " ")
        print(f"Category Path: {display_path}")
        print(f"Preview: {preview}...")
        print("-")

    memoir.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
