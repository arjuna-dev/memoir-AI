#!/usr/bin/env python3
"""
Test script for the new hierarchical batch classification functionality.

This script demonstrates the new classify_all_chunks functionality that classifies
all chunks in a single LLM call using hierarchical category trees.
"""

import asyncio
import os
import tempfile

from memoir_ai import MemoirAI
from memoir_ai.llm.llm_models import Models


async def test_hierarchical_batch_classification():
    """Test the new hierarchical batch classification functionality."""

    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        # Initialize MemoirAI
        print("Initializing MemoirAI...")
        memoir = MemoirAI(
            database_url=f"sqlite:///{db_path}",
            model=Models.openai_gpt_5_nano,
            hierarchy_depth=3,
            batch_size=5,
        )

        # Test content that should generate multiple chunks
        test_content = """
        The United Nations General Assembly witnessed a dramatic walkout yesterday as delegates 
        protested the Israeli Prime Minister's speech. Benjamin Netanyahu addressed the assembly 
        despite widespread international criticism regarding the ongoing conflict in Gaza.
        
        In his speech, Netanyahu defended Israel's military actions and rejected calls for 
        Palestinian statehood. He accused international critics of engaging in "lawfare" 
        against Israel and claimed that Israel was fighting a "seven-front war" against 
        what he termed "barbarism."
        
        The walkout was organized by several Arab and European nations who have been 
        calling for stronger international action against Israel's military operations 
        in Gaza. The protest highlighted the deep divisions within the international 
        community regarding the Palestine-Israel conflict.
        
        Human rights organizations have condemned the civilian casualties in Gaza, 
        while Israel maintains that its actions are necessary for national security. 
        The International Criminal Court has issued arrest warrants for Netanyahu, 
        adding to the international pressure.
        """

        contextual_helper = """
        title="UN Delegates Walk Out on Netanyahu Speech"
        author="Test Author"
        date="2025-10-01"
        topic="News Article About Palestine-Israel Conflict"
        source_type="news_article"
        description="UN delegates lead a mass walkout as Netanyahu defends Israeli actions in Gaza"
        """

        print("Testing hierarchical batch classification...")
        print(f"Content length: {len(test_content)} characters")

        # This will now use classify_all_chunks instead of classify_chunks
        result = await memoir.ingest_text(
            content=test_content,
            source_id="test_hierarchical_batch_classification",
            contextual_helper=contextual_helper,
        )

        print("\n=== Ingestion Results ===")
        print(f"Success: {result.success}")
        print(f"Chunks processed: {result.chunks_processed}")
        print(f"Chunks stored: {result.chunks_stored}")
        print(f"Categories created: {result.categories_created}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Total tokens: {result.total_tokens}")

        if result.chunk_details:
            print("\n=== Chunk Details ===")
            for i, detail in enumerate(result.chunk_details):
                print(f"Chunk {i+1}:")
                print(f"  Category Path: {detail['category_path']}")
                print(f"  Token Count: {detail['token_count']}")
                print(
                    f"  Classification Latency: {detail['classification_latency_ms']}ms"
                )

        if result.error_message:
            print(f"\nError: {result.error_message}")

        if result.warnings:
            print(f"\nWarnings: {result.warnings}")

        # Get category tree to see the hierarchical structure
        print("\n=== Category Tree ===")
        category_tree = memoir.get_category_tree()
        print(f"Total categories: {category_tree.total_categories}")
        print(f"Max depth: {category_tree.max_depth}")
        print(f"Categories by level: {category_tree.categories_by_level}")

        memoir.close()
        return result.success

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up temporary database
        try:
            os.unlink(db_path)
        except:
            pass


if __name__ == "__main__":
    print("Testing Hierarchical Batch Classification")
    print("=" * 50)

    # Note: This test requires OpenAI API key to be set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY environment variable not set.")
        print(
            "This test will fail at LLM interaction step, but code structure is verified."
        )

    success = asyncio.run(test_hierarchical_batch_classification())

    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
