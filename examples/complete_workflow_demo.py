"""
Complete workflow demonstration for MemoirAI library.
Shows integration of database, migration, and text chunking systems.
"""

import asyncio
import os
import tempfile

from memoir_ai import Category, Chunk, MemoirAI, TextChunk
from memoir_ai.exceptions import ConfigurationError, DatabaseError, ValidationError


async def main():
    """Demonstrate complete MemoirAI workflow with all implemented features."""

    print("🚀 MemoirAI Complete Workflow Demo")
    print("=" * 50)

    # Create temporary database for demonstration
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Step 1: Initialize MemoirAI with all systems
        print("\n1. 🏗️  Initializing MemoirAI with all systems...")

        memoir = MemoirAI(
            database_url=f"sqlite:///{db_path}",
            llm_provider="openai",
            model_name="gpt-4o-mini",
            hierarchy_depth=3,
            chunk_min_tokens=15,
            chunk_max_tokens=40,
            batch_size=5,
            max_categories_per_level=50,
            max_token_budget=40000,
        )

        print("✅ MemoirAI initialized successfully!")
        print(f"   Database: {memoir.config.database_url}")
        print(f"   Model: {memoir.config.model_name}")
        print(
            f"   Chunk size: {memoir.config.chunk_min_tokens}-{memoir.config.chunk_max_tokens} tokens"
        )

        # Step 2: Verify database and migration system
        print("\n2. 🗄️  Verifying database and migration system...")

        db_info = memoir.get_database_info()
        print(f"   Schema valid: {db_info['schema_valid']}")
        print(f"   Current revision: {db_info['current_revision']}")
        print(f"   Tables: {list(db_info['table_info'].keys())}")

        for table_name, info in db_info["table_info"].items():
            if "row_count" in info:
                print(f"     - {table_name}: {info['row_count']} rows")

        # Step 3: Text processing and chunking
        print("\n3. 📝 Text processing and chunking...")

        sample_document = """
        Artificial Intelligence and Machine Learning Overview
        
        Artificial intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.
        
        Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.
        
        Natural Language Processing (NLP) is another important subfield of AI that focuses on the interaction between computers and human language. NLP enables machines to read, understand, and derive meaning from human language in a valuable way.
        """

        # Chunk the document
        chunks = memoir.chunk_text(
            content=sample_document,
            source_id="ai_ml_overview_2024",
            metadata={
                "document_type": "educational",
                "topic": "AI/ML",
                "author": "demo_system",
                "created_date": "2024-01-01",
            },
        )

        print(f"   Original document: {len(sample_document)} characters")
        print(f"   Number of chunks created: {len(chunks)}")

        # Show chunk details
        for i, chunk in enumerate(chunks, 1):
            preview = chunk.content.replace("\n", " ").strip()[:60] + "..."
            print(f"   Chunk {i}: {chunk.token_count} tokens - {preview}")

        # Get chunking statistics
        stats = memoir.get_chunking_stats(chunks)
        print(f"\n   Chunking Statistics:")
        print(f"     Total tokens: {stats['total_tokens']}")
        print(f"     Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"     Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"     Chunks below minimum: {stats['chunks_below_min']}")
        print(f"     Chunks above maximum: {stats['chunks_above_max']}")

        # Step 4: Database storage simulation
        print("\n4. 💾 Database storage simulation...")

        # Create category hierarchy manually (simulating what classification would do)
        with memoir.db_manager.get_session() as session:
            # Create categories
            tech_category = Category(name="Technology", level=1, parent_id=None)
            session.add(tech_category)
            session.flush()

            ai_category = Category(
                name="Artificial Intelligence", level=2, parent_id=tech_category.id
            )
            session.add(ai_category)
            session.flush()

            ml_category = Category(
                name="Machine Learning", level=3, parent_id=ai_category.id
            )
            session.add(ml_category)
            session.flush()

            print(f"   Created category hierarchy:")
            print(f"     L1: {tech_category.name}")
            print(f"     L2: {ai_category.name}")
            print(f"     L3: {ml_category.name}")

            # Store chunks in database
            stored_chunks = []
            for i, text_chunk in enumerate(chunks):
                db_chunk = Chunk(
                    content=text_chunk.content,
                    token_count=text_chunk.token_count,
                    category_id=ml_category.id,
                    source_id=text_chunk.source_id,
                    source_metadata=text_chunk.metadata,
                )
                session.add(db_chunk)
                stored_chunks.append(db_chunk)

            session.flush()
            print(f"   Stored {len(stored_chunks)} chunks in database")

        # Step 5: Verify data persistence and retrieval
        print("\n5. 🔍 Verifying data persistence and retrieval...")

        with memoir.db_manager.get_session() as session:
            # Retrieve categories
            categories = session.query(Category).all()
            print(f"   Categories in database: {len(categories)}")

            for category in categories:
                path = category.get_path_string()
                chunk_count = len(category.chunks) if category.chunks else 0
                print(f"     {path} ({chunk_count} chunks)")

            # Retrieve chunks
            chunks_from_db = session.query(Chunk).all()
            print(f"   Chunks in database: {len(chunks_from_db)}")

            for i, chunk in enumerate(chunks_from_db[:3], 1):  # Show first 3
                preview = (
                    chunk.content[:50] + "..."
                    if len(chunk.content) > 50
                    else chunk.content
                )
                category_path = chunk.category.get_path_string()
                print(f"     Chunk {i}: [{category_path}] {preview}")

        # Step 6: Database statistics and health check
        print("\n6. 📊 Database statistics and health check...")

        updated_db_info = memoir.get_database_info()
        print(f"   Database health:")
        for table_name, info in updated_db_info["table_info"].items():
            if "row_count" in info:
                print(f"     - {table_name}: {info['row_count']} rows")

        # Step 7: Configuration and system information
        print("\n7. ⚙️  System configuration summary...")

        print(f"   Configuration:")
        print(f"     Database URL: {memoir.config.database_url}")
        print(f"     LLM Provider: {memoir.config.llm_provider}")
        print(f"     Model: {memoir.config.model_name}")
        print(f"     Hierarchy Depth: {memoir.config.hierarchy_depth}")
        print(
            f"     Chunk Token Range: {memoir.config.chunk_min_tokens}-{memoir.config.chunk_max_tokens}"
        )
        print(f"     Batch Size: {memoir.config.batch_size}")
        print(f"     Max Token Budget: {memoir.config.max_token_budget}")
        print(
            f"     Max Categories Per Level: {memoir.config.max_categories_per_level}"
        )

        # Step 8: Error handling demonstration
        print("\n8. 🛡️  Error handling demonstration...")

        try:
            # Try invalid chunking
            memoir.chunk_text("")
            print("   ❌ This should have failed!")
        except ValidationError as e:
            print(f"   ✅ Validation error handled: {str(e)[:60]}...")

        try:
            # Try invalid configuration
            invalid_memoir = MemoirAI(
                database_url="sqlite:///test.db",
                chunk_min_tokens=1000,
                chunk_max_tokens=500,  # Invalid: min > max
            )
            print("   ❌ This should have failed!")
        except (ValidationError, ConfigurationError) as e:
            print(f"   ✅ Configuration error handled: {str(e)[:60]}...")

        # Step 9: Performance and scalability test
        print("\n9. 🚄 Performance and scalability test...")

        # Test with larger document
        large_document = sample_document * 5  # 5x larger

        import time

        start_time = time.time()
        large_chunks = memoir.chunk_text(large_document, source_id="large_test")
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

        print(f"   Large document processing:")
        print(f"     Document size: {len(large_document)} characters")
        print(f"     Chunks created: {len(large_chunks)}")
        print(f"     Processing time: {processing_time:.1f}ms")
        print(
            f"     Throughput: {len(large_document) / processing_time * 1000:.0f} chars/sec"
        )

        # Step 10: Future integration points
        print("\n10. 🔮 Future integration points...")

        print("   ✅ Completed Systems:")
        print("     - Database models and migrations")
        print("     - Token-based text chunking")
        print("     - Configuration management")
        print("     - Error handling and validation")

        print("   🚧 Next Implementation Steps:")
        print("     - Contextual helper generation")
        print("     - LLM classification with Pydantic AI")
        print("     - Category hierarchy management")
        print("     - Query processing and result aggregation")
        print("     - Complete ingestion and retrieval pipeline")

        print("\n🎉 Complete workflow demonstration finished!")
        print("\n📋 System Status:")
        print("   ✅ Database: Initialized and operational")
        print("   ✅ Migrations: Schema versioning active")
        print("   ✅ Text Chunking: Token-based processing ready")
        print("   ✅ Configuration: Validated and loaded")
        print("   ✅ Error Handling: Comprehensive validation")
        print("   ✅ Integration: All systems working together")

        print(f"\n📈 Performance Metrics:")
        print(f"   - {len(chunks)} chunks processed")
        print(f"   - {stats['total_tokens']} total tokens")
        print(f"   - {len(categories)} categories created")
        print(f"   - {len(chunks_from_db)} chunks stored")
        print(f"   - {processing_time:.1f}ms processing time")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up temporary database: {os.path.basename(db_path)}")


if __name__ == "__main__":
    asyncio.run(main())
