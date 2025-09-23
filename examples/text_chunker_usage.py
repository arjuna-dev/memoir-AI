"""
Text chunker usage example for MemoirAI library.
"""

import asyncio

from memoir_ai.exceptions import ValidationError
from memoir_ai.text_processing import TextChunk, TextChunker


async def main():
    """Demonstrate text chunker functionality."""

    print("📝 MemoirAI Text Chunker Example")
    print("=" * 40)

    # Example 1: Basic chunking
    print("\n1. Basic text chunking...")

    chunker = TextChunker(min_tokens=10, max_tokens=30, model_name="gpt-4o-mini")

    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    
    Machine learning (ML) is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science.
    
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
    """

    chunks = chunker.chunk_text(
        content=sample_text,
        source_id="ai_overview_doc",
        metadata={"topic": "AI/ML", "author": "example"},
    )

    print(f"   Original text length: {len(sample_text)} characters")
    print(f"   Number of chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        preview = (
            chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
        )
        print(f"   Chunk {i}: {chunk.token_count} tokens - {preview}")

    # Example 2: Chunking statistics
    print("\n2. Chunking statistics...")

    stats = chunker.get_chunking_stats(chunks)

    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"   Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"   Chunks below minimum: {stats['chunks_below_min']}")
    print(f"   Chunks above maximum: {stats['chunks_above_max']}")
    print(
        f"   Token distribution (25th/50th/75th percentile): "
        f"{stats['token_distribution']['p25']}/{stats['token_distribution']['p50']}/{stats['token_distribution']['p75']}"
    )

    # Example 3: Different chunking strategies
    print("\n3. Different chunking strategies...")

    test_text = "First sentence. Second sentence! Third sentence? Fourth sentence; Fifth sentence."

    # Strategy 1: Conservative chunking (larger chunks)
    conservative_chunker = TextChunker(
        min_tokens=15,
        max_tokens=50,
        delimiters=[".", "!", "?"],
        merge_small_chunks=True,
    )

    conservative_chunks = conservative_chunker.chunk_text(test_text)
    print(f"   Conservative chunking: {len(conservative_chunks)} chunks")
    for i, chunk in enumerate(conservative_chunks, 1):
        print(f"     Chunk {i}: {chunk.token_count} tokens - {chunk.content}")

    # Strategy 2: Aggressive chunking (smaller chunks)
    aggressive_chunker = TextChunker(
        min_tokens=3,
        max_tokens=8,
        delimiters=[".", "!", "?", ";"],
        merge_small_chunks=False,
    )

    aggressive_chunks = aggressive_chunker.chunk_text(test_text)
    print(f"   Aggressive chunking: {len(aggressive_chunks)} chunks")
    for i, chunk in enumerate(aggressive_chunks, 1):
        print(f"     Chunk {i}: {chunk.token_count} tokens - {chunk.content}")

    # Example 4: Paragraph preservation
    print("\n4. Paragraph preservation...")

    paragraph_text = """This is the first paragraph. It contains multiple sentences that discuss one topic.
    
    This is the second paragraph. It discusses a different topic and should be kept separate from the first paragraph.
    
    This is the third paragraph. It concludes the discussion with final thoughts."""

    paragraph_chunker = TextChunker(
        min_tokens=5, max_tokens=25, preserve_paragraphs=True
    )

    paragraph_chunks = paragraph_chunker.chunk_text(paragraph_text)
    print(f"   Paragraph-aware chunking: {len(paragraph_chunks)} chunks")
    for i, chunk in enumerate(paragraph_chunks, 1):
        preview = chunk.content.replace("\n", " ").strip()[:60] + "..."
        print(f"     Chunk {i}: {chunk.token_count} tokens - {preview}")

    # Example 5: Handling edge cases
    print("\n5. Edge cases...")

    # Very short text
    short_text = "Short."
    short_chunks = chunker.chunk_text(short_text)
    print(f"   Very short text: {len(short_chunks)} chunk(s)")
    print(
        f"     Content: '{short_chunks[0].content}' ({short_chunks[0].token_count} tokens)"
    )

    # Very long single sentence
    long_sentence = "This is an extremely long sentence that contains many words and should definitely exceed the maximum token limit for a single chunk, forcing the chunker to split it into multiple smaller pieces while trying to maintain some semblance of readability and coherence."

    long_chunker = TextChunker(min_tokens=5, max_tokens=15)
    long_chunks = long_chunker.chunk_text(long_sentence)
    print(f"   Long sentence: {len(long_chunks)} chunk(s)")
    for i, chunk in enumerate(long_chunks, 1):
        preview = (
            chunk.content[:40] + "..." if len(chunk.content) > 40 else chunk.content
        )
        print(f"     Chunk {i}: {chunk.token_count} tokens - {preview}")

    # Example 6: Different models
    print("\n6. Different model token counting...")

    test_sentence = (
        "This is a test sentence for comparing token counts across different models."
    )

    models = ["gpt-4o-mini", "gpt-4o-mini", "claude-3-sonnet"]

    for model in models:
        try:
            model_chunker = TextChunker(model_name=model, min_tokens=5, max_tokens=20)
            token_count = model_chunker.count_tokens(test_sentence)
            print(f"   {model}: {token_count} tokens")
        except Exception as e:
            print(f"   {model}: Error - {e}")

    # Example 7: Special characters and unicode
    print("\n7. Special characters and unicode...")

    unicode_text = "This text contains émojis 🚀🎉, special characters (àáâãäå), and numbers: 123,456.78!"

    unicode_chunker = TextChunker(min_tokens=5, max_tokens=15)
    unicode_chunks = unicode_chunker.chunk_text(unicode_text)

    print(f"   Unicode text chunking: {len(unicode_chunks)} chunk(s)")
    for i, chunk in enumerate(unicode_chunks, 1):
        print(f"     Chunk {i}: {chunk.token_count} tokens - {chunk.content}")

    # Example 8: Error handling
    print("\n8. Error handling...")

    try:
        # Invalid chunker configuration
        invalid_chunker = TextChunker(min_tokens=100, max_tokens=50)
        print("   ❌ This should have failed!")
    except ValidationError as e:
        print(f"   ✅ Configuration validation: {str(e)[:80]}...")

    try:
        # Empty content
        chunker.chunk_text("")
        print("   ❌ This should have failed!")
    except ValidationError as e:
        print(f"   ✅ Empty content validation: {str(e)[:80]}...")

    # Example 9: Integration with database models
    print("\n9. Integration with database storage...")

    # Simulate storing chunks (would integrate with database in real usage)
    integration_text = "This demonstrates how chunks would be stored in the database with proper metadata and source tracking."

    integration_chunks = chunker.chunk_text(
        content=integration_text,
        source_id="integration_example",
        metadata={
            "document_type": "example",
            "created_by": "chunker_demo",
            "processing_date": "2024-01-01",
        },
    )

    print(f"   Created {len(integration_chunks)} chunk(s) ready for database storage:")
    for chunk in integration_chunks:
        print(f"     Source: {chunk.source_id}")
        print(f"     Metadata: {chunk.metadata}")
        print(f"     Position: {chunk.start_position}-{chunk.end_position}")
        print(f"     Content: {chunk.content[:50]}...")
        print()

    print("🎉 Text chunker demonstration complete!")
    print("\n📋 Key Features Demonstrated:")
    print("   ✅ Token-based chunking with liteLLM integration")
    print("   ✅ Configurable size constraints and delimiters")
    print("   ✅ Paragraph boundary preservation")
    print("   ✅ Chunk merging and splitting strategies")
    print("   ✅ Comprehensive statistics and analytics")
    print("   ✅ Edge case handling and validation")
    print("   ✅ Multi-model token counting support")
    print("   ✅ Unicode and special character support")
    print("   ✅ Database integration readiness")


if __name__ == "__main__":
    asyncio.run(main())
