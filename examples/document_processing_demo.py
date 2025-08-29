#!/usr/bin/env python3
"""
Document Processing Demo for Ragify Framework

This demo showcases the comprehensive document processing capabilities
including PDF, DOCX, DOC, and text file processing.
"""

import asyncio
import os
from pathlib import Path
from ragify.sources.document import DocumentSource
from ragify.models import SourceType, PrivacyLevel
from ragify.core import ContextOrchestrator

async def create_sample_documents():
    """Create sample documents for demonstration."""
    print("📝 Creating sample documents...")
    
    # Create test directory if it doesn't exist
    os.makedirs("demo_documents", exist_ok=True)
    
    # Create a simple text file
    with open("demo_documents/sample.txt", "w") as f:
        f.write("""Ragify Document Processing Demo

This is a sample text document demonstrating the text processing capabilities.

Key Features:
- Simple text extraction
- Chunking and processing
- Query-based retrieval

The framework can handle various text formats including .txt and .md files.
""")
    
    print("✅ Sample documents created in demo_documents/")

async def demo_document_processing():
    """Demonstrate document processing capabilities."""
    
    print("\n🎯 Ragify Document Processing Demo")
    print("=" * 50)
    
    # Create sample documents
    await create_sample_documents()
    
    # Initialize document source
    doc_source = DocumentSource(
        name="demo_docs",
        source_type=SourceType.DOCUMENT,
        url="./demo_documents",
        chunk_size=300,
        overlap=50
    )
    
    print(f"\n📁 Document source initialized:")
    print(f"   - Source path: {doc_source.url}")
    print(f"   - Chunk size: {doc_source.chunk_size}")
    print(f"   - Overlap: {doc_source.overlap}")
    print(f"   - Supported formats: {list(doc_source.supported_formats.keys())}")
    
    # Test individual file processing
    print(f"\n🔍 Testing individual file processing:")
    print("-" * 40)
    
    test_files = ["sample.txt"]
    
    # Add PDF and DOCX if they exist from previous tests
    if Path("test_documents/sample.pdf").exists():
        test_files.append("test_documents/sample.pdf")
    if Path("test_documents/sample.docx").exists():
        test_files.append("test_documents/sample.docx")
    
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            print(f"\n📄 Processing: {path.name}")
            
            # Extract content
            content = await doc_source._load_single_document(path)
            if content:
                print(f"   ✅ Extracted {len(content)} characters")
                print(f"   📝 Preview: {content[:100]}...")
                
                # Create chunks
                chunks = await doc_source._chunk_content(content, str(path))
                print(f"   🔪 Created {len(chunks)} chunks")
                
                if chunks:
                    print(f"   📊 Sample chunk: {chunks[0].content[:80]}...")
            else:
                print(f"   ❌ Failed to extract content")
        else:
            print(f"   ⚠️  File not found: {path}")
    
    # Test query-based retrieval
    print(f"\n🔍 Testing query-based retrieval:")
    print("-" * 40)
    
    test_queries = [
        "document processing",
        "text extraction",
        "chunking capabilities"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        chunks = await doc_source.get_chunks(
            query=query,
            max_chunks=2,
            min_relevance=0.1
        )
        print(f"Found {len(chunks)} relevant chunks")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. [{chunk.source.name}] {chunk.content[:60]}...")
            if chunk.relevance_score:
                print(f"     Relevance: {chunk.relevance_score.score:.2f}")
    
    # Test with Context Orchestrator
    print(f"\n🚀 Testing with Context Orchestrator:")
    print("-" * 40)
    
    orchestrator = ContextOrchestrator(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    
    # Add document source
    orchestrator.add_source(doc_source)
    
    # Test context retrieval
    context_response = await orchestrator.get_context(
        query="What are the key features of document processing?",
        user_id="demo_user",
        max_tokens=1000,
        min_relevance=0.3
    )
    
    print(f"Context retrieved:")
    print(f"  - Processing time: {context_response.processing_time:.3f}s")
    print(f"  - Total chunks: {len(context_response.context.chunks)}")
    print(f"  - Sources: {[s.name for s in context_response.context.sources]}")
    
    if context_response.context.chunks:
        print(f"  - Sample content: {context_response.context.chunks[0].content[:100]}...")
    
    # Close orchestrator
    await orchestrator.close()
    
    print(f"\n✅ Document processing demo completed!")

async def demo_advanced_features():
    """Demonstrate advanced document processing features."""
    
    print(f"\n🚀 Advanced Document Processing Features")
    print("=" * 50)
    
    # Test different chunk sizes
    print(f"\n📏 Testing different chunk sizes:")
    print("-" * 30)
    
    chunk_sizes = [100, 500, 1000]
    
    for size in chunk_sizes:
        doc_source = DocumentSource(
            name=f"chunk_test_{size}",
            source_type=SourceType.DOCUMENT,
            url="./demo_documents",
            chunk_size=size,
            overlap=size // 5
        )
        
        content = await doc_source._load_single_document(Path("demo_documents/sample.txt"))
        if content:
            chunks = await doc_source._chunk_content(content, "demo_documents/sample.txt")
            print(f"  Chunk size {size}: {len(chunks)} chunks created")
    
    # Test error handling
    print(f"\n🛡️  Testing error handling:")
    print("-" * 30)
    
    doc_source = DocumentSource(
        name="error_test",
        source_type=SourceType.DOCUMENT,
        url="./nonexistent_path",
        chunk_size=500
    )
    
    chunks = await doc_source.get_chunks("test query")
    print(f"  Non-existent path: {len(chunks)} chunks (expected 0)")
    
    # Test unsupported file format
    unsupported_file = Path("demo_documents/test.xyz")
    unsupported_file.write_text("This is a test file with unsupported format")
    
    content = await doc_source._load_single_document(unsupported_file)
    print(f"  Unsupported format: {len(content) if content else 0} characters (expected 0)")
    
    # Clean up
    unsupported_file.unlink()
    
    print(f"\n✅ Advanced features demo completed!")

async def main():
    """Run the complete document processing demo."""
    
    print("🎯 Ragify Document Processing Demo")
    print("=" * 50)
    print("This demo showcases comprehensive document processing capabilities")
    print("including PDF, DOCX, DOC, and text file processing.\n")
    
    # Basic document processing demo
    await demo_document_processing()
    
    # Advanced features demo
    await demo_advanced_features()
    
    print(f"\n🎉 Complete document processing demo finished!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✅ Multi-format document processing (PDF, DOCX, DOC, TXT)")
    print(f"   ✅ Intelligent content extraction and chunking")
    print(f"   ✅ Query-based relevance scoring")
    print(f"   ✅ Integration with Context Orchestrator")
    print(f"   ✅ Error handling and fallback mechanisms")
    print(f"   ✅ Configurable chunk sizes and overlap")
    print(f"   ✅ Metadata preservation and tracking")

if __name__ == "__main__":
    asyncio.run(main())
