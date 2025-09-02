#!/usr/bin/env python3
"""
Vector Database Demo

This example demonstrates the vector database functionality of Ragify,
showing how to use ChromaDB, Pinecone, Weaviate, and FAISS for
storing and searching context embeddings.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from ragify.storage.vector_db import VectorDatabase
from ragify.models import ContextChunk, ContextSource, SourceType, RelevanceScore
from ragify.exceptions import VectorDBError


def generate_sample_embeddings(texts: List[str], dimension: int = 384) -> List[List[float]]:
    """
    Generate sample embeddings for demonstration purposes.
    In a real application, you would use a proper embedding model.
    """
    import numpy as np
    
    embeddings = []
    for i, text in enumerate(texts):
        # Create a deterministic embedding based on text content
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, dimension).tolist()
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        embedding = [x / norm for x in embedding]
        embeddings.append(embedding)
    
    return embeddings


async def demonstrate_chroma_db():
    """Demonstrate ChromaDB functionality."""
    print("\n🔵 ChromaDB Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_path = os.path.join(temp_dir, "chroma_db")
        
        try:
            # Initialize ChromaDB
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            print(f"✅ Connected to ChromaDB at: {chroma_path}")
            
            # Create sample data
            chunks = [
                ContextChunk(
                    content="Machine learning is a subset of artificial intelligence",
                    source=ContextSource(name="ML Guide", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.9),
                    created_at=datetime.utcnow(),
                    token_count=12
                ),
                ContextChunk(
                    content="Deep learning uses neural networks with multiple layers",
                    source=ContextSource(name="DL Book", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.8),
                    created_at=datetime.utcnow(),
                    token_count=10
                ),
                ContextChunk(
                    content="Natural language processing helps computers understand text",
                    source=ContextSource(name="NLP Paper", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.7),
                    created_at=datetime.utcnow(),
                    token_count=11
                )
            ]
            
            texts = [chunk.content for chunk in chunks]
            embeddings = generate_sample_embeddings(texts)
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(chunks, embeddings)
            print(f"✅ Stored {len(vector_ids)} embeddings")
            
            # Search for similar vectors
            query_text = "What is machine learning?"
            query_embedding = generate_sample_embeddings([query_text])[0]
            
            results = await vector_db.search_similar(query_embedding, top_k=2)
            print(f"✅ Found {len(results)} similar vectors")
            
            for i, (vector_id, score, metadata) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f}, Source: {metadata.get('source_name', 'Unknown')}")
            
            # Test with filters
            filtered_results = await vector_db.search_similar(
                query_embedding,
                top_k=2,
                filters={"source_type": "document"}
            )
            print(f"✅ Found {len(filtered_results)} filtered results")
            
            # Get statistics
            stats = await vector_db.get_stats()
            print(f"📊 Database stats: {stats['total_vectors']} vectors, {stats['searches_performed']} searches")
            
            await vector_db.close()
            print("✅ ChromaDB demo completed")
            
        except Exception as e:
            print(f"❌ ChromaDB demo failed: {e}")


async def demonstrate_faiss_db():
    """Demonstrate FAISS functionality."""
    print("\n🟡 FAISS Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        faiss_path = os.path.join(temp_dir, "faiss_index")
        
        try:
            # Initialize FAISS
            vector_db = VectorDatabase(f"faiss://{faiss_path}")
            await vector_db.connect()
            print(f"✅ Connected to FAISS at: {faiss_path}")
            
            # Create sample data
            chunks = [
                ContextChunk(
                    content="Python is a high-level programming language",
                    source=ContextSource(name="Python Doc", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.9),
                    created_at=datetime.utcnow(),
                    token_count=8
                ),
                ContextChunk(
                    content="JavaScript is used for web development",
                    source=ContextSource(name="JS Guide", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.8),
                    created_at=datetime.utcnow(),
                    token_count=7
                ),
                ContextChunk(
                    content="Java is an object-oriented programming language",
                    source=ContextSource(name="Java Book", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.7),
                    created_at=datetime.utcnow(),
                    token_count=9
                )
            ]
            
            texts = [chunk.content for chunk in chunks]
            embeddings = generate_sample_embeddings(texts)
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(chunks, embeddings)
            print(f"✅ Stored {len(vector_ids)} embeddings")
            
            # Search for similar vectors
            query_text = "What programming language should I learn?"
            query_embedding = generate_sample_embeddings([query_text])[0]
            
            results = await vector_db.search_similar(query_embedding, top_k=2)
            print(f"✅ Found {len(results)} similar vectors")
            
            for i, (vector_id, score, metadata) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f}, Source: {metadata.get('source_name', 'Unknown')}")
            
            # Create index with custom parameters
            await vector_db.create_index({
                "index_type": "ivf",
                "nlist": 50,
                "dimension": 384
            })
            print("✅ Created custom FAISS index")
            
            # Get statistics
            stats = await vector_db.get_stats()
            print(f"📊 Database stats: {stats['total_vectors']} vectors")
            
            await vector_db.close()
            print("✅ FAISS demo completed")
            
        except Exception as e:
            print(f"❌ FAISS demo failed: {e}")


async def demonstrate_pinecone_db():
    """Demonstrate Pinecone functionality with real API integration."""
    print("\n🟢 Pinecone Demo")
    print("=" * 40)
    
    try:
        # Note: This demo requires actual Pinecone credentials
        # For demonstration, we'll show the setup process
        
        print("📝 Pinecone Setup Instructions:")
        print("1. Sign up at https://www.pinecone.io/")
        print("2. Get your API key from the console")
        print("3. Create an index with dimension 384")
        print("4. Set environment variables:")
        print("   export PINECONE_API_KEY='your_api_key'")
        print("   export PINECONE_INDEX_NAME='your_index_name'")
        
        # Check if credentials are available
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if api_key and index_name:
            print(f"✅ Found Pinecone credentials")
            
            # Initialize Pinecone
            vector_db = VectorDatabase(f"pinecone://{api_key}:{index_name}")
            await vector_db.connect()
            print(f"✅ Connected to Pinecone index: {index_name}")
            
            # Create sample data
            chunks = [
                ContextChunk(
                    content="Cloud computing provides scalable infrastructure",
                    source=ContextSource(name="Cloud Guide", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.9),
                    created_at=datetime.utcnow(),
                    token_count=8
                ),
                ContextChunk(
                    content="AWS offers various cloud services",
                    source=ContextSource(name="AWS Doc", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.8),
                    created_at=datetime.utcnow(),
                    token_count=6
                )
            ]
            
            texts = [chunk.content for chunk in chunks]
            embeddings = generate_sample_embeddings(texts)
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(chunks, embeddings)
            print(f"✅ Stored {len(vector_ids)} embeddings")
            
            # Search for similar vectors
            query_text = "What is cloud computing?"
            query_embedding = generate_sample_embeddings([query_text])[0]
            
            results = await vector_db.search_similar(query_embedding, top_k=2)
            print(f"✅ Found {len(results)} similar vectors")
            
            for i, (vector_id, score, metadata) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.3f}, Source: {metadata.get('source_name', 'Unknown')}")
            
            await vector_db.close()
            print("✅ Pinecone demo completed")
            
        else:
            print("⚠️  Pinecone credentials not found. Skipping live demo.")
            print("   Set PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables to test.")
            
    except Exception as e:
        print(f"❌ Pinecone demo failed: {e}")


async def demonstrate_weaviate_db():
    """Demonstrate Weaviate functionality with real API integration."""
    print("\n🟣 Weaviate Demo")
    print("=" * 40)
    
    try:
        # Note: This demo requires actual Weaviate setup
        # For demonstration, we'll show the setup process
        
        print("📝 Weaviate Setup Instructions:")
        print("1. Install Weaviate: docker run -d -p 8080:8080 semitechnologies/weaviate:latest")
        print("2. Or use Weaviate Cloud Services")
        print("3. Set environment variables:")
        print("   export WEAVIATE_API_KEY='your_api_key' (if using WCS)")
        print("   export OPENAI_API_KEY='your_openai_key' (optional)")
        
        # Check if Weaviate is available
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        
        print(f"🔗 Attempting to connect to Weaviate at: {weaviate_url}")
        
        # Initialize Weaviate
        vector_db = VectorDatabase(f"weaviate://{weaviate_url.replace('http://', '')}")
        await vector_db.connect()
        print(f"✅ Connected to Weaviate")
        
        # Create sample data
        chunks = [
            ContextChunk(
                content="GraphQL is a query language for APIs",
                source=ContextSource(name="GraphQL Doc", source_type=SourceType.DOCUMENT),
                relevance_score=RelevanceScore(score=0.9),
                created_at=datetime.utcnow(),
                token_count=9
            ),
            ContextChunk(
                content="REST APIs use HTTP methods for communication",
                source=ContextSource(name="REST Guide", source_type=SourceType.DOCUMENT),
                relevance_score=RelevanceScore(score=0.8),
                created_at=datetime.utcnow(),
                token_count=8
            )
        ]
        
        texts = [chunk.content for chunk in chunks]
        embeddings = generate_sample_embeddings(texts)
        
        # Store embeddings
        vector_ids = await vector_db.store_embeddings(chunks, embeddings)
        print(f"✅ Stored {len(vector_ids)} embeddings")
        
        # Search for similar vectors
        query_text = "What is an API?"
        query_embedding = generate_sample_embeddings([query_text])[0]
        
        results = await vector_db.search_similar(query_embedding, top_k=2)
        print(f"✅ Found {len(results)} similar vectors")
        
        for i, (vector_id, score, metadata) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.3f}, Source: {metadata.get('source_name', 'Unknown')}")
        
        await vector_db.close()
        print("✅ Weaviate demo completed")
        
    except Exception as e:
        print(f"❌ Weaviate demo failed: {e}")
        print("   Make sure Weaviate is running or check your connection settings.")


async def demonstrate_comparison():
    """Demonstrate comparison between different vector databases."""
    print("\n📊 Vector Database Comparison")
    print("=" * 50)
    
    # Sample data for comparison
    chunks = [
        ContextChunk(
            content="Vector databases store high-dimensional data efficiently",
            source=ContextSource(name="Vector DB Guide", source_type=SourceType.DOCUMENT),
            relevance_score=RelevanceScore(score=0.9),
            created_at=datetime.utcnow(),
            token_count=10
        ),
        ContextChunk(
            content="Similarity search finds the most relevant vectors",
            source=ContextSource(name="Search Doc", source_type=SourceType.DOCUMENT),
            relevance_score=RelevanceScore(score=0.8),
            created_at=datetime.utcnow(),
            token_count=8
        )
    ]
    
    texts = [chunk.content for chunk in chunks]
    embeddings = generate_sample_embeddings(texts)
    query_embedding = generate_sample_embeddings(["What are vector databases?"])[0]
    
    databases = [
        ("ChromaDB", "chroma"),
        ("FAISS", "faiss"),
    ]
    
    results_comparison = {}
    
    for db_name, db_type in databases:
        print(f"\n🔍 Testing {db_name}...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if db_type == "chroma":
                    db_path = os.path.join(temp_dir, "chroma_db")
                    vector_db = VectorDatabase(f"chroma://{db_path}")
                elif db_type == "faiss":
                    db_path = os.path.join(temp_dir, "faiss_index")
                    vector_db = VectorDatabase(f"faiss://{db_path}")
                
                await vector_db.connect()
                
                # Store embeddings
                start_time = asyncio.get_event_loop().time()
                vector_ids = await vector_db.store_embeddings(chunks, embeddings)
                store_time = asyncio.get_event_loop().time() - start_time
                
                # Search embeddings
                start_time = asyncio.get_event_loop().time()
                results = await vector_db.search_similar(query_embedding, top_k=2)
                search_time = asyncio.get_event_loop().time() - start_time
                
                # Get statistics
                stats = await vector_db.get_stats()
                
                results_comparison[db_name] = {
                    "store_time": store_time,
                    "search_time": search_time,
                    "results_count": len(results),
                    "total_vectors": stats["total_vectors"],
                    "avg_search_time": stats["avg_search_time"]
                }
                
                await vector_db.close()
                
                print(f"  ✅ Store time: {store_time:.4f}s")
                print(f"  ✅ Search time: {search_time:.4f}s")
                print(f"  ✅ Results: {len(results)}")
                
        except Exception as e:
            print(f"  ❌ {db_name} failed: {e}")
            results_comparison[db_name] = {"error": str(e)}
    
    # Print comparison summary
    print(f"\n📈 Performance Comparison Summary:")
    print("-" * 40)
    
    for db_name, results in results_comparison.items():
        if "error" not in results:
            print(f"{db_name:12} | Store: {results['store_time']:.4f}s | Search: {results['search_time']:.4f}s | Results: {results['results_count']}")
        else:
            print(f"{db_name:12} | Error: {results['error']}")


async def demonstrate_advanced_features():
    """Demonstrate advanced vector database features."""
    print("\n🚀 Advanced Features Demo")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_path = os.path.join(temp_dir, "chroma_db")
        
        try:
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Create diverse sample data
            chunks = [
                ContextChunk(
                    content="Python is great for data science",
                    source=ContextSource(name="Python Doc", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.9),
                    created_at=datetime.utcnow(),
                    token_count=7
                ),
                ContextChunk(
                    content="JavaScript is perfect for web development",
                    source=ContextSource(name="JS Guide", source_type=SourceType.API),
                    relevance_score=RelevanceScore(score=0.8),
                    created_at=datetime.utcnow(),
                    token_count=8
                ),
                ContextChunk(
                    content="Java is excellent for enterprise applications",
                    source=ContextSource(name="Java Book", source_type=SourceType.DATABASE),
                    relevance_score=RelevanceScore(score=0.7),
                    created_at=datetime.utcnow(),
                    token_count=9
                )
            ]
            
            texts = [chunk.content for chunk in chunks]
            embeddings = generate_sample_embeddings(texts)
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(chunks, embeddings)
            print(f"✅ Stored {len(vector_ids)} embeddings")
            
            # Test different search scenarios
            scenarios = [
                ("General search", "What programming language should I learn?", {}),
                ("Document filter", "What programming language should I learn?", {"source_type": "document"}),
                ("API filter", "What programming language should I learn?", {"source_type": "api"}),
                ("High threshold", "What programming language should I learn?", {}, 0.8),
                ("Low threshold", "What programming language should I learn?", {}, 0.1),
            ]
            
            for scenario_name, query_text, filters, *threshold in scenarios:
                min_score = threshold[0] if threshold else 0.0
                query_embedding = generate_sample_embeddings([query_text])[0]
                
                results = await vector_db.search_similar(
                    query_embedding,
                    top_k=3,
                    min_score=min_score,
                    filters=filters
                )
                
                print(f"\n🔍 {scenario_name}:")
                print(f"  Query: {query_text}")
                print(f"  Filters: {filters}")
                print(f"  Min score: {min_score}")
                print(f"  Results: {len(results)}")
                
                for i, (vector_id, score, metadata) in enumerate(results, 1):
                    print(f"    {i}. Score: {score:.3f}, Source: {metadata.get('source_name', 'Unknown')} ({metadata.get('source_type', 'Unknown')})")
            
            # Test metadata retrieval
            print(f"\n📋 Metadata Retrieval:")
            for vector_id in vector_ids:
                metadata = await vector_db.get_metadata(vector_id)
                if metadata:
                    print(f"  {vector_id}: {metadata.get('source_name', 'Unknown')} ({metadata.get('source_type', 'Unknown')})")
            
            # Test statistics
            stats = await vector_db.get_stats()
            print(f"\n📊 Final Statistics:")
            print(f"  Total vectors: {stats['total_vectors']}")
            print(f"  Searches performed: {stats['searches_performed']}")
            print(f"  Average search time: {stats['avg_search_time']:.4f}s")
            
            await vector_db.close()
            print("✅ Advanced features demo completed")
            
        except Exception as e:
            print(f"❌ Advanced features demo failed: {e}")


async def main():
    """Main demonstration function."""
    print("🎯 Ragify Vector Database Demo")
    print("=" * 50)
    print("This demo showcases vector database functionality with")
    print("ChromaDB, Pinecone, Weaviate, and FAISS support.\n")
    
    try:
        # Demonstrate each vector database
        await demonstrate_chroma_db()
        await demonstrate_faiss_db()
        await demonstrate_pinecone_db()
        await demonstrate_weaviate_db()
        
        # Demonstrate comparison
        await demonstrate_comparison()
        
        # Demonstrate advanced features
        await demonstrate_advanced_features()
        
        print("\n🎉 Vector database demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Multi-database support (ChromaDB, Pinecone, Weaviate, FAISS)")
        print("   ✅ Embedding storage and retrieval")
        print("   ✅ Similarity search with filters")
        print("   ✅ Metadata management")
        print("   ✅ Performance comparison")
        print("   ✅ Statistics tracking")
        print("   ✅ Error handling")
        
        print("\n📚 Usage Examples:")
        print("   # ChromaDB (local)")
        print("   vector_db = VectorDatabase('chroma:///path/to/chroma')")
        print("   ")
        print("   # ChromaDB (remote)")
        print("   vector_db = VectorDatabase('chroma://localhost:8000')")
        print("   ")
        print("   # Pinecone (cloud)")
        print("   vector_db = VectorDatabase('pinecone://api_key:index_name')")
        print("   ")
        print("   # Weaviate (local/cloud)")
        print("   vector_db = VectorDatabase('weaviate://localhost:8080')")
        print("   ")
        print("   # FAISS (local)")
        print("   vector_db = VectorDatabase('faiss:///path/to/index')")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
