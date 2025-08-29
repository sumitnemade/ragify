# 💡 **Examples & Tutorials**

This document provides comprehensive examples and tutorials for using Ragify in various scenarios.

## 🚀 **Quick Start Examples**

### **1. Basic Context Retrieval**

```python
import asyncio
from ragify import ContextOrchestrator, ContextRequest, OrchestratorConfig
from ragify.sources import DocumentSource
from ragify.models import SourceConfig, SourceType, PrivacyLevel

async def basic_example():
    """Basic example of context retrieval."""
    
    # 1. Create configuration
    config = OrchestratorConfig(
        fusion_strategy="highest_relevance",
        vector_db_type="chromadb",
        vector_db_url="memory://"
    )
    
    # 2. Initialize orchestrator
    orchestrator = ContextOrchestrator(config)
    
    # 3. Add document source
    doc_config = SourceConfig(
        id="docs",
        name="Documentation",
        source_type=SourceType.DOCUMENT,
        privacy_level=PrivacyLevel.PUBLIC
    )
    
    doc_source = DocumentSource(
        config=doc_config,
        document_paths=["docs/api.md", "docs/guide.pdf"]
    )
    
    await orchestrator.add_source(doc_source)
    
    # 4. Create request
    request = ContextRequest(
        query="How to use the API?",
        max_chunks=5,
        min_relevance=0.7
    )
    
    # 5. Get context
    response = await orchestrator.get_context(request)
    
    # 6. Process results
    print(f"Found {len(response.chunks)} relevant chunks")
    for i, chunk in enumerate(response.chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Relevance: {chunk.relevance_score.score:.3f}")
        print(f"  Source: {chunk.source.name}")

# Run the example
asyncio.run(basic_example())
```

### **2. Multi-Source Context Fusion**

```python
import asyncio
from ragify import ContextOrchestrator, ContextRequest, OrchestratorConfig
from ragify.sources import DocumentSource, APISource, DatabaseSource
from ragify.models import SourceConfig, SourceType, PrivacyLevel

async def multi_source_example():
    """Example with multiple data sources."""
    
    # 1. Initialize orchestrator
    config = OrchestratorConfig(
        fusion_strategy="consensus",
        enable_semantic_analysis=True
    )
    orchestrator = ContextOrchestrator(config)
    
    # 2. Add document source
    doc_source = DocumentSource(
        config=SourceConfig(
            id="docs",
            name="Documentation",
            source_type=SourceType.DOCUMENT,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        document_paths=["docs/"]
    )
    
    # 3. Add API source
    api_source = APISource(
        config=SourceConfig(
            id="api",
            name="External API",
            source_type=SourceType.API,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        base_url="https://api.example.com",
        auth_type="bearer",
        auth_config={"token": "your_token"}
    )
    
    # 4. Add database source
    db_source = DatabaseSource(
        config=SourceConfig(
            id="database",
            name="Knowledge Base",
            source_type=SourceType.DATABASE,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        connection_string="postgresql://user:pass@localhost/kb",
        db_type="postgresql"
    )
    
    # 5. Add all sources
    await orchestrator.add_source(doc_source, priority=3)
    await orchestrator.add_source(api_source, priority=2)
    await orchestrator.add_source(db_source, priority=1)
    
    # 6. Get context from all sources
    request = ContextRequest(
        query="machine learning best practices",
        max_chunks=10,
        min_relevance=0.6
    )
    
    response = await orchestrator.get_context(request)
    
    # 7. Analyze results
    print(f"Total chunks: {len(response.chunks)}")
    print(f"Sources used: {response.sources}")
    print(f"Processing time: {response.processing_time:.3f}s")
    
    if response.fusion_metadata:
        print(f"Conflicts resolved: {len(response.fusion_metadata.conflicts)}")

asyncio.run(multi_source_example())
```

## 🤖 **Chatbot Integration Examples**

### **1. Simple Chatbot**

```python
import asyncio
from ragify import ContextOrchestrator, ContextRequest
from ragify.sources import DocumentSource

class SimpleChatbot:
    def __init__(self):
        self.orchestrator = ContextOrchestrator()
        self.conversation_history = []
    
    async def setup_sources(self):
        """Setup data sources for the chatbot."""
        # Add documentation source
        doc_source = DocumentSource(
            config=SourceConfig(
                id="faq",
                name="FAQ Database",
                source_type=SourceType.DOCUMENT,
                privacy_level=PrivacyLevel.PUBLIC
            ),
            document_paths=["data/faq.md", "data/knowledge_base.pdf"]
        )
        await self.orchestrator.add_source(doc_source)
    
    async def get_response(self, user_message: str) -> str:
        """Get chatbot response based on user message."""
        # 1. Create context request
        request = ContextRequest(
            query=user_message,
            max_chunks=3,
            min_relevance=0.7
        )
        
        # 2. Get relevant context
        response = await self.orchestrator.get_context(request)
        
        # 3. Generate response
        if response.chunks:
            # Use the most relevant chunk
            best_chunk = max(response.chunks, key=lambda c: c.relevance_score.score)
            return f"Based on our knowledge base: {best_chunk.content}"
        else:
            return "I don't have enough information to answer that question."
    
    async def chat(self):
        """Interactive chat session."""
        await self.setup_sources()
        
        print("Chatbot: Hello! How can I help you today?")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            
            response = await self.get_response(user_input)
            print(f"Chatbot: {response}")

# Run chatbot
# asyncio.run(SimpleChatbot().chat())
```

### **2. Advanced Chatbot with Confidence**

```python
import asyncio
from typing import List, Dict, Any

class AdvancedChatbot:
    def __init__(self):
        self.orchestrator = ContextOrchestrator()
        self.user_context = {}
    
    async def get_confidence_based_response(
        self, 
        user_message: str, 
        user_id: str = None
    ) -> Dict[str, Any]:
        """Get response with confidence scoring."""
        
        # Update user context
        if user_id:
            self.user_context = {
                'user_id': user_id,
                'conversation_history': self.get_conversation_history(user_id)
            }
        
        # Create request
        request = ContextRequest(
            query=user_message,
            max_chunks=5,
            min_relevance=0.5,
            user_id=user_id
        )
        
        # Get context
        response = await self.orchestrator.get_context(
            request, 
            user_context=self.user_context
        )
        
        # Analyze confidence
        if response.chunks:
            avg_confidence = sum(
                chunk.relevance_score.score for chunk in response.chunks
            ) / len(response.chunks)
            
            if avg_confidence > 0.8:
                confidence_level = "high"
                response_text = self.generate_confident_response(response.chunks)
            elif avg_confidence > 0.6:
                confidence_level = "medium"
                response_text = self.generate_cautious_response(response.chunks)
            else:
                confidence_level = "low"
                response_text = self.generate_uncertain_response(response.chunks)
        else:
            confidence_level = "none"
            response_text = "I don't have enough information to answer that question."
        
        return {
            'response': response_text,
            'confidence_level': confidence_level,
            'confidence_score': avg_confidence if response.chunks else 0.0,
            'sources_used': response.sources,
            'chunks_count': len(response.chunks)
        }
    
    def generate_confident_response(self, chunks: List[ContextChunk]) -> str:
        """Generate confident response from high-quality chunks."""
        best_chunk = max(chunks, key=lambda c: c.relevance_score.score)
        return f"I'm confident that: {best_chunk.content}"
    
    def generate_cautious_response(self, chunks: List[ContextChunk]) -> str:
        """Generate cautious response from medium-quality chunks."""
        return f"Based on available information: {' '.join(c.content for c in chunks[:2])}"
    
    def generate_uncertain_response(self, chunks: List[ContextChunk]) -> str:
        """Generate uncertain response from low-quality chunks."""
        return f"I found some related information, but I'm not entirely sure: {chunks[0].content}"

# Usage
async def test_advanced_chatbot():
    chatbot = AdvancedChatbot()
    await chatbot.setup_sources()
    
    response = await chatbot.get_confidence_based_response(
        "What are the best practices for machine learning?",
        user_id="user123"
    )
    
    print(f"Response: {response['response']}")
    print(f"Confidence: {response['confidence_level']} ({response['confidence_score']:.3f})")
    print(f"Sources: {response['sources_used']}")

# asyncio.run(test_advanced_chatbot())
```

## 📊 **Data Source Examples**

### **1. Document Processing**

```python
import asyncio
from pathlib import Path
from ragify.sources import DocumentSource

async def document_processing_example():
    """Example of processing various document types."""
    
    # Create document source with multiple formats
    doc_source = DocumentSource(
        config=SourceConfig(
            id="mixed_docs",
            name="Mixed Documents",
            source_type=SourceType.DOCUMENT,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        document_paths=[
            "docs/api_reference.md",
            "docs/user_guide.pdf",
            "docs/technical_spec.docx",
            "docs/readme.txt"
        ],
        supported_formats=["md", "pdf", "docx", "txt"]
    )
    
    # Test different queries
    queries = [
        "API authentication",
        "installation guide",
        "configuration options",
        "troubleshooting"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        chunks = await doc_source.get_chunks(query, max_chunks=3)
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk.content[:80]}... (Score: {chunk.relevance_score.score:.3f})")

# asyncio.run(document_processing_example())
```

### **2. API Integration**

```python
import asyncio
from ragify.sources import APISource

async def api_integration_example():
    """Example of integrating with external APIs."""
    
    # GitHub API integration
    github_api = APISource(
        config=SourceConfig(
            id="github",
            name="GitHub API",
            source_type=SourceType.API,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        base_url="https://api.github.com",
        auth_type="bearer",
        auth_config={"token": "your_github_token"},
        rate_limit=5000  # GitHub's rate limit
    )
    
    # Stack Overflow API integration
    stackoverflow_api = APISource(
        config=SourceConfig(
            id="stackoverflow",
            name="Stack Overflow API",
            source_type=SourceType.API,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        base_url="https://api.stackexchange.com/2.3",
        auth_type="none",
        rate_limit=100
    )
    
    # Test queries
    queries = [
        "python async programming",
        "machine learning algorithms",
        "web development best practices"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        
        # Search GitHub
        github_chunks = await github_api.get_chunks(query, max_chunks=2)
        print(f"  GitHub results: {len(github_chunks)}")
        
        # Search Stack Overflow
        so_chunks = await stackoverflow_api.get_chunks(query, max_chunks=2)
        print(f"  Stack Overflow results: {len(so_chunks)}")

# asyncio.run(api_integration_example())
```

### **3. Database Integration**

```python
import asyncio
from ragify.sources import DatabaseSource

async def database_integration_example():
    """Example of integrating with databases."""
    
    # PostgreSQL integration
    postgres_source = DatabaseSource(
        config=SourceConfig(
            id="postgres_kb",
            name="PostgreSQL Knowledge Base",
            source_type=SourceType.DATABASE,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        connection_string="postgresql://user:pass@localhost/knowledge_base",
        db_type="postgresql",
        query_template="""
            SELECT content, title, category 
            FROM articles 
            WHERE content ILIKE %s 
            ORDER BY relevance_score DESC 
            LIMIT %s
        """
    )
    
    # SQLite integration for local data
    sqlite_source = DatabaseSource(
        config=SourceConfig(
            id="sqlite_local",
            name="SQLite Local Data",
            source_type=SourceType.DATABASE,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        connection_string="sqlite:///local_data.db",
        db_type="sqlite",
        query_template="""
            SELECT content, tags 
            FROM documents 
            WHERE content LIKE ? 
            LIMIT ?
        """
    )
    
    # Test queries
    test_queries = [
        "machine learning",
        "data analysis",
        "software architecture"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        try:
            # Query PostgreSQL
            pg_chunks = await postgres_source.get_chunks(query, max_chunks=2)
            print(f"  PostgreSQL: {len(pg_chunks)} results")
            
            # Query SQLite
            sqlite_chunks = await sqlite_source.get_chunks(query, max_chunks=2)
            print(f"  SQLite: {len(sqlite_chunks)} results")
            
        except Exception as e:
            print(f"  Error: {e}")

# asyncio.run(database_integration_example())
```

### **4. Real-time Data Sources**

```python
import asyncio
from ragify.sources import RealtimeSource

async def realtime_source_example():
    """Example of real-time data sources."""
    
    # WebSocket real-time source
    websocket_source = RealtimeSource(
        config=SourceConfig(
            id="websocket_feed",
            name="WebSocket Feed",
            source_type=SourceType.REALTIME,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        protocol="websocket",
        endpoint="wss://echo.websocket.org",
        update_interval=1.0
    )
    
    # MQTT real-time source
    mqtt_source = RealtimeSource(
        config=SourceConfig(
            id="mqtt_sensor",
            name="MQTT Sensor Data",
            source_type=SourceType.REALTIME,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        protocol="mqtt",
        endpoint="mqtt://broker.hivemq.com:1883",
        update_interval=2.0
    )
    
    # Redis Pub/Sub source
    redis_source = RealtimeSource(
        config=SourceConfig(
            id="redis_events",
            name="Redis Events",
            source_type=SourceType.REALTIME,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        protocol="redis",
        endpoint="redis://localhost:6379",
        update_interval=0.5
    )
    
    # Test real-time queries
    async def test_realtime_queries():
        query = "latest updates"
        
        # Test WebSocket
        try:
            ws_chunks = await websocket_source.get_chunks(query, max_chunks=3)
            print(f"WebSocket: {len(ws_chunks)} real-time chunks")
        except Exception as e:
            print(f"WebSocket error: {e}")
        
        # Test MQTT
        try:
            mqtt_chunks = await mqtt_source.get_chunks(query, max_chunks=3)
            print(f"MQTT: {len(mqtt_chunks)} real-time chunks")
        except Exception as e:
            print(f"MQTT error: {e}")
        
        # Test Redis
        try:
            redis_chunks = await redis_source.get_chunks(query, max_chunks=3)
            print(f"Redis: {len(redis_chunks)} real-time chunks")
        except Exception as e:
            print(f"Redis error: {e}")
    
    await test_realtime_queries()

# asyncio.run(realtime_source_example())
```

## 🎯 **Scoring and Fusion Examples**

### **1. Custom Scoring Weights**

```python
import asyncio
from ragify.engines import ContextScoringEngine

async def custom_scoring_example():
    """Example of custom scoring weights."""
    
    # Define custom weights for different use cases
    technical_weights = {
        'semantic_similarity': 0.35,
        'keyword_overlap': 0.15,
        'source_authority': 0.25,
        'content_quality': 0.15,
        'user_preference': 0.05,
        'freshness': 0.03,
        'contextual_relevance': 0.02
    }
    
    opinion_weights = {
        'semantic_similarity': 0.20,
        'keyword_overlap': 0.10,
        'source_authority': 0.15,
        'content_quality': 0.10,
        'user_preference': 0.30,
        'freshness': 0.10,
        'contextual_relevance': 0.05
    }
    
    # Create scoring engines
    technical_scorer = ContextScoringEngine(
        weights=technical_weights,
        ensemble_method='weighted_average'
    )
    
    opinion_scorer = ContextScoringEngine(
        weights=opinion_weights,
        ensemble_method='geometric_mean'
    )
    
    # Test scoring
    test_chunk = ContextChunk(
        id=uuid4(),
        content="Machine learning algorithms require careful hyperparameter tuning.",
        source=ContextSource(
            id=uuid4(),
            name="Technical Blog",
            source_type=SourceType.DOCUMENT
        ),
        relevance_score=RelevanceScore(score=0.0)
    )
    
    query = "machine learning best practices"
    
    # Score with technical weights
    technical_score = await technical_scorer.calculate_score(test_chunk, query)
    print(f"Technical score: {technical_score.score:.3f}")
    
    # Score with opinion weights
    opinion_score = await opinion_scorer.calculate_score(test_chunk, query)
    print(f"Opinion score: {opinion_score.score:.3f}")

# asyncio.run(custom_scoring_example())
```

### **2. Conflict Resolution Examples**

```python
import asyncio
from ragify.engines import IntelligentContextFusionEngine
from ragify.models import ConflictResolutionStrategy

async def conflict_resolution_example():
    """Example of different conflict resolution strategies."""
    
    # Create conflicting chunks
    chunk1 = ContextChunk(
        id=uuid4(),
        content="Python 3.9 is the latest stable version.",
        source=ContextSource(id=uuid4(), name="Official Docs", source_type=SourceType.DOCUMENT),
        relevance_score=RelevanceScore(score=0.8)
    )
    
    chunk2 = ContextChunk(
        id=uuid4(),
        content="Python 3.11 is the latest stable version.",
        source=ContextSource(id=uuid4(), name="Blog Post", source_type=SourceType.DOCUMENT),
        relevance_score=RelevanceScore(score=0.6)
    )
    
    chunk3 = ContextChunk(
        id=uuid4(),
        content="Python 3.10 is the most widely used version.",
        source=ContextSource(id=uuid4(), name="Survey Data", source_type=SourceType.DATABASE),
        relevance_score=RelevanceScore(score=0.9)
    )
    
    chunks = [chunk1, chunk2, chunk3]
    
    # Test different resolution strategies
    strategies = [
        ConflictResolutionStrategy.HIGHEST_RELEVANCE,
        ConflictResolutionStrategy.HIGHEST_AUTHORITY,
        ConflictResolutionStrategy.CONSENSUS,
        ConflictResolutionStrategy.WEIGHTED_AVERAGE
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.value}")
        
        fusion_engine = IntelligentContextFusionEngine(
            conflict_resolution_strategy=strategy
        )
        
        # Detect conflicts
        conflicts = await fusion_engine.detect_conflicts(chunks)
        print(f"  Conflicts detected: {len(conflicts)}")
        
        # Resolve conflicts
        resolved_chunks = await fusion_engine.resolve_conflicts(conflicts, strategy)
        print(f"  Resolved chunks: {len(resolved_chunks)}")
        
        for chunk in resolved_chunks:
            print(f"    - {chunk.content[:50]}... (Score: {chunk.relevance_score.score:.3f})")

# asyncio.run(conflict_resolution_example())
```

## 🗄️ **Storage Examples**

### **1. Vector Database Integration**

```python
import asyncio
from ragify.storage import VectorDatabase

async def vector_database_example():
    """Example of vector database integration."""
    
    # Initialize different vector databases
    chroma_db = VectorDatabase(
        db_type="chromadb",
        connection_string="memory://",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    faiss_db = VectorDatabase(
        db_type="faiss",
        connection_string="memory://",
        dimension=384
    )
    
    # Store chunks
    test_chunks = [
        ContextChunk(
            id=uuid4(),
            content="Machine learning is a subset of artificial intelligence.",
            source=ContextSource(id=uuid4(), name="AI Guide", source_type=SourceType.DOCUMENT),
            relevance_score=RelevanceScore(score=0.8)
        ),
        ContextChunk(
            id=uuid4(),
            content="Deep learning uses neural networks with multiple layers.",
            source=ContextSource(id=uuid4(), name="ML Book", source_type=SourceType.DOCUMENT),
            relevance_score=RelevanceScore(score=0.9)
        )
    ]
    
    # Store in ChromaDB
    chroma_ids = await chroma_db.store_chunks(test_chunks, "ml_collection")
    print(f"Stored {len(chroma_ids)} chunks in ChromaDB")
    
    # Store in FAISS
    faiss_ids = await faiss_db.store_chunks(test_chunks, "ml_collection")
    print(f"Stored {len(faiss_ids)} chunks in FAISS")
    
    # Search in ChromaDB
    chroma_results = await chroma_db.search_similar(
        "neural networks",
        top_k=2,
        collection_name="ml_collection"
    )
    print(f"ChromaDB search results: {len(chroma_results)}")
    
    # Search in FAISS
    faiss_results = await faiss_db.search_similar(
        "neural networks",
        top_k=2,
        collection_name="ml_collection"
    )
    print(f"FAISS search results: {len(faiss_results)}")

# asyncio.run(vector_database_example())
```

### **2. Caching Examples**

```python
import asyncio
from ragify.storage import CacheManager

async def caching_example():
    """Example of caching strategies."""
    
    # Memory cache
    memory_cache = CacheManager(
        cache_type="memory",
        ttl=3600,
        max_size=1000
    )
    
    # Redis cache
    redis_cache = CacheManager(
        cache_type="redis",
        connection_string="redis://localhost:6379",
        ttl=7200
    )
    
    # Test caching
    test_data = {
        "query": "machine learning",
        "results": ["result1", "result2", "result3"],
        "timestamp": "2024-01-01T12:00:00Z"
    }
    
    # Store in memory cache
    await memory_cache.set("ml_query", test_data, ttl=1800)
    
    # Store in Redis cache
    await redis_cache.set("ml_query", test_data, ttl=3600)
    
    # Retrieve from memory cache
    memory_result = await memory_cache.get("ml_query")
    print(f"Memory cache hit: {memory_result is not None}")
    
    # Retrieve from Redis cache
    redis_result = await redis_cache.get("ml_query")
    print(f"Redis cache hit: {redis_result is not None}")
    
    # Cache statistics
    print(f"Memory cache size: {memory_cache.get_stats()}")
    print(f"Redis cache hit rate: {redis_cache.get_hit_rate():.2f}")

# asyncio.run(caching_example())
```

## 🔒 **Privacy and Security Examples**

### **1. Data Encryption**

```python
import asyncio
from ragify.storage import PrivacyManager

async def privacy_example():
    """Example of privacy and security features."""
    
    # Initialize privacy manager
    privacy_manager = PrivacyManager(
        encryption_key="your-secret-key-here",
        anonymization_enabled=True,
        pii_detection_enabled=True
    )
    
    # Test data with PII
    sensitive_data = """
    User: John Doe
    Email: john.doe@example.com
    Phone: +1-555-123-4567
    Address: 123 Main St, Anytown, USA
    
    This is sensitive information that needs to be protected.
    """
    
    # Encrypt data
    encrypted_data = await privacy_manager.encrypt_data(sensitive_data)
    print(f"Encrypted data: {encrypted_data[:50]}...")
    
    # Decrypt data
    decrypted_data = await privacy_manager.decrypt_data(encrypted_data)
    print(f"Decrypted data matches: {decrypted_data == sensitive_data}")
    
    # Anonymize data
    anonymized_data = await privacy_manager.anonymize_data(sensitive_data)
    print(f"Anonymized data: {anonymized_data}")

# asyncio.run(privacy_example())
```

## 📈 **Performance Optimization Examples**

### **1. Batch Processing**

```python
import asyncio
from ragify import ContextOrchestrator

async def batch_processing_example():
    """Example of batch processing for better performance."""
    
    orchestrator = ContextOrchestrator()
    
    # Batch of queries
    queries = [
        "machine learning algorithms",
        "deep learning frameworks",
        "neural network architectures",
        "data preprocessing techniques",
        "model evaluation metrics"
    ]
    
    # Process queries in parallel
    tasks = []
    for query in queries:
        request = ContextRequest(query=query, max_chunks=3)
        task = orchestrator.get_context(request)
        tasks.append(task)
    
    # Execute all queries concurrently
    responses = await asyncio.gather(*tasks)
    
    # Process results
    total_chunks = sum(len(response.chunks) for response in responses)
    total_time = sum(response.processing_time for response in responses)
    
    print(f"Processed {len(queries)} queries")
    print(f"Total chunks retrieved: {total_chunks}")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per query: {total_time/len(queries):.3f}s")

# asyncio.run(batch_processing_example())
```

### **2. Connection Pooling**

```python
import asyncio
from ragify.sources import DatabaseSource

async def connection_pooling_example():
    """Example of connection pooling for database sources."""
    
    # Database source with connection pooling
    db_source = DatabaseSource(
        config=SourceConfig(
            id="pooled_db",
            name="Pooled Database",
            source_type=SourceType.DATABASE,
            privacy_level=PrivacyLevel.PRIVATE
        ),
        connection_string="postgresql://user:pass@localhost/db",
        db_type="postgresql",
        connection_pool_size=20  # Larger pool for high concurrency
    )
    
    # Simulate concurrent queries
    async def query_database(query: str):
        return await db_source.get_chunks(query, max_chunks=2)
    
    # Create multiple concurrent queries
    queries = [f"query_{i}" for i in range(10)]
    tasks = [query_database(query) for query in queries]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful_queries = sum(1 for r in results if not isinstance(r, Exception))
    print(f"Successful queries: {successful_queries}/{len(queries)}")
    
    # Close connections
    await db_source.close()

# asyncio.run(connection_pooling_example())
```

---

## 📚 **Next Steps**

- **[API Reference](api-reference.md)** - Complete API documentation
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Performance](performance.md)** - Performance optimization guide
