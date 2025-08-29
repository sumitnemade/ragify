# 🚀 **Basic Usage Guide**

This guide will help you get started with Ragify and show you how to use its core features.

## 🎯 **Quick Start**

### **1. Installation**

```bash
# Install Ragify
pip install ragify

# Or install from source
git clone https://github.com/sumitnemade/ragify.git
cd ragify
pip install -e .
```

### **2. Basic Setup**

```python
import asyncio
from ragify import ContextOrchestrator, OrchestratorConfig, ContextRequest

# Create configuration
config = OrchestratorConfig(
    max_contexts=100,
    default_chunk_size=1000,
    default_overlap=200
)

# Initialize orchestrator
orchestrator = ContextOrchestrator(config)
```

### **3. Your First Context Request**

```python
async def get_context():
    # Create a context request
    request = ContextRequest(
        query="What is machine learning?",
        user_id="user123",
        session_id="session456"
    )
    
    # Get context from orchestrator
    response = await orchestrator.get_context(request)
    
    # Print results
    print(f"Query: {response.query}")
    print(f"Total chunks: {len(response.chunks)}")
    print(f"Sources: {[chunk.source.name for chunk in response.chunks]}")
    
    # Access individual chunks
    for i, chunk in enumerate(response.chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Source: {chunk.source.name}")
        print(f"  Relevance: {chunk.relevance_score.score:.3f}")

# Run the example
asyncio.run(get_context())
```

## 📚 **Working with Data Sources**

### **1. Document Sources**

```python
from ragify.sources import DocumentSource
from ragify.models import SourceType

# Create a document source
doc_source = DocumentSource(
    name="my_documents",
    source_type=SourceType.DOCUMENT,
    url="/path/to/your/documents",  # Directory path
    chunk_size=1000,
    overlap=200
)

# Add to orchestrator
orchestrator.add_source(doc_source)

# Query documents
request = ContextRequest(
    query="Find information about Python programming",
    user_id="user123"
)

response = await orchestrator.get_context(request)
```

### **2. API Sources**

```python
from ragify.sources import APISource

# Create an API source
api_source = APISource(
    name="weather_api",
    source_type=SourceType.API,
    url="https://api.weatherapi.com/v1/current.json",
    auth_type="api_key",
    auth_config={"api_key": "your_api_key_here"},
    headers={"Accept": "application/json"}
)

# Add to orchestrator
orchestrator.add_source(api_source)

# Query API
request = ContextRequest(
    query="What's the weather in New York?",
    user_id="user123"
)

response = await orchestrator.get_context(request)
```

### **3. Database Sources**

```python
from ragify.sources import DatabaseSource

# Create a database source
db_source = DatabaseSource(
    name="user_database",
    source_type=SourceType.DATABASE,
    url="postgresql://user:password@localhost:5432/mydb",
    db_type="postgresql",
    query_template="SELECT content, relevance FROM documents WHERE content ILIKE '%{query}%'"
)

# Add to orchestrator
orchestrator.add_source(db_source)

# Query database
request = ContextRequest(
    query="Find user profiles",
    user_id="user123"
)

response = await orchestrator.get_context(request)
```

## 🔄 **Context Fusion Examples**

### **1. Basic Fusion**

```python
# Multiple sources will be automatically fused
sources = [
    DocumentSource(name="docs", url="/docs"),
    APISource(name="api", url="https://api.example.com"),
    DatabaseSource(name="db", url="postgresql://...")
]

for source in sources:
    orchestrator.add_source(source)

# Fusion happens automatically
request = ContextRequest(
    query="Machine learning algorithms",
    user_id="user123"
)

response = await orchestrator.get_context(request)
print(f"Fused {len(response.chunks)} chunks from {len(sources)} sources")
```

### **2. Custom Fusion Strategy**

```python
from ragify.engines import IntelligentContextFusionEngine
from ragify.models import ConflictResolutionStrategy

# Configure fusion engine
fusion_engine = IntelligentContextFusionEngine(
    conflict_resolution_strategy=ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    enable_semantic_analysis=True,
    enable_factual_verification=True
)

# Use custom fusion engine
orchestrator.fusion_engine = fusion_engine
```

## 🎯 **Scoring and Relevance**

### **1. Understanding Scores**

```python
# Each chunk has a relevance score
for chunk in response.chunks:
    score = chunk.relevance_score
    print(f"Score: {score.score:.3f}")
    print(f"Confidence: {score.confidence_level:.3f}")
    print(f"Factors: {score.factors}")
```

### **2. Filtering by Relevance**

```python
# Filter chunks by minimum relevance
min_relevance = 0.7
relevant_chunks = [
    chunk for chunk in response.chunks 
    if chunk.relevance_score.score >= min_relevance
]

print(f"Found {len(relevant_chunks)} highly relevant chunks")
```

### **3. Custom Scoring Weights**

```python
from ragify.engines import ContextScoringEngine

# Configure scoring engine with custom weights
scoring_engine = ContextScoringEngine(
    scoring_weights={
        'semantic_similarity': 0.4,
        'keyword_overlap': 0.3,
        'source_authority': 0.2,
        'content_quality': 0.1
    }
)

# Use custom scoring engine
orchestrator.scoring_engine = scoring_engine
```

## 🔐 **Privacy and Security**

### **1. Privacy Levels**

```python
from ragify.models import PrivacyLevel

# Request with specific privacy level
request = ContextRequest(
    query="Sensitive information",
    user_id="user123",
    privacy_level=PrivacyLevel.RESTRICTED  # Encrypted + anonymized
)

response = await orchestrator.get_context(request)
```

### **2. Encryption**

```python
from ragify.storage import PrivacyManager

# Initialize privacy manager with encryption
privacy_manager = PrivacyManager(
    default_privacy_level=PrivacyLevel.RESTRICTED,
    encryption_key="your_encryption_key"
)

# Privacy is automatically applied based on request level
```

## 💾 **Storage and Caching**

### **1. Vector Database Storage**

```python
from ragify.storage import VectorDatabase

# Configure vector database
vector_db = VectorDatabase(
    vector_db_url="chromadb://localhost:8000"
)

# Add to orchestrator
orchestrator.vector_database = vector_db
```

### **2. Cache Management**

```python
from ragify.storage import CacheManager

# Configure cache
cache_manager = CacheManager(
    cache_url="redis://localhost:6379"
)

# Add to orchestrator
orchestrator.cache_manager = cache_manager
```

## 🔄 **Real-time Updates**

### **1. Real-time Sources**

```python
from ragify.sources import RealtimeSource

# Create real-time source
realtime_source = RealtimeSource(
    name="live_data",
    source_type=SourceType.REALTIME,
    url="ws://localhost:8080/stream",
    connection_type="websocket"
)

# Add to orchestrator
orchestrator.add_source(realtime_source)

# Real-time updates are automatically processed
```

### **2. Update Notifications**

```python
# Subscribe to updates
async def handle_update(update_data):
    print(f"Received update: {update_data}")

orchestrator.updates_engine.subscribe_to_updates(
    source_name="live_data",
    callback=handle_update
)
```

## 📊 **Advanced Usage**

### **1. Batch Processing**

```python
# Process multiple queries
queries = [
    "Machine learning basics",
    "Python programming",
    "Data science tools"
]

responses = []
for query in queries:
    request = ContextRequest(query=query, user_id="user123")
    response = await orchestrator.get_context(request)
    responses.append(response)

print(f"Processed {len(responses)} queries")
```

### **2. Custom Chunk Processing**

```python
# Custom chunk processing
async def process_chunks(chunks):
    processed_chunks = []
    for chunk in chunks:
        # Add custom processing
        chunk.metadata['processed'] = True
        chunk.metadata['timestamp'] = datetime.now().isoformat()
        processed_chunks.append(chunk)
    return processed_chunks

# Apply custom processing
response.chunks = await process_chunks(response.chunks)
```

### **3. Error Handling**

```python
import asyncio
from ragify.exceptions import ICOException

async def safe_context_request(query, user_id):
    try:
        request = ContextRequest(query=query, user_id=user_id)
        response = await orchestrator.get_context(request)
        return response
    except ICOException as e:
        print(f"Ragify error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Use with error handling
response = await safe_context_request("test query", "user123")
if response:
    print(f"Success: {len(response.chunks)} chunks")
```

## 🎯 **Best Practices**

### **1. Configuration Management**

```python
# Use environment variables for configuration
import os
from dotenv import load_dotenv

load_dotenv()

config = OrchestratorConfig(
    max_contexts=int(os.getenv('MAX_CONTEXTS', 100)),
    default_chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    default_overlap=int(os.getenv('OVERLAP', 200))
)
```

### **2. Resource Management**

```python
# Proper cleanup
async def main():
    orchestrator = ContextOrchestrator(config)
    try:
        # Your code here
        response = await orchestrator.get_context(request)
    finally:
        await orchestrator.close()

# Run with proper cleanup
asyncio.run(main())
```

### **3. Performance Optimization**

```python
# Use connection pooling
# Configure appropriate chunk sizes
# Enable caching for frequently accessed data
# Use async/await for I/O operations
```

## 📚 **Next Steps**

- **[Examples](examples.md)** - More detailed examples
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Configuration](configuration.md)** - Advanced configuration options
- **[Deployment](deployment.md)** - Production deployment guide

---

**Need Help?** [Create an issue](https://github.com/sumitnemade/ragify/issues) or [join discussions](https://github.com/sumitnemade/ragify/discussions)
