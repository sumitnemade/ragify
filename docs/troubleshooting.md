# 🔧 **Troubleshooting Guide**

This guide helps you resolve common issues and problems when using Ragify.

## 🚨 **Common Issues**

### **1. Installation Problems**

#### **Issue: ModuleNotFoundError for dependencies**

```bash
ModuleNotFoundError: No module named 'chromadb'
```

**Solution:**
```bash
# Install missing dependencies
pip install chromadb

# Or install all dependencies
pip install -r requirements.txt

# For development dependencies
pip install -r requirements-dev.txt
```

#### **Issue: Version conflicts**

```bash
ERROR: Cannot install ragify because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Create a fresh virtual environment
python -m venv ragify_env
source ragify_env/bin/activate  # On Windows: ragify_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install with --no-deps to avoid conflicts
pip install ragify --no-deps
pip install -r requirements.txt
```

### **2. Configuration Issues**

#### **Issue: Invalid configuration**

```python
ValidationError: 1 validation error for OrchestratorConfig
vector_db_url
  field required (type=value_error.missing)
```

**Solution:**
```python
from ragify import OrchestratorConfig

# Provide all required fields
config = OrchestratorConfig(
    vector_db_url="memory://",  # Required field
    fusion_strategy="highest_relevance",
    enable_semantic_analysis=True
)
```

#### **Issue: Environment variables not loaded**

```python
ConfigurationError: Database connection string not found
```

**Solution:**
```bash
# Set environment variables
export RAGIFY_DB_URL="postgresql://user:pass@localhost/db"
export RAGIFY_VECTOR_DB_URL="memory://"
export RAGIFY_CACHE_TYPE="memory"

# Or use a .env file
echo "RAGIFY_DB_URL=postgresql://user:pass@localhost/db" > .env
echo "RAGIFY_VECTOR_DB_URL=memory://" >> .env
echo "RAGIFY_CACHE_TYPE=memory" >> .env
```

### **3. Data Source Issues**

#### **Issue: Document source not finding files**

```python
FileNotFoundError: Document file not found: docs/api.md
```

**Solution:**
```python
from pathlib import Path

# Check if file exists
doc_path = Path("docs/api.md")
if not doc_path.exists():
    print(f"File not found: {doc_path}")
    # Create the file or use correct path
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("# API Documentation\n\nContent here...")

# Use absolute paths
doc_source = DocumentSource(
    config=config,
    document_paths=[str(Path.cwd() / "docs" / "api.md")]
)
```

#### **Issue: API source connection timeout**

```python
TimeoutError: Request timed out after 30 seconds
```

**Solution:**
```python
# Increase timeout
api_source = APISource(
    config=config,
    base_url="https://api.example.com",
    timeout=60  # Increase timeout to 60 seconds
)

# Or use retry logic
api_source = APISource(
    config=config,
    base_url="https://api.example.com",
    retry_attempts=3,
    retry_delay=1.0
)
```

#### **Issue: Database connection failed**

```python
ConnectionError: Failed to connect to database
```

**Solution:**
```python
# Check connection string
connection_string = "postgresql://user:pass@localhost/db"

# Test connection manually
import asyncpg
try:
    conn = await asyncpg.connect(connection_string)
    await conn.close()
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")

# Use connection pooling
db_source = DatabaseSource(
    config=config,
    connection_string=connection_string,
    connection_pool_size=5,
    max_retries=3
)
```

### **4. Vector Database Issues**

#### **Issue: ChromaDB initialization error**

```python
ChromaDBError: Failed to initialize ChromaDB
```

**Solution:**
```python
# Use in-memory ChromaDB for testing
vector_db = VectorDatabase(
    db_type="chromadb",
    connection_string="memory://"
)

# For persistent storage, ensure directory exists
import os
os.makedirs("./chroma_db", exist_ok=True)

vector_db = VectorDatabase(
    db_type="chromadb",
    connection_string="./chroma_db"
)
```

#### **Issue: FAISS dimension mismatch**

```python
ValueError: Dimension mismatch: expected 384, got 1536
```

**Solution:**
```python
# Match embedding model dimension
vector_db = VectorDatabase(
    db_type="faiss",
    connection_string="memory://",
    dimension=1536  # Match your embedding model dimension
)

# Or use a different embedding model
vector_db = VectorDatabase(
    db_type="faiss",
    connection_string="memory://",
    embedding_model="all-MiniLM-L6-v2",  # 384 dimensions
    dimension=384
)
```

### **5. Performance Issues**

#### **Issue: Slow context retrieval**

```python
# Context retrieval taking too long
```

**Solution:**
```python
# 1. Enable caching
config = OrchestratorConfig(
    cache_type="memory",
    cache_ttl=3600,
    enable_caching=True
)

# 2. Use connection pooling
db_source = DatabaseSource(
    config=config,
    connection_string=connection_string,
    connection_pool_size=10
)

# 3. Limit concurrent sources
config = OrchestratorConfig(
    max_concurrent_sources=5  # Reduce from default 10
)

# 4. Use batch processing
async def batch_get_context(orchestrator, queries):
    tasks = [
        orchestrator.get_context(ContextRequest(query=q))
        for q in queries
    ]
    return await asyncio.gather(*tasks)
```

#### **Issue: High memory usage**

```python
# Memory usage growing over time
```

**Solution:**
```python
# 1. Use streaming for large documents
doc_source = DocumentSource(
    config=config,
    document_paths=document_paths,
    chunk_size=1000,  # Smaller chunks
    chunk_overlap=100
)

# 2. Enable garbage collection
import gc
gc.collect()

# 3. Use external cache
config = OrchestratorConfig(
    cache_type="redis",
    connection_string="redis://localhost:6379"
)

# 4. Limit cache size
cache_manager = CacheManager(
    cache_type="memory",
    max_size=1000,  # Limit cache entries
    ttl=1800  # Shorter TTL
)
```

### **6. Fusion and Scoring Issues**

#### **Issue: Low relevance scores**

```python
# All chunks have very low relevance scores
```

**Solution:**
```python
# 1. Adjust scoring weights
custom_weights = {
    'semantic_similarity': 0.4,  # Increase semantic weight
    'keyword_overlap': 0.3,
    'source_authority': 0.2,
    'content_quality': 0.1
}

scoring_engine = ContextScoringEngine(weights=custom_weights)

# 2. Lower minimum relevance threshold
request = ContextRequest(
    query="your query",
    min_relevance=0.3  # Lower from default 0.7
)

# 3. Check embedding model
vector_db = VectorDatabase(
    db_type="chromadb",
    embedding_model="all-MiniLM-L6-v2"  # Try different model
)
```

#### **Issue: Too many conflicts detected**

```python
# Fusion engine detecting too many conflicts
```

**Solution:**
```python
# 1. Adjust similarity threshold
fusion_engine = IntelligentContextFusionEngine(
    similarity_threshold=0.9  # Increase from default 0.8
)

# 2. Use different conflict resolution strategy
fusion_engine = IntelligentContextFusionEngine(
    conflict_resolution_strategy=ConflictResolutionStrategy.CONSENSUS
)

# 3. Disable semantic analysis for simple cases
fusion_engine = IntelligentContextFusionEngine(
    enable_semantic_analysis=False
)
```

### **7. Privacy and Security Issues**

#### **Issue: Encryption key not set**

```python
ConfigurationError: Encryption key required for sensitive data
```

**Solution:**
```python
# Set encryption key
privacy_manager = PrivacyManager(
    encryption_key="your-secure-key-here"
)

# Or use environment variable
import os
os.environ["RAGIFY_ENCRYPTION_KEY"] = "your-secure-key-here"

privacy_manager = PrivacyManager()
```

#### **Issue: PII detection not working**

```python
# PII not being detected in text
```

**Solution:**
```python
# 1. Enable PII detection
privacy_manager = PrivacyManager(
    pii_detection_enabled=True,
    pii_patterns=[
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b'  # Phone
    ]
)

# 2. Use custom PII patterns
custom_patterns = [
    r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',  # IBAN
    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'  # Credit card
]

privacy_manager = PrivacyManager(
    pii_detection_enabled=True,
    pii_patterns=custom_patterns
)
```

## 🔍 **Debugging Techniques**

### **1. Enable Debug Logging**

```python
import logging
import structlog

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### **2. Performance Profiling**

```python
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def profile_operation(operation_name: str):
    """Profile an async operation."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"{operation_name} took {duration:.3f} seconds")

# Usage
async def debug_context_retrieval():
    async with profile_operation("Context Retrieval"):
        response = await orchestrator.get_context(request)
    
    print(f"Retrieved {len(response.chunks)} chunks")
    print(f"Sources used: {response.sources}")
    print(f"Processing time: {response.processing_time:.3f}s")
```

### **3. Source Health Checks**

```python
async def check_source_health(orchestrator):
    """Check health of all data sources."""
    sources = await orchestrator.list_sources()
    
    for source in sources:
        try:
            # Test source with simple query
            chunks = await source.get_chunks("test", max_chunks=1)
            status = "healthy"
            chunk_count = len(chunks)
        except Exception as e:
            status = "error"
            chunk_count = 0
            error_msg = str(e)
        
        print(f"Source: {source.name}")
        print(f"  Status: {status}")
        print(f"  Chunks: {chunk_count}")
        if status == "error":
            print(f"  Error: {error_msg}")
        print()

# Usage
await check_source_health(orchestrator)
```

### **4. Memory Usage Monitoring**

```python
import psutil
import os

def monitor_memory_usage():
    """Monitor memory usage of the current process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Get memory percentage
    memory_percent = process.memory_percent()
    print(f"Memory percentage: {memory_percent:.2f}%")

# Usage
monitor_memory_usage()
```

## 🛠️ **Common Fixes**

### **1. Connection Pool Exhaustion**

```python
# Problem: Too many database connections
# Solution: Implement connection pooling properly

db_source = DatabaseSource(
    config=config,
    connection_string=connection_string,
    connection_pool_size=5,  # Limit pool size
    max_overflow=10,  # Allow overflow connections
    pool_timeout=30  # Timeout for getting connection
)

# Always close connections
try:
    chunks = await db_source.get_chunks(query)
finally:
    await db_source.close()
```

### **2. Cache Memory Leaks**

```python
# Problem: Cache growing indefinitely
# Solution: Implement proper cache eviction

cache_manager = CacheManager(
    cache_type="memory",
    max_size=1000,  # Limit cache size
    ttl=3600,  # Set TTL
    eviction_policy="lru"  # Use LRU eviction
)

# Clear cache periodically
import asyncio
async def clear_cache_periodically():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await cache_manager.clear()

# Start cache clearing task
asyncio.create_task(clear_cache_periodically())
```

### **3. Rate Limiting Issues**

```python
# Problem: API rate limits exceeded
# Solution: Implement proper rate limiting

api_source = APISource(
    config=config,
    base_url="https://api.example.com",
    rate_limit=100,  # Requests per minute
    rate_limit_window=60,  # Window in seconds
    retry_attempts=3,
    retry_delay=1.0,
    backoff_factor=2.0
)

# Use exponential backoff
api_source = APISource(
    config=config,
    base_url="https://api.example.com",
    retry_attempts=5,
    retry_delay=1.0,
    backoff_factor=2.0,
    max_retry_delay=60.0
)
```

### **4. Large File Processing**

```python
# Problem: Large files causing memory issues
# Solution: Use streaming processing

doc_source = DocumentSource(
    config=config,
    document_paths=large_files,
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50,  # Minimal overlap
    max_file_size=10 * 1024 * 1024,  # 10MB limit
    enable_streaming=True
)

# Process files in batches
async def process_large_files(file_paths, batch_size=5):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        
        # Process batch
        for file_path in batch:
            chunks = await doc_source.process_file(file_path)
            # Process chunks...
        
        # Clear memory
        gc.collect()
```

## 📊 **Monitoring and Alerts**

### **1. Health Check Endpoint**

```python
from fastapi import FastAPI, HTTPException
from ragify import ContextOrchestrator

app = FastAPI()
orchestrator = ContextOrchestrator()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test basic functionality
        request = ContextRequest(query="test", max_chunks=1)
        response = await orchestrator.get_context(request)
        
        return {
            "status": "healthy",
            "chunks_retrieved": len(response.chunks),
            "processing_time": response.processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/sources")
async def source_health_check():
    """Check health of all sources."""
    sources = await orchestrator.list_sources()
    health_status = {}
    
    for source in sources:
        try:
            chunks = await source.get_chunks("test", max_chunks=1)
            health_status[source.name] = {
                "status": "healthy",
                "chunks": len(chunks)
            }
        except Exception as e:
            health_status[source.name] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_status
```

### **2. Performance Metrics**

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    operation: str
    duration: float
    success: bool
    error: str = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def record_metric(self, operation: str, duration: float, success: bool, error: str = None):
        """Record a performance metric."""
        metric = PerformanceMetrics(operation, duration, success, error)
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, any]:
        """Get performance summary."""
        if not self.metrics:
            return {}
        
        successful_ops = [m for m in self.metrics if m.success]
        failed_ops = [m for m in self.metrics if not m.success]
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.metrics),
            "avg_duration": sum(m.duration for m in successful_ops) / len(successful_ops) if successful_ops else 0,
            "max_duration": max(m.duration for m in self.metrics),
            "min_duration": min(m.duration for m in self.metrics)
        }

# Usage
monitor = PerformanceMonitor()

async def monitored_context_retrieval(request):
    start_time = time.time()
    try:
        response = await orchestrator.get_context(request)
        duration = time.time() - start_time
        monitor.record_metric("context_retrieval", duration, True)
        return response
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_metric("context_retrieval", duration, False, str(e))
        raise

# Get performance summary
summary = monitor.get_summary()
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average duration: {summary['avg_duration']:.3f}s")
```

## 🆘 **Getting Help**

### **1. Enable Verbose Logging**

```python
import logging

# Enable all loggers
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("ragify").setLevel(logging.DEBUG)

# Log to file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ragify_debug.log'),
        logging.StreamHandler()
    ]
)
```

### **2. Create Minimal Reproduction**

```python
# Create a minimal example that reproduces the issue
async def minimal_reproduction():
    """Minimal example to reproduce the issue."""
    
    # 1. Basic setup
    config = OrchestratorConfig(
        vector_db_url="memory://",
        cache_type="memory"
    )
    
    orchestrator = ContextOrchestrator(config)
    
    # 2. Add minimal source
    doc_source = DocumentSource(
        config=SourceConfig(
            id="test",
            name="Test",
            source_type=SourceType.DOCUMENT,
            privacy_level=PrivacyLevel.PUBLIC
        ),
        document_paths=["test.txt"]
    )
    
    await orchestrator.add_source(doc_source)
    
    # 3. Test with simple query
    request = ContextRequest(query="test", max_chunks=1)
    
    try:
        response = await orchestrator.get_context(request)
        print(f"Success: {len(response.chunks)} chunks")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# Run minimal reproduction
asyncio.run(minimal_reproduction())
```

### **3. Collect System Information**

```python
import sys
import platform
import ragify

def collect_system_info():
    """Collect system information for debugging."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "ragify_version": ragify.__version__,
        "installed_packages": [
            "chromadb",
            "pinecone-client",
            "weaviate-client",
            "faiss-cpu",
            "redis",
            "asyncpg",
            "aiohttp"
        ]
    }

# Print system information
info = collect_system_info()
for key, value in info.items():
    print(f"{key}: {value}")
```

---

## 📚 **Next Steps**

- **[Performance](performance.md)** - Performance optimization guide
- **[Configuration](configuration.md)** - Advanced configuration options
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples.md)** - Code examples and tutorials
