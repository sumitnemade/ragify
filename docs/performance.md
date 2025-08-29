# ⚡ **Performance Optimization Guide**

This guide provides comprehensive strategies for optimizing Ragify's performance in various deployment scenarios.

## 🎯 **Performance Overview**

Ragify's performance depends on multiple factors including data source efficiency, caching strategies, vector database optimization, and system resources.

```mermaid
graph TB
    subgraph "Performance Factors"
        A[Data Source Performance] --> E[Overall System Performance]
        B[Caching Strategy] --> E
        C[Vector Database] --> E
        D[Concurrency] --> E
        F[Memory Management] --> E
        G[Network Latency] --> E
    end
    
    subgraph "Optimization Areas"
        H[Query Optimization]
        I[Connection Pooling]
        J[Batch Processing]
        K[Indexing]
        L[Load Balancing]
    end
    
    E --> H
    E --> I
    E --> J
    E --> K
    E --> L
```

## 📊 **Performance Benchmarks**

### **1. Baseline Performance**

```python
import asyncio
import time
from ragify import ContextOrchestrator, ContextRequest

async def baseline_performance_test():
    """Baseline performance test."""
    
    # Initialize orchestrator
    config = OrchestratorConfig(
        vector_db_url="memory://",
        cache_type="memory",
        max_concurrent_sources=5
    )
    orchestrator = ContextOrchestrator(config)
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "deep learning frameworks",
        "data preprocessing techniques",
        "model evaluation metrics",
        "neural network architectures"
    ]
    
    results = []
    
    for query in test_queries:
        start_time = time.time()
        
        request = ContextRequest(
            query=query,
            max_chunks=10,
            min_relevance=0.7
        )
        
        response = await orchestrator.get_context(request)
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'query': query,
            'duration': duration,
            'chunks': len(response.chunks),
            'processing_time': response.processing_time
        })
    
    # Calculate statistics
    avg_duration = sum(r['duration'] for r in results) / len(results)
    avg_chunks = sum(r['chunks'] for r in results) / len(results)
    
    print(f"Average query duration: {avg_duration:.3f}s")
    print(f"Average chunks retrieved: {avg_chunks:.1f}")
    print(f"Average processing time: {sum(r['processing_time'] for r in results) / len(results):.3f}s")

# Run baseline test
asyncio.run(baseline_performance_test())
```

### **2. Performance Targets**

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| **Query Response Time** | < 100ms | 100-500ms | > 500ms |
| **Context Retrieval** | < 50ms | 50-200ms | > 200ms |
| **Vector Search** | < 20ms | 20-100ms | > 100ms |
| **Cache Hit Rate** | > 90% | 70-90% | < 70% |
| **Memory Usage** | < 1GB | 1-4GB | > 4GB |
| **Concurrent Queries** | > 1000 | 100-1000 | < 100 |

## 🚀 **Optimization Strategies**

### **1. Caching Optimization**

#### **Multi-Level Caching**

```python
from ragify.storage import CacheManager

class MultiLevelCache:
    def __init__(self):
        # L1: In-memory cache (fastest)
        self.l1_cache = CacheManager(
            cache_type="memory",
            max_size=10000,
            ttl=300  # 5 minutes
        )
        
        # L2: Redis cache (medium speed)
        self.l2_cache = CacheManager(
            cache_type="redis",
            connection_string="redis://localhost:6379",
            ttl=3600  # 1 hour
        )
        
        # L3: Disk cache (slowest but persistent)
        self.l3_cache = CacheManager(
            cache_type="file",
            cache_dir="./cache",
            ttl=86400  # 24 hours
        )
    
    async def get(self, key: str):
        """Get value from multi-level cache."""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 cache
        value = await self.l2_cache.get(key)
        if value is not None:
            # Populate L1 cache
            await self.l1_cache.set(key, value, ttl=300)
            return value
        
        # Try L3 cache
        value = await self.l3_cache.get(key)
        if value is not None:
            # Populate L1 and L2 caches
            await self.l1_cache.set(key, value, ttl=300)
            await self.l2_cache.set(key, value, ttl=3600)
            return value
        
        return None
    
    async def set(self, key: str, value: any, ttl: int = None):
        """Set value in all cache levels."""
        await self.l1_cache.set(key, value, ttl=300)
        await self.l2_cache.set(key, value, ttl=3600)
        await self.l3_cache.set(key, value, ttl=86400)

# Usage
multi_cache = MultiLevelCache()
```

#### **Intelligent Cache Warming**

```python
class CacheWarmer:
    def __init__(self, orchestrator, cache_manager):
        self.orchestrator = orchestrator
        self.cache_manager = cache_manager
        self.popular_queries = []
    
    async def warm_cache(self):
        """Warm cache with popular queries."""
        popular_queries = [
            "machine learning",
            "python programming",
            "data science",
            "artificial intelligence",
            "deep learning"
        ]
        
        tasks = []
        for query in popular_queries:
            task = self.warm_query(query)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def warm_query(self, query: str):
        """Warm cache for a specific query."""
        try:
            request = ContextRequest(
                query=query,
                max_chunks=10,
                min_relevance=0.7
            )
            
            response = await self.orchestrator.get_context(request)
            
            # Cache the response
            cache_key = f"query:{hash(query)}"
            await self.cache_manager.set(cache_key, response, ttl=3600)
            
            print(f"Warmed cache for query: {query}")
        except Exception as e:
            print(f"Failed to warm cache for {query}: {e}")

# Usage
cache_warmer = CacheWarmer(orchestrator, cache_manager)
await cache_warmer.warm_cache()
```

### **2. Vector Database Optimization**

#### **Index Optimization**

```python
from ragify.storage import VectorDatabase

class OptimizedVectorDB:
    def __init__(self):
        self.vector_db = VectorDatabase(
            db_type="chromadb",
            connection_string="memory://",
            embedding_model="all-MiniLM-L6-v2",
            dimension=384
        )
        
        # Optimize index parameters
        self.index_config = {
            'nlist': 1000,  # Number of clusters
            'nprobe': 10,   # Number of clusters to search
            'metric': 'cosine'
        }
    
    async def optimize_index(self):
        """Optimize vector database index."""
        # For FAISS
        if self.vector_db.db_type == "faiss":
            await self.vector_db.optimize_index(
                nlist=self.index_config['nlist'],
                nprobe=self.index_config['nprobe']
            )
        
        # For ChromaDB
        elif self.vector_db.db_type == "chromadb":
            await self.vector_db.optimize_collection(
                optimize_threshold=1000  # Optimize after 1000 vectors
            )
    
    async def batch_insert(self, chunks: List[ContextChunk]):
        """Insert chunks in batches for better performance."""
        batch_size = 1000
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            await self.vector_db.store_chunks(batch)
            
            # Optimize index periodically
            if (i + batch_size) % 10000 == 0:
                await self.optimize_index()

# Usage
optimized_db = OptimizedVectorDB()
await optimized_db.batch_insert(large_chunk_list)
```

#### **Embedding Model Optimization**

```python
class EmbeddingOptimizer:
    def __init__(self):
        self.models = {
            'fast': 'all-MiniLM-L6-v2',      # 384d, fast
            'balanced': 'all-mpnet-base-v2',  # 768d, balanced
            'accurate': 'all-MiniLM-L12-v2'   # 384d, accurate
        }
    
    def select_model(self, use_case: str, performance_requirement: str):
        """Select optimal embedding model."""
        if performance_requirement == 'speed':
            return self.models['fast']
        elif performance_requirement == 'accuracy':
            return self.models['accurate']
        else:
            return self.models['balanced']
    
    async def benchmark_models(self, test_queries: List[str]):
        """Benchmark different embedding models."""
        results = {}
        
        for model_name, model_id in self.models.items():
            vector_db = VectorDatabase(
                db_type="chromadb",
                embedding_model=model_id
            )
            
            start_time = time.time()
            
            for query in test_queries:
                await vector_db.search_similar(query, top_k=10)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[model_name] = {
                'duration': duration,
                'avg_time_per_query': duration / len(test_queries)
            }
        
        return results

# Usage
optimizer = EmbeddingOptimizer()
benchmark_results = await optimizer.benchmark_models(test_queries)
print("Model benchmarks:", benchmark_results)
```

### **3. Connection Pooling**

#### **Database Connection Pooling**

```python
from ragify.sources import DatabaseSource

class OptimizedDatabaseSource(DatabaseSource):
    def __init__(self, config, connection_string, **kwargs):
        super().__init__(config, connection_string, **kwargs)
        
        # Optimize connection pool
        self.pool_config = {
            'min_size': 5,
            'max_size': 20,
            'max_queries': 50000,
            'max_inactive_connection_lifetime': 300.0,
            'setup': self._setup_connection,
            'cleanup': self._cleanup_connection
        }
    
    async def _setup_connection(self, connection):
        """Setup connection with optimizations."""
        # Set session variables for better performance
        await connection.execute("SET SESSION work_mem = '256MB'")
        await connection.execute("SET SESSION shared_preload_libraries = 'pg_stat_statements'")
        await connection.execute("SET SESSION track_activity_query_size = 2048")
    
    async def _cleanup_connection(self, connection):
        """Cleanup connection."""
        await connection.execute("RESET ALL")
    
    async def get_optimized_chunks(self, query: str, max_chunks: int = 10):
        """Get chunks with optimized query."""
        # Use prepared statements
        prepared_query = """
            SELECT content, relevance_score, metadata
            FROM documents
            WHERE content ILIKE $1
            ORDER BY relevance_score DESC
            LIMIT $2
        """
        
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(prepared_query, f"%{query}%", max_chunks)
            return [self._row_to_chunk(row) for row in rows]

# Usage
optimized_db = OptimizedDatabaseSource(
    config=config,
    connection_string="postgresql://user:pass@localhost/db",
    connection_pool_size=20
)
```

#### **API Connection Pooling**

```python
import aiohttp
from ragify.sources import APISource

class OptimizedAPISource(APISource):
    def __init__(self, config, base_url, **kwargs):
        super().__init__(config, base_url, **kwargs)
        
        # Optimize HTTP session
        self.session_config = {
            'connector': aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            ),
            'timeout': aiohttp.ClientTimeout(
                total=30,
                connect=10,
                sock_read=30
            ),
            'headers': {
                'User-Agent': 'Ragify/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        }
    
    async def _create_session(self):
        """Create optimized HTTP session."""
        return aiohttp.ClientSession(**self.session_config)
    
    async def get_optimized_chunks(self, query: str, max_chunks: int = 10):
        """Get chunks with connection pooling."""
        async with self._create_session() as session:
            # Use connection pooling
            async with session.get(
                f"{self.base_url}/search",
                params={'q': query, 'limit': max_chunks},
                headers=self.session_config['headers']
            ) as response:
                data = await response.json()
                return self._parse_response(data)

# Usage
optimized_api = OptimizedAPISource(
    config=config,
    base_url="https://api.example.com"
)
```

### **4. Batch Processing**

#### **Query Batching**

```python
class BatchProcessor:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.batch_size = 50
    
    async def process_batch(self, queries: List[str]):
        """Process multiple queries in batch."""
        # Group queries by similarity
        query_groups = self._group_similar_queries(queries)
        
        results = []
        
        for group in query_groups:
            # Process similar queries together
            batch_results = await self._process_query_group(group)
            results.extend(batch_results)
        
        return results
    
    def _group_similar_queries(self, queries: List[str]):
        """Group similar queries for batch processing."""
        # Simple grouping by first word
        groups = {}
        for query in queries:
            first_word = query.split()[0].lower()
            if first_word not in groups:
                groups[first_word] = []
            groups[first_word].append(query)
        
        return list(groups.values())
    
    async def _process_query_group(self, queries: List[str]):
        """Process a group of similar queries."""
        tasks = []
        
        for query in queries:
            request = ContextRequest(
                query=query,
                max_chunks=10,
                min_relevance=0.7
            )
            task = self.orchestrator.get_context(request)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# Usage
batch_processor = BatchProcessor(orchestrator)
results = await batch_processor.process_batch(large_query_list)
```

#### **Data Source Batching**

```python
class DataSourceBatcher:
    def __init__(self, sources: List[BaseDataSource]):
        self.sources = sources
        self.batch_size = 10
    
    async def batch_get_chunks(self, query: str, max_chunks: int = 10):
        """Get chunks from multiple sources in batches."""
        all_chunks = []
        
        # Process sources in batches
        for i in range(0, len(self.sources), self.batch_size):
            batch_sources = self.sources[i:i + self.batch_size]
            
            # Create tasks for batch
            tasks = []
            for source in batch_sources:
                task = source.get_chunks(query, max_chunks)
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, list):
                    all_chunks.extend(result)
                else:
                    print(f"Source error: {result}")
        
        return all_chunks

# Usage
batcher = DataSourceBatcher(sources)
chunks = await batcher.batch_get_chunks("machine learning")
```

### **5. Memory Optimization**

#### **Streaming Processing**

```python
class StreamingProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    async def process_large_document(self, file_path: str):
        """Process large documents in streaming fashion."""
        chunks = []
        
        async with aiofiles.open(file_path, 'r') as file:
            buffer = ""
            
            async for line in file:
                buffer += line
                
                if len(buffer) >= self.chunk_size:
                    # Process buffer
                    chunk = await self._process_buffer(buffer)
                    chunks.append(chunk)
                    
                    # Clear buffer
                    buffer = ""
                    
                    # Yield chunk to avoid memory buildup
                    yield chunk
        
        # Process remaining buffer
        if buffer:
            chunk = await self._process_buffer(buffer)
            chunks.append(chunk)
            yield chunk
    
    async def _process_buffer(self, buffer: str):
        """Process a buffer of text."""
        # Simple chunking by sentences
        sentences = buffer.split('.')
        return ContextChunk(
            id=uuid4(),
            content=buffer,
            source=ContextSource(id=uuid4(), name="streaming", source_type=SourceType.DOCUMENT),
            relevance_score=RelevanceScore(score=0.0)
        )

# Usage
processor = StreamingProcessor(chunk_size=1000)
async for chunk in processor.process_large_document("large_file.txt"):
    # Process chunk without loading entire file
    await vector_db.store_chunks([chunk])
```

#### **Memory Monitoring**

```python
import psutil
import gc
from typing import Dict, Any

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        memory_usage = self.get_memory_usage()
        return memory_usage['percent'] > (self.memory_threshold * 100)
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear_old_entries()
        
        print(f"Memory cleanup completed. Usage: {self.get_memory_usage()['percent']:.1f}%")

# Usage
memory_monitor = MemoryMonitor()

async def memory_aware_processing():
    """Process data with memory monitoring."""
    for i in range(1000):
        # Process data
        await process_data_chunk(i)
        
        # Check memory usage
        if memory_monitor.should_cleanup():
            memory_monitor.cleanup_memory()
```

### **6. Concurrency Optimization**

#### **Async Task Management**

```python
import asyncio
from asyncio import Semaphore, Task
from typing import List, Callable

class ConcurrencyManager:
    def __init__(self, max_concurrent: int = 100):
        self.semaphore = Semaphore(max_concurrent)
        self.active_tasks: List[Task] = []
    
    async def execute_with_limit(self, coro_func: Callable, *args, **kwargs):
        """Execute coroutine with concurrency limit."""
        async with self.semaphore:
            task = asyncio.create_task(coro_func(*args, **kwargs))
            self.active_tasks.append(task)
            
            try:
                result = await task
                return result
            finally:
                self.active_tasks.remove(task)
    
    async def execute_batch(self, coro_funcs: List[Callable], *args, **kwargs):
        """Execute multiple coroutines with concurrency limit."""
        tasks = []
        
        for func in coro_funcs:
            task = self.execute_with_limit(func, *args, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_active_count(self) -> int:
        """Get number of active tasks."""
        return len(self.active_tasks)
    
    async def wait_for_completion(self):
        """Wait for all active tasks to complete."""
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

# Usage
concurrency_manager = ConcurrencyManager(max_concurrent=50)

async def process_with_concurrency_limit():
    """Process data with concurrency limits."""
    coro_funcs = [process_chunk for chunk in chunks]
    results = await concurrency_manager.execute_batch(coro_funcs)
    return results
```

#### **Load Balancing**

```python
class LoadBalancer:
    def __init__(self, sources: List[BaseDataSource]):
        self.sources = sources
        self.source_weights = {source.id: 1.0 for source in sources}
        self.source_stats = {source.id: {'requests': 0, 'errors': 0} for source in sources}
    
    def select_source(self, query: str) -> BaseDataSource:
        """Select optimal source based on load balancing."""
        # Simple round-robin with weights
        total_weight = sum(self.source_weights.values())
        random_value = random.uniform(0, total_weight)
        
        current_weight = 0
        for source in self.sources:
            current_weight += self.source_weights[source.id]
            if random_value <= current_weight:
                return source
        
        return self.sources[0]  # Fallback
    
    async def get_chunks_balanced(self, query: str, max_chunks: int = 10):
        """Get chunks using load balancing."""
        source = self.select_source(query)
        
        try:
            chunks = await source.get_chunks(query, max_chunks)
            self.source_stats[source.id]['requests'] += 1
            return chunks
        except Exception as e:
            self.source_stats[source.id]['errors'] += 1
            # Reduce weight for failed source
            self.source_weights[source.id] *= 0.8
            raise e
    
    def get_source_stats(self) -> Dict[str, Dict]:
        """Get source statistics."""
        return self.source_stats

# Usage
load_balancer = LoadBalancer(sources)
chunks = await load_balancer.get_chunks_balanced("machine learning")
```

## 📈 **Performance Monitoring**

### **1. Real-time Metrics**

```python
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PerformanceMetric:
    operation: str
    duration: float
    timestamp: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class PerformanceTracker:
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.start_time = time.time()
    
    def record_metric(self, operation: str, duration: float, success: bool, 
                     error: str = None, metadata: Dict = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            success=success,
            error=error,
            metadata=metadata
        )
        self.metrics.append(metric)
    
    def get_summary(self, operation: str = None) -> Dict:
        """Get performance summary."""
        filtered_metrics = self.metrics
        if operation:
            filtered_metrics = [m for m in self.metrics if m.operation == operation]
        
        if not filtered_metrics:
            return {}
        
        successful_metrics = [m for m in filtered_metrics if m.success]
        failed_metrics = [m for m in filtered_metrics if not m.success]
        
        return {
            'total_operations': len(filtered_metrics),
            'successful_operations': len(successful_metrics),
            'failed_operations': len(failed_metrics),
            'success_rate': len(successful_metrics) / len(filtered_metrics),
            'avg_duration': sum(m.duration for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0,
            'min_duration': min(m.duration for m in filtered_metrics),
            'max_duration': max(m.duration for m in filtered_metrics),
            'p95_duration': self._percentile([m.duration for m in successful_metrics], 95),
            'p99_duration': self._percentile([m.duration for m in successful_metrics], 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def export_metrics(self, filename: str):
        """Export metrics to file."""
        import json
        
        with open(filename, 'w') as f:
            json.dump([m.__dict__ for m in self.metrics], f, indent=2)

# Usage
tracker = PerformanceTracker()

async def tracked_operation():
    start_time = time.time()
    try:
        result = await some_operation()
        duration = time.time() - start_time
        tracker.record_metric("some_operation", duration, True)
        return result
    except Exception as e:
        duration = time.time() - start_time
        tracker.record_metric("some_operation", duration, False, str(e))
        raise

# Get summary
summary = tracker.get_summary("some_operation")
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average duration: {summary['avg_duration']:.3f}s")
print(f"P95 duration: {summary['p95_duration']:.3f}s")
```

### **2. Health Checks**

```python
class HealthChecker:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.health_metrics = {}
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': time.time()
        }
        
        # Check data sources
        sources_health = await self._check_sources()
        health_status['checks']['sources'] = sources_health
        
        # Check vector database
        vector_db_health = await self._check_vector_db()
        health_status['checks']['vector_db'] = vector_db_health
        
        # Check cache
        cache_health = await self._check_cache()
        health_status['checks']['cache'] = cache_health
        
        # Check memory usage
        memory_health = self._check_memory()
        health_status['checks']['memory'] = memory_health
        
        # Determine overall status
        all_healthy = all(
            check.get('status') == 'healthy' 
            for check in health_status['checks'].values()
        )
        
        health_status['overall_status'] = 'healthy' if all_healthy else 'degraded'
        
        return health_status
    
    async def _check_sources(self) -> Dict[str, Any]:
        """Check health of all data sources."""
        sources = await self.orchestrator.list_sources()
        source_health = {}
        
        for source in sources:
            try:
                start_time = time.time()
                chunks = await source.get_chunks("test", max_chunks=1)
                duration = time.time() - start_time
                
                source_health[source.name] = {
                    'status': 'healthy',
                    'response_time': duration,
                    'chunks_retrieved': len(chunks)
                }
            except Exception as e:
                source_health[source.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return source_health
    
    async def _check_vector_db(self) -> Dict[str, Any]:
        """Check vector database health."""
        try:
            start_time = time.time()
            results = await self.orchestrator.vector_db.search_similar("test", top_k=1)
            duration = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': duration,
                'results_count': len(results)
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # Test cache operations
            test_key = "health_check"
            test_value = {"test": "data"}
            
            await self.orchestrator.cache_manager.set(test_key, test_value, ttl=60)
            retrieved_value = await self.orchestrator.cache_manager.get(test_key)
            
            return {
                'status': 'healthy',
                'hit_rate': self.orchestrator.cache_manager.get_hit_rate(),
                'test_passed': retrieved_value == test_value
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_usage_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        return {
            'status': 'healthy' if memory_percent < 80 else 'warning',
            'usage_mb': memory_usage_mb,
            'usage_percent': memory_percent
        }

# Usage
health_checker = HealthChecker(orchestrator)
health_status = await health_checker.check_health()
print(f"Overall status: {health_status['overall_status']}")
```

## 🎯 **Performance Tuning**

### **1. Configuration Optimization**

```python
class PerformanceOptimizer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def optimize_configuration(self, workload_type: str) -> OrchestratorConfig:
        """Optimize configuration for specific workload."""
        if workload_type == "high_throughput":
            return OrchestratorConfig(
                max_concurrent_sources=20,
                cache_type="redis",
                cache_ttl=1800,
                vector_db_type="faiss",
                enable_caching=True,
                request_timeout=15.0,
                retry_attempts=2
            )
        elif workload_type == "low_latency":
            return OrchestratorConfig(
                max_concurrent_sources=5,
                cache_type="memory",
                cache_ttl=3600,
                vector_db_type="chromadb",
                enable_caching=True,
                request_timeout=5.0,
                retry_attempts=1
            )
        elif workload_type == "memory_efficient":
            return OrchestratorConfig(
                max_concurrent_sources=3,
                cache_type="file",
                cache_ttl=7200,
                vector_db_type="faiss",
                enable_caching=False,
                request_timeout=30.0,
                retry_attempts=3
            )
        else:
            return OrchestratorConfig()  # Default configuration
    
    async def auto_tune(self, test_queries: List[str]):
        """Automatically tune performance based on test queries."""
        # Test different configurations
        configs = [
            self.optimize_configuration("high_throughput"),
            self.optimize_configuration("low_latency"),
            self.optimize_configuration("memory_efficient")
        ]
        
        best_config = None
        best_performance = 0
        
        for config in configs:
            # Apply configuration
            self.orchestrator.config = config
            
            # Test performance
            start_time = time.time()
            for query in test_queries:
                request = ContextRequest(query=query, max_chunks=10)
                await self.orchestrator.get_context(request)
            
            end_time = time.time()
            performance = len(test_queries) / (end_time - start_time)
            
            if performance > best_performance:
                best_performance = performance
                best_config = config
        
        return best_config

# Usage
optimizer = PerformanceOptimizer(orchestrator)
best_config = await optimizer.auto_tune(test_queries)
print(f"Best configuration: {best_config}")
```

---

## 📚 **Next Steps**

- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Configuration](configuration.md)** - Advanced configuration options
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples.md)** - Code examples and tutorials
