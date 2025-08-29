# 📚 **API Reference**

This document provides a complete reference for all Ragify APIs, classes, methods, and configuration options.

## 🎯 **Overview**

Ragify provides a comprehensive API for intelligent context orchestration with multiple data sources, fusion engines, and storage backends.

```mermaid
graph TB
    subgraph "Core API"
        A[ContextOrchestrator] --> B[get_context]
        A --> C[add_source]
        A --> D[remove_source]
        A --> E[update_source]
    end
    
    subgraph "Engines"
        F[IntelligentContextFusionEngine]
        G[ContextScoringEngine]
        H[ContextStorageEngine]
        I[ContextUpdatesEngine]
    end
    
    subgraph "Sources"
        J[DocumentSource]
        K[APISource]
        L[DatabaseSource]
        M[RealtimeSource]
    end
    
    subgraph "Storage"
        N[VectorDatabase]
        O[CacheManager]
        P[PrivacyManager]
    end
```

## 🏗️ **Core Classes**

### **ContextOrchestrator**

The main orchestrator class that manages all context operations.

```python
class ContextOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        fusion_engine: Optional[IntelligentContextFusionEngine] = None,
        scoring_engine: Optional[ContextScoringEngine] = None,
        storage_engine: Optional[ContextStorageEngine] = None,
        updates_engine: Optional[ContextUpdatesEngine] = None
    ):
        """
        Initialize the context orchestrator.
        
        Args:
            config: Orchestrator configuration
            fusion_engine: Optional custom fusion engine
            scoring_engine: Optional custom scoring engine
            storage_engine: Optional custom storage engine
            updates_engine: Optional custom updates engine
        """
```

#### **Methods**

##### **get_context**

```python
async def get_context(
    self,
    request: ContextRequest,
    user_context: Optional[Dict[str, Any]] = None
) -> ContextResponse:
    """
    Retrieve context based on the request.
    
    Args:
        request: Context request containing query and parameters
        user_context: Optional user context for personalization
    
    Returns:
        ContextResponse with relevant chunks and metadata
    
    Raises:
        ContextRetrievalError: If context retrieval fails
        ValidationError: If request validation fails
    """
```

**Example:**
```python
from ragify import ContextOrchestrator, ContextRequest

# Initialize orchestrator
orchestrator = ContextOrchestrator(config)

# Create request
request = ContextRequest(
    query="machine learning algorithms",
    max_chunks=10,
    min_relevance=0.7
)

# Get context
response = await orchestrator.get_context(request)
print(f"Found {len(response.chunks)} relevant chunks")
```

##### **add_source**

```python
async def add_source(
    self,
    source: BaseDataSource,
    priority: int = 0
) -> None:
    """
    Add a new data source to the orchestrator.
    
    Args:
        source: Data source to add
        priority: Source priority (higher = more important)
    
    Raises:
        SourceAlreadyExistsError: If source with same ID already exists
        ValidationError: If source validation fails
    """
```

##### **remove_source**

```python
async def remove_source(self, source_id: str) -> None:
    """
    Remove a data source from the orchestrator.
    
    Args:
        source_id: ID of the source to remove
    
    Raises:
        SourceNotFoundError: If source doesn't exist
    """
```

##### **update_source**

```python
async def update_source(
    self,
    source_id: str,
    config: SourceConfig
) -> None:
    """
    Update an existing data source configuration.
    
    Args:
        source_id: ID of the source to update
        config: New source configuration
    
    Raises:
        SourceNotFoundError: If source doesn't exist
        ValidationError: If configuration validation fails
    """
```

##### **list_sources**

```python
async def list_sources(self) -> List[BaseDataSource]:
    """
    List all registered data sources.
    
    Returns:
        List of registered data sources
    """
```

##### **get_source_status**

```python
async def get_source_status(self, source_id: str) -> Dict[str, Any]:
    """
    Get status information for a specific source.
    
    Args:
        source_id: ID of the source
    
    Returns:
        Dictionary with source status information
    
    Raises:
        SourceNotFoundError: If source doesn't exist
    """
```

## 🔄 **Fusion Engine API**

### **IntelligentContextFusionEngine**

```python
class IntelligentContextFusionEngine:
    def __init__(
        self,
        conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
        enable_semantic_analysis: bool = True,
        enable_factual_verification: bool = True,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the intelligent context fusion engine.
        
        Args:
            conflict_resolution_strategy: Strategy for resolving conflicts
            enable_semantic_analysis: Enable semantic conflict detection
            enable_factual_verification: Enable factual verification
            similarity_threshold: Threshold for similarity detection
        """
```

#### **Methods**

##### **fuse_chunks**

```python
async def fuse_chunks(
    self,
    chunks: List[ContextChunk],
    user_context: Optional[Dict[str, Any]] = None
) -> List[ContextChunk]:
    """
    Fuse multiple context chunks intelligently.
    
    Args:
        chunks: List of context chunks to fuse
        user_context: Optional user context for personalization
    
    Returns:
        List of fused context chunks
    """
```

##### **detect_conflicts**

```python
async def detect_conflicts(
    self,
    chunks: List[ContextChunk]
) -> List[ConflictInfo]:
    """
    Detect conflicts between context chunks.
    
    Args:
        chunks: List of context chunks to analyze
    
    Returns:
        List of detected conflicts
    """
```

##### **resolve_conflicts**

```python
async def resolve_conflicts(
    self,
    conflicts: List[ConflictInfo],
    strategy: Optional[ConflictResolutionStrategy] = None
) -> List[ContextChunk]:
    """
    Resolve conflicts using specified strategy.
    
    Args:
        conflicts: List of conflicts to resolve
        strategy: Resolution strategy (uses default if None)
    
    Returns:
        List of resolved context chunks
    """
```

## 🎯 **Scoring Engine API**

### **ContextScoringEngine**

```python
class ContextScoringEngine:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        ensemble_method: str = 'weighted_average',
        enable_confidence_bounds: bool = True,
        confidence_level: float = 0.95
    ):
        """
        Initialize the context scoring engine.
        
        Args:
            weights: Factor weights for scoring
            ensemble_method: Method for combining factor scores
            enable_confidence_bounds: Enable confidence interval calculation
            confidence_level: Confidence level for bounds
        """
```

#### **Methods**

##### **calculate_score**

```python
async def calculate_score(
    self,
    chunk: ContextChunk,
    query: str,
    user_context: Optional[Dict[str, Any]] = None
) -> RelevanceScore:
    """
    Calculate relevance score for a context chunk.
    
    Args:
        chunk: Context chunk to score
        query: Query string
        user_context: Optional user context
    
    Returns:
        RelevanceScore with score and confidence bounds
    """
```

##### **calculate_batch_scores**

```python
async def calculate_batch_scores(
    self,
    chunks: List[ContextChunk],
    query: str,
    user_context: Optional[Dict[str, Any]] = None
) -> List[RelevanceScore]:
    """
    Calculate scores for multiple chunks efficiently.
    
    Args:
        chunks: List of context chunks
        query: Query string
        user_context: Optional user context
    
    Returns:
        List of relevance scores
    """
```

##### **get_factor_breakdown**

```python
async def get_factor_breakdown(
    self,
    chunk: ContextChunk,
    query: str,
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Get detailed breakdown of factor scores.
    
    Args:
        chunk: Context chunk
        query: Query string
        user_context: Optional user context
    
    Returns:
        Dictionary with individual factor scores
    """
```

## 📊 **Data Sources API**

### **BaseDataSource**

```python
class BaseDataSource(ABC):
    def __init__(self, config: SourceConfig):
        """
        Initialize base data source.
        
        Args:
            config: Source configuration
        """
        self.config = config
        self.id = config.id
        self.name = config.name
        self.source_type = config.source_type
        self.privacy_level = config.privacy_level
```

#### **Abstract Methods**

##### **get_chunks**

```python
@abstractmethod
async def get_chunks(
    self,
    query: str,
    max_chunks: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[ContextChunk]:
    """
    Retrieve context chunks from this source.
    
    Args:
        query: Search query
        max_chunks: Maximum number of chunks to return
        filters: Optional filters to apply
    
    Returns:
        List of context chunks
    
    Raises:
        SourceError: If chunk retrieval fails
    """
```

##### **refresh**

```python
@abstractmethod
async def refresh(self) -> None:
    """
    Refresh the data source (e.g., reload documents, refresh connections).
    
    Raises:
        SourceError: If refresh fails
    """
```

##### **close**

```python
@abstractmethod
async def close(self) -> None:
    """
    Close the data source and clean up resources.
    
    Raises:
        SourceError: If close fails
    """
```

### **DocumentSource**

```python
class DocumentSource(BaseDataSource):
    def __init__(
        self,
        config: SourceConfig,
        document_paths: List[str],
        supported_formats: Optional[List[str]] = None
    ):
        """
        Initialize document source.
        
        Args:
            config: Source configuration
            document_paths: List of document file paths
            supported_formats: List of supported file formats
        """
```

**Example:**
```python
from ragify.sources import DocumentSource
from ragify.models import SourceConfig, SourceType, PrivacyLevel

config = SourceConfig(
    id="doc_source_1",
    name="Technical Documentation",
    source_type=SourceType.DOCUMENT,
    privacy_level=PrivacyLevel.PUBLIC
)

doc_source = DocumentSource(
    config=config,
    document_paths=["docs/api.md", "docs/guide.pdf"],
    supported_formats=["md", "pdf", "docx", "txt"]
)

# Add to orchestrator
await orchestrator.add_source(doc_source)
```

### **APISource**

```python
class APISource(BaseDataSource):
    def __init__(
        self,
        config: SourceConfig,
        base_url: str,
        auth_type: str = "none",
        auth_config: Optional[Dict[str, Any]] = None,
        rate_limit: Optional[int] = None
    ):
        """
        Initialize API source.
        
        Args:
            config: Source configuration
            base_url: Base URL for API
            auth_type: Authentication type
            auth_config: Authentication configuration
            rate_limit: Rate limit (requests per minute)
        """
```

**Example:**
```python
from ragify.sources import APISource

api_source = APISource(
    config=config,
    base_url="https://api.example.com",
    auth_type="bearer",
    auth_config={"token": "your_token_here"},
    rate_limit=100
)
```

### **DatabaseSource**

```python
class DatabaseSource(BaseDataSource):
    def __init__(
        self,
        config: SourceConfig,
        connection_string: str,
        db_type: str = "postgresql",
        query_template: Optional[str] = None,
        connection_pool_size: int = 10
    ):
        """
        Initialize database source.
        
        Args:
            config: Source configuration
            connection_string: Database connection string
            db_type: Database type
            query_template: SQL query template
            connection_pool_size: Connection pool size
        """
```

**Example:**
```python
from ragify.sources import DatabaseSource

db_source = DatabaseSource(
    config=config,
    connection_string="postgresql://user:pass@localhost/db",
    db_type="postgresql",
    query_template="SELECT content FROM documents WHERE content ILIKE %s LIMIT %s"
)
```

### **RealtimeSource**

```python
class RealtimeSource(BaseDataSource):
    def __init__(
        self,
        config: SourceConfig,
        protocol: str = "websocket",
        endpoint: str = "",
        auth_config: Optional[Dict[str, Any]] = None,
        update_interval: float = 1.0
    ):
        """
        Initialize real-time source.
        
        Args:
            config: Source configuration
            protocol: Communication protocol
            endpoint: Endpoint URL
            auth_config: Authentication configuration
            update_interval: Update interval in seconds
        """
```

## 🗄️ **Storage API**

### **VectorDatabase**

```python
class VectorDatabase:
    def __init__(
        self,
        db_type: str = "chromadb",
        connection_string: str = "memory://",
        embedding_model: Optional[str] = None,
        dimension: int = 1536
    ):
        """
        Initialize vector database.
        
        Args:
            db_type: Vector database type
            connection_string: Connection string
            embedding_model: Embedding model name
            dimension: Vector dimension
        """
```

#### **Methods**

##### **store_chunks**

```python
async def store_chunks(
    self,
    chunks: List[ContextChunk],
    collection_name: str = "default"
) -> List[str]:
    """
    Store context chunks in vector database.
    
    Args:
        chunks: List of context chunks
        collection_name: Collection name
    
    Returns:
        List of stored chunk IDs
    """
```

##### **search_similar**

```python
async def search_similar(
    self,
    query: str,
    top_k: int = 10,
    collection_name: str = "default",
    similarity_threshold: float = 0.7
) -> List[Tuple[ContextChunk, float]]:
    """
    Search for similar chunks.
    
    Args:
        query: Search query
        top_k: Number of results to return
        collection_name: Collection name
        similarity_threshold: Minimum similarity threshold
    
    Returns:
        List of (chunk, similarity_score) tuples
    """
```

### **CacheManager**

```python
class CacheManager:
    def __init__(
        self,
        cache_type: str = "memory",
        connection_string: Optional[str] = None,
        ttl: int = 3600,
        max_size: int = 1000
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_type: Cache type (memory, redis, memcached)
            connection_string: Connection string for external cache
            ttl: Time to live in seconds
            max_size: Maximum cache size
        """
```

#### **Methods**

##### **get**

```python
async def get(self, key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None if not found
    """
```

##### **set**

```python
async def set(
    self,
    key: str,
    value: Any,
    ttl: Optional[int] = None
) -> None:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
    """
```

##### **delete**

```python
async def delete(self, key: str) -> bool:
    """
    Delete value from cache.
    
    Args:
        key: Cache key
    
    Returns:
        True if deleted, False if not found
    """
```

### **PrivacyManager**

```python
class PrivacyManager:
    def __init__(
        self,
        encryption_key: Optional[str] = None,
        anonymization_enabled: bool = True,
        pii_detection_enabled: bool = True
    ):
        """
        Initialize privacy manager.
        
        Args:
            encryption_key: Encryption key for sensitive data
            anonymization_enabled: Enable data anonymization
            pii_detection_enabled: Enable PII detection
        """
```

#### **Methods**

##### **encrypt_data**

```python
async def encrypt_data(self, data: str) -> str:
    """
    Encrypt sensitive data.
    
    Args:
        data: Data to encrypt
    
    Returns:
        Encrypted data
    """
```

##### **decrypt_data**

```python
async def decrypt_data(self, encrypted_data: str) -> str:
    """
    Decrypt encrypted data.
    
    Args:
        encrypted_data: Encrypted data
    
    Returns:
        Decrypted data
    """
```

##### **anonymize_data**

```python
async def anonymize_data(self, data: str) -> str:
    """
    Anonymize data by removing PII.
    
    Args:
        data: Data to anonymize
    
    Returns:
        Anonymized data
    """
```

## 📋 **Data Models**

### **ContextRequest**

```python
class ContextRequest(BaseModel):
    query: str
    max_chunks: int = 10
    min_relevance: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### **ContextResponse**

```python
class ContextResponse(BaseModel):
    chunks: List[ContextChunk]
    total_chunks: int
    sources: List[str]
    fusion_metadata: Optional[FusionMetadata] = None
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
```

### **ContextChunk**

```python
class ContextChunk(BaseModel):
    id: UUID
    content: str
    source: ContextSource
    relevance_score: RelevanceScore
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### **RelevanceScore**

```python
class RelevanceScore(BaseModel):
    score: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    factors: Optional[Dict[str, float]] = None
    ensemble_method: Optional[str] = None
```

### **SourceConfig**

```python
class SourceConfig(BaseModel):
    id: str
    name: str
    source_type: SourceType
    privacy_level: PrivacyLevel
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

## ⚙️ **Configuration**

### **OrchestratorConfig**

```python
class OrchestratorConfig(BaseModel):
    # Fusion settings
    fusion_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE
    enable_semantic_analysis: bool = True
    similarity_threshold: float = 0.8
    
    # Scoring settings
    scoring_weights: Dict[str, float] = Field(default_factory=lambda: DEFAULT_WEIGHTS)
    ensemble_method: str = "weighted_average"
    enable_confidence_bounds: bool = True
    
    # Storage settings
    vector_db_type: str = "chromadb"
    vector_db_url: str = "memory://"
    cache_type: str = "memory"
    cache_ttl: int = 3600
    
    # Privacy settings
    encryption_enabled: bool = False
    anonymization_enabled: bool = True
    
    # Performance settings
    max_concurrent_sources: int = 10
    request_timeout: float = 30.0
    retry_attempts: int = 3
```

## 🚨 **Error Handling**

### **Exception Classes**

```python
class RagifyError(Exception):
"""Base exception for all Ragify errors."""
    pass

class ContextRetrievalError(RagifyError):
    """Raised when context retrieval fails."""
    pass

class SourceError(RagifyError):
    """Raised when data source operations fail."""
    pass

class ValidationError(RagifyError):
    """Raised when validation fails."""
    pass

class ConfigurationError(RagifyError):
    """Raised when configuration is invalid."""
    pass
```

### **Error Handling Example**

```python
from ragify import ContextOrchestrator, ContextRetrievalError, SourceError

try:
    response = await orchestrator.get_context(request)
except ContextRetrievalError as e:
    print(f"Failed to retrieve context: {e}")
    # Handle context retrieval failure
except SourceError as e:
    print(f"Source error: {e}")
    # Handle source-specific error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## 📊 **Monitoring and Logging**

### **Structured Logging**

```python
import structlog

logger = structlog.get_logger()

# Log context retrieval
logger.info(
    "context_retrieved",
    query=request.query,
    chunks_count=len(response.chunks),
    processing_time=response.processing_time,
    sources=response.sources
)

# Log source operations
logger.info(
    "source_operation",
    source_id=source.id,
    operation="get_chunks",
    chunks_retrieved=len(chunks),
    duration=duration
)
```

### **Performance Metrics**

```python
# Track performance metrics
metrics = {
    "context_retrieval_time": response.processing_time,
    "chunks_retrieved": len(response.chunks),
    "sources_queried": len(response.sources),
    "cache_hit_rate": cache_manager.get_hit_rate(),
    "fusion_conflicts": len(fusion_metadata.conflicts) if fusion_metadata else 0
}
```

---

## 📚 **Next Steps**

- **[Examples](examples.md)** - Code examples and tutorials
- **[Configuration](configuration.md)** - Detailed configuration guide
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Performance](performance.md)** - Performance optimization guide
