# 📊 **Data Sources**

Ragify supports multiple types of data sources, each designed to handle specific data formats and access patterns. This document explains each source type and how to configure them.

## 🎯 **Overview**

```mermaid
graph TB
    subgraph "Data Sources"
        A[Document Sources] --> A1[PDF Files]
        A --> A2[DOCX Files]
        A --> A3[TXT Files]
        A --> A4[Markdown Files]
        
        B[API Sources] --> B1[REST APIs]
        B --> B2[GraphQL APIs]
        B --> B3[SOAP APIs]
        
        C[Database Sources] --> C1[PostgreSQL]
        C --> C2[MySQL]
        C --> C3[SQLite]
        C --> C4[MongoDB]
        
        D[Real-time Sources] --> D1[WebSocket]
        D --> D2[MQTT]
        D --> D3[Kafka]
        D --> D4[Redis Pub/Sub]
    end
    
    subgraph "Processing"
        E[Text Extraction]
        F[Chunking]
        G[Metadata Extraction]
        H[Relevance Scoring]
    end
    
    A1 --> E
    A2 --> E
    B1 --> E
    C1 --> E
    D1 --> E
    
    E --> F
    F --> G
    G --> H
```

## 📄 **Document Sources**

Document sources handle various file formats and extract text content for processing.

### **Supported Formats**

| Format | Extension | Processor | Features |
|--------|-----------|-----------|----------|
| PDF | `.pdf` | PyPDF2, pdfplumber | Text, tables, images |
| Word | `.docx` | python-docx | Text, formatting, tables |
| Word (Legacy) | `.doc` | docx2txt | Text extraction |
| Text | `.txt` | Built-in | Plain text |
| Markdown | `.md` | Built-in | Formatted text |

### **Configuration**

```python
from ragify.sources import DocumentSource
from ragify.models import SourceType

# Basic document source
doc_source = DocumentSource(
    name="company_docs",
    source_type=SourceType.DOCUMENT,
    url="/path/to/documents",
    chunk_size=1000,
    overlap=200
)

# Advanced configuration
doc_source = DocumentSource(
    name="research_papers",
    source_type=SourceType.DOCUMENT,
    url="/research/papers",
    chunk_size=1500,
    overlap=300,
    supported_formats=['.pdf', '.docx'],
    metadata_extraction=True,
    table_extraction=True
)
```

### **Usage Example**

```python
import asyncio
from ragify.sources import DocumentSource
from ragify import ContextOrchestrator, ContextRequest

async def process_documents():
    # Create document source
    doc_source = DocumentSource(
        name="technical_docs",
        url="./docs",
        chunk_size=1000,
        overlap=200
    )
    
    # Add to orchestrator
    orchestrator = ContextOrchestrator()
    orchestrator.add_source(doc_source)
    
    # Query documents
    request = ContextRequest(
        query="API documentation for authentication",
        user_id="developer123"
    )
    
    response = await orchestrator.get_context(request)
    
    # Process results
    for chunk in response.chunks:
        print(f"Source: {chunk.source.name}")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Relevance: {chunk.relevance_score.score:.3f}")
        print("---")

# Run example
asyncio.run(process_documents())
```

### **Document Processing Flow**

```mermaid
graph LR
    A[Document File] --> B[Format Detection]
    B --> C[Text Extraction]
    C --> D[Content Cleaning]
    D --> E[Chunking]
    E --> F[Metadata Extraction]
    F --> G[Context Chunks]
```

## 🌐 **API Sources**

API sources connect to external REST APIs and other web services to fetch real-time data.

### **Supported Authentication Types**

| Type | Description | Configuration |
|------|-------------|---------------|
| None | No authentication | `auth_type="none"` |
| Basic | Username/password | `auth_config={"username": "user", "password": "pass"}` |
| Bearer | Token-based | `auth_config={"token": "your_token"}` |
| API Key | Key-based | `auth_config={"api_key": "your_key"}` |
| OAuth2 | OAuth 2.0 | `auth_config={"client_id": "id", "client_secret": "secret"}` |

### **Configuration**

```python
from ragify.sources import APISource

# Basic API source
api_source = APISource(
    name="weather_api",
    source_type=SourceType.API,
    url="https://api.weatherapi.com/v1/current.json",
    auth_type="api_key",
    auth_config={"api_key": "your_api_key"}
)

# Advanced API source
api_source = APISource(
    name="news_api",
    source_type=SourceType.API,
    url="https://newsapi.org/v2/everything",
    auth_type="api_key",
    auth_config={"api_key": "your_api_key"},
    headers={
        "Accept": "application/json",
        "User-Agent": "Ragify/1.0"
    },
    rate_limit=100,  # requests per minute
    timeout=30,
    retry_attempts=3
)
```

### **Usage Example**

```python
async def fetch_api_data():
    # Create API source
    api_source = APISource(
        name="github_api",
        url="https://api.github.com/search/repositories",
        auth_type="bearer",
        auth_config={"token": "your_github_token"},
        headers={"Accept": "application/vnd.github.v3+json"}
    )
    
    # Add to orchestrator
    orchestrator = ContextOrchestrator()
    orchestrator.add_source(api_source)
    
    # Query API
    request = ContextRequest(
        query="Python machine learning repositories",
        user_id="developer123"
    )
    
    response = await orchestrator.get_context(request)
    
    # Process results
    for chunk in response.chunks:
        print(f"Repository: {chunk.metadata.get('name', 'Unknown')}")
        print(f"Description: {chunk.content[:100]}...")
        print(f"Stars: {chunk.metadata.get('stargazers_count', 0)}")
        print("---")

# Run example
asyncio.run(fetch_api_data())
```

### **API Request Flow**

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant A as API Source
    participant E as External API
    
    U->>O: Context Request
    O->>A: Query API
    A->>E: HTTP Request
    E->>A: API Response
    A->>O: Processed Data
    O->>U: Context Response
```

## 🗄️ **Database Sources**

Database sources connect to various database systems to query structured data.

### **Supported Databases**

| Database | Driver | Connection String Format |
|----------|--------|-------------------------|
| PostgreSQL | asyncpg | `postgresql://user:pass@host:port/db` |
| MySQL | aiomysql | `mysql://user:pass@host:port/db` |
| SQLite | aiosqlite | `sqlite:///path/to/file.db` |
| MongoDB | motor | `mongodb://user:pass@host:port/db` |

### **Configuration**

```python
from ragify.sources import DatabaseSource

# PostgreSQL source
pg_source = DatabaseSource(
    name="user_database",
    source_type=SourceType.DATABASE,
    url="postgresql://user:password@localhost:5432/mydb",
    db_type="postgresql",
    query_template="""
        SELECT content, relevance, created_at 
        FROM documents 
        WHERE content ILIKE '%{query}%'
        ORDER BY relevance DESC
        LIMIT 10
    """
)

# MongoDB source
mongo_source = DatabaseSource(
    name="product_database",
    source_type=SourceType.DATABASE,
    url="mongodb://localhost:27017/products",
    db_type="mongodb",
    query_template="""
        db.products.find({
            "$text": {"$search": "{query}"}
        }).limit(10)
    """
)
```

### **Usage Example**

```python
async def query_database():
    # Create database source
    db_source = DatabaseSource(
        name="customer_db",
        url="postgresql://user:pass@localhost:5432/customers",
        db_type="postgresql",
        query_template="""
            SELECT 
                customer_name,
                email,
                preferences,
                created_at
            FROM customers 
            WHERE customer_name ILIKE '%{query}%'
            OR email ILIKE '%{query}%'
            LIMIT 5
        """
    )
    
    # Add to orchestrator
    orchestrator = ContextOrchestrator()
    orchestrator.add_source(db_source)
    
    # Query database
    request = ContextRequest(
        query="john smith",
        user_id="admin123"
    )
    
    response = await orchestrator.get_context(request)
    
    # Process results
    for chunk in response.chunks:
        print(f"Customer: {chunk.metadata.get('customer_name', 'Unknown')}")
        print(f"Email: {chunk.metadata.get('email', 'N/A')}")
        print(f"Content: {chunk.content[:100]}...")
        print("---")

# Run example
asyncio.run(query_database())
```

### **Database Query Flow**

```mermaid
graph LR
    A[Query] --> B[Template Processing]
    B --> C[Database Connection]
    C --> D[Execute Query]
    D --> E[Result Processing]
    E --> F[Context Chunks]
```

## ⚡ **Real-time Sources**

Real-time sources provide live data streams and real-time updates.

### **Supported Protocols**

| Protocol | Description | Use Case |
|----------|-------------|----------|
| WebSocket | Bidirectional communication | Live chat, real-time dashboards |
| MQTT | Lightweight messaging | IoT devices, sensors |
| Kafka | High-throughput streaming | Event processing, analytics |
| Redis Pub/Sub | Fast messaging | Caching, notifications |

### **Configuration**

```python
from ragify.sources import RealtimeSource

# WebSocket source
ws_source = RealtimeSource(
    name="live_chat",
    source_type=SourceType.REALTIME,
    url="ws://localhost:8080/chat",
    connection_type="websocket",
    auto_reconnect=True,
    heartbeat_interval=30
)

# MQTT source
mqtt_source = RealtimeSource(
    name="sensor_data",
    source_type=SourceType.REALTIME,
    url="mqtt://localhost:1883",
    connection_type="mqtt",
    topics=["sensors/temperature", "sensors/humidity"],
    qos=1
)

# Kafka source
kafka_source = RealtimeSource(
    name="event_stream",
    source_type=SourceType.REALTIME,
    url="localhost:9092",
    connection_type="kafka",
    topics=["user_events", "system_events"],
    group_id="ragify_consumer"
)
```

### **Usage Example**

```python
async def handle_realtime_data():
    # Create real-time source
    realtime_source = RealtimeSource(
        name="stock_prices",
        url="ws://localhost:8080/stocks",
        connection_type="websocket"
    )
    
    # Add to orchestrator
    orchestrator = ContextOrchestrator()
    orchestrator.add_source(realtime_source)
    
    # Subscribe to updates
    async def handle_stock_update(data):
        print(f"Stock Update: {data}")
    
    orchestrator.updates_engine.subscribe_to_updates(
        source_name="stock_prices",
        callback=handle_stock_update
    )
    
    # Query real-time data
    request = ContextRequest(
        query="current stock prices",
        user_id="trader123"
    )
    
    response = await orchestrator.get_context(request)
    
    # Process results
    for chunk in response.chunks:
        print(f"Stock: {chunk.metadata.get('symbol', 'Unknown')}")
        print(f"Price: {chunk.metadata.get('price', 'N/A')}")
        print(f"Timestamp: {chunk.metadata.get('timestamp', 'N/A')}")
        print("---")

# Run example
asyncio.run(handle_realtime_data())
```

### **Real-time Data Flow**

```mermaid
sequenceDiagram
    participant S as Source
    participant R as Realtime Source
    participant O as Orchestrator
    participant U as User
    
    S->>R: Data Stream
    R->>O: Process Update
    O->>U: Notify User
    
    U->>O: Query Request
    O->>R: Get Current Data
    R->>O: Latest Data
    O->>U: Context Response
```

## 🔧 **Advanced Configuration**

### **1. Custom Source Configuration**

```python
# Custom document source with specific settings
custom_doc_source = DocumentSource(
    name="legal_documents",
    url="/legal/docs",
    chunk_size=2000,  # Larger chunks for legal documents
    overlap=400,
    metadata_extraction=True,
    table_extraction=True,
    image_extraction=False,  # Disable image processing
    custom_processors={
        'pre_process': lambda text: text.upper(),
        'post_process': lambda chunks: [c for c in chunks if len(c.content) > 100]
    }
)
```

### **2. Source Chaining**

```python
# Chain multiple sources for comprehensive results
sources = [
    DocumentSource(name="docs", url="/docs"),
    APISource(name="api", url="https://api.example.com"),
    DatabaseSource(name="db", url="postgresql://..."),
    RealtimeSource(name="live", url="ws://...")
]

# Add all sources to orchestrator
for source in sources:
    orchestrator.add_source(source)

# Fusion engine will automatically combine results
```

### **3. Source Filtering**

```python
# Filter sources based on query
async def get_context_with_filtering(query, user_id, source_types=None):
    request = ContextRequest(
        query=query,
        user_id=user_id,
        source_types=source_types  # Only use specified source types
    )
    
    response = await orchestrator.get_context(request)
    return response

# Use only document sources
doc_response = await get_context_with_filtering(
    "Python documentation",
    "user123",
    source_types=[SourceType.DOCUMENT]
)

# Use only API sources
api_response = await get_context_with_filtering(
    "Weather forecast",
    "user123",
    source_types=[SourceType.API]
)
```

## 📊 **Performance Optimization**

### **1. Connection Pooling**

```python
# Configure connection pooling for database sources
db_source = DatabaseSource(
    name="high_traffic_db",
    url="postgresql://user:pass@localhost:5432/db",
    db_type="postgresql",
    pool_size=20,  # Connection pool size
    max_overflow=30,  # Additional connections
    pool_timeout=30,  # Connection timeout
    pool_recycle=3600  # Recycle connections every hour
)
```

### **2. Caching**

```python
# Enable caching for API sources
api_source = APISource(
    name="cached_api",
    url="https://api.example.com",
    cache_enabled=True,
    cache_ttl=300,  # Cache for 5 minutes
    cache_key_template="api:{query}:{user_id}"
)
```

### **3. Rate Limiting**

```python
# Configure rate limiting for API sources
api_source = APISource(
    name="rate_limited_api",
    url="https://api.example.com",
    rate_limit=100,  # 100 requests per minute
    rate_limit_window=60,  # 1 minute window
    retry_attempts=3,
    retry_delay=1  # 1 second delay between retries
)
```

## 🔍 **Monitoring and Debugging**

### **1. Source Health Checks**

```python
# Check source health
async def check_source_health():
    for source_name, source in orchestrator.sources.items():
        try:
            health = await source.health_check()
            print(f"{source_name}: {health.status}")
        except Exception as e:
            print(f"{source_name}: ERROR - {e}")

# Run health check
asyncio.run(check_source_health())
```

### **2. Source Metrics**

```python
# Get source metrics
async def get_source_metrics():
    for source_name, source in orchestrator.sources.items():
        metrics = await source.get_metrics()
        print(f"{source_name}:")
        print(f"  Requests: {metrics.get('requests', 0)}")
        print(f"  Errors: {metrics.get('errors', 0)}")
        print(f"  Response Time: {metrics.get('avg_response_time', 0):.2f}ms")
        print("---")

# Get metrics
asyncio.run(get_source_metrics())
```

---

## 📚 **Next Steps**

- **[Context Fusion](context-fusion.md)** - Learn how data from multiple sources is combined
- **[Scoring System](scoring.md)** - Understand how relevance is calculated
- **[Configuration](configuration.md)** - Advanced configuration options
- **[API Reference](api-reference.md)** - Complete API documentation
