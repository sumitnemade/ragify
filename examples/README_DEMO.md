# Ragify Examples

This directory contains focused examples demonstrating each core feature of the Ragify plugin.

## Core Feature Examples

### 1. **Basic Usage** (`basic_usage.py`)
- **Purpose**: Get started with Ragify quickly
- **Features**: Basic orchestrator setup, simple context retrieval
- **Use Case**: First-time users, basic integration

### 2. **Document Processing** (`document_processing_demo.py`)
- **Purpose**: Process and extract context from documents
- **Features**: File parsing, text chunking, metadata extraction
- **Use Case**: Document analysis, content extraction

### 3. **Vector Database Operations** (`vector_db_demo.py`)
- **Purpose**: Store and search vector embeddings
- **Features**: Multiple vector DB backends, similarity search, indexing
- **Use Case**: Semantic search, recommendation systems

### 4. **Intelligent Fusion** (`intelligent_fusion_demo.py`)
- **Purpose**: Merge context from multiple sources intelligently
- **Features**: Conflict detection, resolution strategies, confidence scoring
- **Use Case**: Multi-source data integration, conflict resolution

### 5. **Multi-Factor Scoring** (`multi_factor_scoring_demo.py`)
- **Purpose**: Advanced relevance scoring with multiple factors
- **Features**: Semantic similarity, temporal relevance, source authority
- **Use Case**: Content ranking, personalized recommendations

### 6. **API & Database Integration** (`api_database_integrations_demo.py`)
- **Purpose**: Connect to external APIs and databases
- **Features**: REST API integration, database queries, data transformation
- **Use Case**: External data sources, real-time data integration

### 7. **Real-time Synchronization** (`realtime_sync_demo.py`)
- **Purpose**: Handle real-time data streams and updates
- **Features**: WebSocket connections, event-driven updates, live data
- **Use Case**: Live dashboards, real-time monitoring

### 8. **Cache Management** (`cache_management_demo.py`)
- **Purpose**: Optimize performance with intelligent caching
- **Features**: Multi-level caching, TTL management, cache invalidation
- **Use Case**: Performance optimization, reduced API calls

## Running Examples

```bash
# Set Python path to include src directory
export PYTHONPATH=src

# Run any example
python examples/basic_usage.py
python examples/vector_db_demo.py
python examples/intelligent_fusion_demo.py
```

## Example Structure

Each example follows a consistent pattern:
1. **Setup**: Initialize orchestrator and sources
2. **Configuration**: Configure specific features
3. **Demonstration**: Show core functionality
4. **Cleanup**: Proper resource management

## Dependencies

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- Examples use in-memory storage for simplicity
- Production deployments should use persistent storage
- Some examples may require external services (APIs, databases)
- All examples include error handling and logging
