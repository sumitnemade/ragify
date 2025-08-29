# Ragify - Intelligent Context Orchestration Plugin

A generic, open-source plugin for intelligent context management in LLM-powered applications.

## 🎯 Overview

Ragify is a context orchestration framework that provides a solid foundation for managing context from multiple data sources. It includes intelligent multi-source context fusion, vector database support, and a modular architecture for building advanced context management solutions.

## 🚀 Key Features

- **Intelligent Multi-Source Context Fusion**: Advanced conflict detection and resolution
- **Conflict Resolution Strategies**: Highest authority, newest data, consensus, weighted average
- **Conflict Types**: Content contradictions, factual disagreements, temporal conflicts, source authority, data freshness, semantic conflicts
- **Vector Database Support**: Full integration with ChromaDB, Pinecone, Weaviate, and FAISS
- **Multi-Factor Relevance Scoring**: Advanced relevance assessment with ensemble methods
- **Privacy Level Framework**: Configurable privacy controls (Public, Private, Restricted)
- **Modular Architecture**: Pluggable data sources and engines
- **Async Processing**: Non-blocking operations throughout
- **Structured Logging**: Comprehensive logging with structlog

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                    Ragify Core Interface                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Context   │ │   Context   │ │   Context   │ │   Context   │ │
│  │  Fusion     │ │  Scoring    │ │  Storage    │ │  Updates    │ │
│  │  Engine     │ │  Engine     │ │  Engine     │ │  Engine     │ │
│  │ (Framework) │ │ (Basic)     │ │ (Framework) │ │ (Framework) │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Vector    │ │   Cache     │ │   Privacy   │ │   Analytics │ │
│  │  Database   │ │  Manager    │ │  Manager    │ │  Engine     │ │
│  │(Framework)  │ │(Framework)  │ │ (Basic)     │ │(Framework)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Data Source Layer                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Documents  │ │    APIs     │ │  Databases  │ │ Real-time   │ │
│  │ (Basic)     │ │ (Basic)     │ │ (Basic)     │ │   Data      │ │
│  │             │ │             │ │             │ │(Framework)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
pip install ragify
```

## 🔧 Quick Start

```python
from ragify import ContextOrchestrator
from ragify.sources import DocumentSource, APISource, DatabaseSource

# Initialize the orchestrator
orchestrator = ContextOrchestrator(
    vector_db_url="redis://localhost:6379",
    cache_url="redis://localhost:6379",
    privacy_level="enterprise"
)

# Add data sources
orchestrator.add_source(DocumentSource("./docs"))
orchestrator.add_source(APISource("https://api.example.com"))
orchestrator.add_source(DatabaseSource("postgresql://..."))

# Get context
context = await orchestrator.get_context(
    query="What are the latest sales figures?",
    user_id="user123",
    session_id="session456",
    max_tokens=4000
)

print(f"Total chunks: {len(context.context.chunks)}")
print(f"Sources: {[chunk.source.name for chunk in context.context.chunks]}")
```

## ⚠️ Current Status

**This is a framework/prototype with the following status:**

### ✅ **Implemented**
- Core orchestrator architecture
- Data models and validation
- Basic relevance scoring with embeddings
- Privacy level framework
- Source abstraction layer
- Async processing framework
- Comprehensive logging
- **Intelligent multi-source context fusion with conflict resolution**
- **Vector database support (ChromaDB, Pinecone, Weaviate, FAISS)**
- Test suite (comprehensive test coverage)

### 🔄 **Framework Only (Placeholder Implementations)**
- Advanced compliance features (GDPR, HIPAA, SOX)
- Enterprise-grade security features

### 📋 **Development Roadmap**
- [x] **COMPLETED**: Implement vector database connections (ChromaDB, Pinecone, Weaviate, FAISS)
- [x] **COMPLETED**: Add document processing libraries (PDF, DOCX, DOC, TXT, MD)
- [x] **COMPLETED**: Add real-time streaming capabilities (WebSocket, MQTT, Redis, Kafka)
- [x] **COMPLETED**: Implement statistical confidence calculations
- [ ] Add basic compliance features

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
