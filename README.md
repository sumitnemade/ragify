# Ragify - Smart Context for LLM-Powered Applications

**A framework for managing context from multiple data sources with conflict resolution, built specifically for LLM-powered applications.**

## 🎯 What is Ragify?

A Python framework that combines data from multiple sources (docs, APIs, databases, real-time) and resolves conflicts. Built specifically for **LLM-powered applications** that need accurate, current information.

## 🚀 Why LLM-Powered Applications Need This?

**LLM-powered applications** often need to combine information from multiple sources. **Ragify helps** by:

- **Detecting conflicts** between data sources
- **Resolving contradictions** using source authority and freshness
- **Combining data** into coherent context
- **Managing privacy** with configurable security levels
- **Processing sources** concurrently for better performance

## 🤖 LLM Application Benefits

### **LLM Chatbots & Assistants:**
- **Conflict detection** in responses
- **Multi-source context** management
- **Source tracking** for transparency

### **LLM Knowledge Systems:**
- **Data conflict resolution** between repositories
- **Multi-source data fusion**
- **Source authority weighting**

### **LLM Research & Analysis Tools:**
- **Combines** papers, databases, live data
- **Detects conflicts** automatically
- **Source validation** capabilities

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                LLM-Powered Application                       │
├─────────────────────────────────────────────────────────────────┤
│                    Ragify Core                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Fusion    │ │   Scoring   │ │   Storage   │ │   Updates   │ │
│  │  Engine     │ │  Engine     │ │  Engine     │ │  Engine     │ │
│  │ (Conflicts) │ │ (Relevance) │ │ (Save)      │ │ (Live)      │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Vector    │ │   Cache     │ │   Privacy   │ │   Monitor   │ │
│  │     DB      │ │  Manager    │ │  Manager    │ │  Engine     │ │
│  │(ChromaDB,   │ │(Redis,      │ │ (Encrypt)   │ │(Performance)│ │
│  │Pinecone,    │ │Memcached)   │ │             │ │             │ │
│  │Weaviate,    │ │             │ │             │ │             │ │
│  │FAISS)       │ │             │ │             │ │             │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Data Sources                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Documents  │ │    APIs     │ │  Databases  │ │ Real-time   │ │
│  │ (PDF, DOCX, │ │ (REST,      │ │ (SQL,       │ │   Data      │ │
│  │ TXT, MD)    │ │ GraphQL)    │ │ NoSQL)      │ │(WebSocket,  │ │
│  │             │ │             │ │             │ │ MQTT, Kafka)│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### **Data Flow:**
1. **Request** → Orchestrator receives context query
2. **Concurrent processing** → Sources processed in parallel  
3. **Relevance scoring** → Assessment of chunk relevance
4. **Conflict resolution** → Detection and resolution of contradictions
5. **Storage** → Context storage with privacy controls
6. **Response** → Processed context delivered

## 🌟 Featured Example: RAG Chat Assistant

**See Ragify in action!** Check out our RAG-based Chat Assistant that demonstrates the framework's capabilities:

### 🤖 [RAG Chat Assistant](examples/rag_chat_assistant/)

A functional chatbot that showcases:
- **📚 PDF Document Processing** with intelligent chunking
- **🔍 Vector Search** using FAISS database
- **🤖 OpenAI Integration** for AI-powered responses
- **💬 Streamlit UI** with modern chat interface
- **🧠 Intelligent Context Fusion** and conflict resolution
- **⚡ Real-time Processing** with async operations

**Great for learning Ragify!** This example shows how to build a RAG system using Ragify's core features.

## 📦 Setup

```bash
# Get code
git clone https://github.com/sumitnemade/ragify.git
cd ragify

# Install deps
pip install -r requirements.txt

# Set path
export PYTHONPATH=src:$PYTHONPATH
```

## 🔧 Quick Start

```python
import sys
sys.path.insert(0, 'src')

from ragify import ContextOrchestrator
from ragify.sources import DocumentSource, APISource

# Setup
orchestrator = ContextOrchestrator(
    vector_db_url="memory://",
    privacy_level="private"
)

# Add sources
orchestrator.add_source(DocumentSource("./docs"))
orchestrator.add_source(APISource("https://api.company.com/data"))

# Get context
context = await orchestrator.get_context(
    query="Latest sales and trends?",
    max_chunks=10
)
```

## 🏗️ Key Features

- **Conflict Detection**: Identifies data contradictions
- **Multi-Source Support**: Documents, APIs, databases, real-time data
- **Privacy Management**: Configurable security levels
- **Vector Database Support**: ChromaDB, Pinecone, Weaviate, FAISS
- **Concurrent Processing**: Parallel source handling
- **Relevance Scoring**: Multi-factor assessment

## 📊 Use Cases

- **AI Chatbots**: Multi-source context management
- **Knowledge Systems**: Data conflict resolution
- **Research Tools**: Multi-source data combination
- **Enterprise Search**: Privacy-controlled search
- **Data Integration**: Combining multiple data sources

## 🚀 Status

**Current Status**: ✅
- Core framework implemented
- Test suite included
- Examples provided
- **🔐 Security & Privacy**: ✅ **Implemented and tested** (GDPR, HIPAA, SOX features)
- Not on PyPI yet

## 🔐 **Security & Privacy Features - IMPLEMENTED**

**✅ Features implemented and tested with working examples**

### **Security Manager**
- **Multi-level Encryption**: AES-256, RSA-2048, Hybrid encryption
- **Access Control**: Role-based permissions (RBAC)
- **Security Monitoring**: Real-time alerts and scoring
- **Key Management**: Automatic rotation and secure storage

### **Privacy Manager**
- **Data Protection**: Sensitive data detection and anonymization
- **Privacy Controls**: 4-tier privacy levels with policies
- **Compliance**: Real-time privacy compliance checking
- **Integration**: Full security manager integration

### **Compliance Manager**
- **GDPR**: Data subject management, right to be forgotten
- **HIPAA**: PHI protection, minimum necessary principle
- **SOX**: Financial data integrity, audit trails
- **Multi-framework**: Simultaneous compliance support

**🎯 Demo**: `python examples/security_encryption_and_access_control.py` - See all features in action!

## 📚 Learn More

- **[RAG Chat Assistant](examples/rag_chat_assistant/)** - Functional RAG chatbot example ⭐
- **[Getting Started](examples/getting_started_with_ragify.py)** - Basic RAGify setup and usage
- **[Document Processing](examples/document_ingestion_and_processing.py)** - Document processing capabilities
- **[Conflict Resolution](examples/conflict_detection_and_resolution.py)** - Intelligent conflict resolution
- **[Vector Database Operations](examples/vector_database_operations.py)** - Vector database operations
- **[Security & Privacy](examples/security_encryption_and_access_control.py)** - Security, encryption & access control
- **[Examples Index](examples/INDEX.md)** - Quick navigation guide
- **[Examples README](examples/README_EXAMPLES.md)** - Comprehensive developer guide

## 🤝 Contribute

Help us improve! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT - see [LICENSE](LICENSE).
