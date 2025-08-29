# 🚀 **Installation Guide**

This guide will help you install and set up Ragify on your system.

## 📋 **Prerequisites**

Before installing Ragify, ensure you have the following:

### **System Requirements**
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB free space

### **Python Environment**
```bash
# Check Python version
python --version

# Should output: Python 3.8.x or higher
```

## 🔧 **Installation Methods**

### **Method 1: Using pip (Recommended)**

```bash
# Install from PyPI (when available)
pip install ragify

# Or install from GitHub
pip install git+https://github.com/sumitnemade/ragify.git
```

### **Method 2: From Source**

```bash
# Clone the repository
git clone https://github.com/sumitnemade/ragify.git
cd ragify

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### **Method 3: Using Docker (Coming Soon)**

```bash
# Pull the Docker image
docker pull ragify/ragify:latest

# Run the container
docker run -p 8000:8000 ragify/ragify:latest
```

## 📦 **Dependencies Installation**

Ragify has several optional dependencies depending on your use case:

### **Core Dependencies (Required)**
```bash
pip install pydantic numpy pandas structlog
```

### **Database Dependencies (Optional)**
```bash
# PostgreSQL
pip install asyncpg psycopg2-binary

# MySQL
pip install aiomysql

# SQLite (built-in)
# No additional installation needed

# MongoDB
pip install pymongo motor
```

### **Vector Database Dependencies (Optional)**
```bash
# ChromaDB
pip install chromadb

# Pinecone
pip install pinecone-client

# Weaviate
pip install weaviate-client

# FAISS
pip install faiss-cpu
```

### **Document Processing Dependencies (Optional)**
```bash
# PDF processing
pip install PyPDF2 pdfplumber

# Word document processing
pip install python-docx docx2txt
```

### **Real-time Dependencies (Optional)**
```bash
# WebSocket support
pip install websockets

# MQTT support
pip install asyncio-mqtt

# Kafka support
pip install kafka-python
```

## 🔐 **Security Dependencies (Optional)**
```bash
# Encryption and hashing
pip install cryptography bcrypt
```

## ⚙️ **Configuration Setup**

### **1. Environment Variables**

Create a `.env` file in your project root:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/ragify
REDIS_URL=redis://localhost:6379

# Vector Database Configuration
CHROMADB_URL=http://localhost:8000
PINECONE_API_KEY=your_pinecone_api_key
WEAVIATE_URL=http://localhost:8080

# Security Configuration
ENCRYPTION_KEY=your_encryption_key_here
JWT_SECRET=your_jwt_secret_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### **2. Configuration File**

Create `config.yaml`:

```yaml
# Ragify Configuration
ragify:
  # Core settings
  max_contexts: 1000
  default_chunk_size: 1000
  default_overlap: 200
  
  # Privacy settings
  default_privacy_level: PRIVATE
  encryption_enabled: true
  anonymization_enabled: true
  
  # Performance settings
  cache_enabled: true
  compression_enabled: true
  max_concurrent_requests: 10
  
  # Storage settings
  storage_backend: postgresql
  vector_backend: chromadb
  
  # Logging settings
  log_level: INFO
  log_format: json
```

## 🧪 **Verification**

### **1. Basic Installation Test**

```python
# Test basic import
import ragify
print("✅ Ragify imported successfully!")

# Test core functionality
from ragify import ContextOrchestrator
print("✅ Core components available!")
```

### **2. Feature-Specific Tests**

```python
# Test document processing
from ragify.sources import DocumentSource
print("✅ Document processing available!")

# Test API integration
from ragify.sources import APISource
print("✅ API integration available!")

# Test database integration
from ragify.sources import DatabaseSource
print("✅ Database integration available!")

# Test vector databases
from ragify.storage import VectorDatabase
print("✅ Vector database support available!")
```

## 🚀 **Quick Start Test**

```python
import asyncio
from ragify import ContextOrchestrator, OrchestratorConfig

async def test_installation():
    # Create configuration
    config = OrchestratorConfig(
        max_contexts=100,
        default_chunk_size=1000,
        default_overlap=200
    )
    
    # Initialize orchestrator
    orchestrator = ContextOrchestrator(config)
    
    # Test basic functionality
    request = ContextRequest(
        query="Hello, Ragify!",
        user_id="test_user",
        session_id="test_session"
    )
    
    response = await orchestrator.get_context(request)
    print(f"✅ Test successful! Retrieved {len(response.chunks)} chunks")

# Run the test
asyncio.run(test_installation())
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. Import Errors**
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

#### **2. Database Connection Issues**
```bash
# Check if database is running
# PostgreSQL
sudo systemctl status postgresql

# Redis
redis-cli ping
```

#### **3. Vector Database Issues**
```bash
# Check ChromaDB
curl http://localhost:8000/api/v1/heartbeat

# Check Pinecone
python -c "import pinecone; print('Pinecone available')"
```

#### **4. Permission Issues**
```bash
# Fix file permissions
chmod +x scripts/*.py
chmod 600 .env
```

### **Getting Help**

If you encounter issues:

1. **Check the logs**: Look for error messages in the console
2. **Verify dependencies**: Ensure all required packages are installed
3. **Check configuration**: Verify your `.env` and `config.yaml` files
4. **Search issues**: Check existing GitHub issues
5. **Create issue**: Report new problems with detailed information

## 📚 **Next Steps**

After successful installation:

1. **[Basic Usage](basic-usage.md)** - Learn how to use Ragify
2. **[Configuration](configuration.md)** - Configure for your needs
3. **[Examples](examples.md)** - See real-world examples
4. **[API Reference](api-reference.md)** - Complete API documentation

---

**Need Help?** [Create an issue](https://github.com/sumitnemade/ragify/issues) or [join discussions](https://github.com/sumitnemade/ragify/discussions)
