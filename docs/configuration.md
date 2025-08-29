# ⚙️ **Configuration Guide**

This guide explains how to configure Ragify for different use cases and environments.

## 🎯 **Overview**

Ragify uses a flexible configuration system that supports multiple configuration sources and environments.

```mermaid
graph TB
    subgraph "Configuration Sources"
        A[Environment Variables]
        B[Configuration Files]
        C[Code Configuration]
        D[Command Line Args]
    end
    
    subgraph "Configuration Layers"
        E[Default Config]
        F[Environment Config]
        G[User Config]
        H[Runtime Config]
    end
    
    A --> I[Config Manager]
    B --> I
    C --> I
    D --> I
    
    E --> J[Final Config]
    F --> J
    G --> J
    H --> J
```

## 🔧 **Basic Configuration**

### **1. Environment Variables**

```bash
# Core Configuration
export RAGIFY_MAX_CONTEXTS=1000
export RAGIFY_DEFAULT_CHUNK_SIZE=1000
export RAGIFY_DEFAULT_OVERLAP=200

# Database Configuration
export RAGIFY_DATABASE_URL=postgresql://user:pass@localhost:5432/ragify
export RAGIFY_REDIS_URL=redis://localhost:6379

# Vector Database Configuration
export RAGIFY_VECTOR_DB_URL=chromadb://localhost:8000
export RAGIFY_PINECONE_API_KEY=your_api_key

# Security Configuration
export RAGIFY_ENCRYPTION_KEY=your_encryption_key
export RAGIFY_JWT_SECRET=your_jwt_secret

# Logging Configuration
export RAGIFY_LOG_LEVEL=INFO
export RAGIFY_LOG_FORMAT=json
```

### **2. Configuration File (YAML)**

```yaml
# config.yaml
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
  
  # Database configuration
  database:
    url: postgresql://user:pass@localhost:5432/ragify
    pool_size: 20
    max_overflow: 30
    
  # Cache configuration
  cache:
    url: redis://localhost:6379
    ttl: 3600
    max_size: 10000
    
  # Vector database configuration
  vector_database:
    url: chromadb://localhost:8000
    collection_name: ragify_contexts
    embedding_model: all-MiniLM-L6-v2
    
  # Logging configuration
  logging:
    level: INFO
    format: json
    file: logs/ragify.log
```

### **3. Code Configuration**

```python
from ragify import OrchestratorConfig

# Basic configuration
config = OrchestratorConfig(
    max_contexts=1000,
    default_chunk_size=1000,
    default_overlap=200
)

# Advanced configuration
config = OrchestratorConfig(
    max_contexts=1000,
    default_chunk_size=1000,
    default_overlap=200,
    privacy_level=PrivacyLevel.RESTRICTED,
    cache_enabled=True,
    compression_enabled=True,
    max_concurrent_requests=10,
    storage_backend="postgresql",
    vector_backend="chromadb"
)
```

## 🏗️ **Advanced Configuration**

### **1. Multi-Environment Configuration**

```python
import os
from ragify import OrchestratorConfig

def get_config(environment=None):
    if not environment:
        environment = os.getenv('RAGIFY_ENV', 'development')
    
    base_config = {
        'max_contexts': 1000,
        'default_chunk_size': 1000,
        'default_overlap': 200
    }
    
    environment_configs = {
        'development': {
            'database_url': 'sqlite:///dev.db',
            'vector_db_url': 'memory://',
            'log_level': 'DEBUG',
            'cache_enabled': False
        },
        'staging': {
            'database_url': 'postgresql://user:pass@staging-db:5432/ragify',
            'vector_db_url': 'chromadb://staging-chroma:8000',
            'log_level': 'INFO',
            'cache_enabled': True
        },
        'production': {
            'database_url': 'postgresql://user:pass@prod-db:5432/ragify',
            'vector_db_url': 'pinecone://prod-index',
            'log_level': 'WARNING',
            'cache_enabled': True,
            'encryption_enabled': True,
            'anonymization_enabled': True
        }
    }
    
    config = {**base_config, **environment_configs[environment]}
    return OrchestratorConfig(**config)

# Usage
config = get_config('production')
orchestrator = ContextOrchestrator(config)
```

### **2. Source-Specific Configuration**

```python
from ragify.sources import DocumentSource, APISource, DatabaseSource

# Document source configuration
doc_config = {
    'chunk_size': 1500,
    'overlap': 300,
    'supported_formats': ['.pdf', '.docx', '.txt'],
    'metadata_extraction': True,
    'table_extraction': True
}

doc_source = DocumentSource(
    name="company_docs",
    url="./documents",
    **doc_config
)

# API source configuration
api_config = {
    'timeout': 30,
    'retry_attempts': 3,
    'rate_limit': 100,
    'headers': {
        'User-Agent': 'Ragify/1.0',
        'Accept': 'application/json'
    }
}

api_source = APISource(
    name="external_api",
    url="https://api.example.com",
    auth_type="api_key",
    auth_config={"api_key": "your_key"},
    **api_config
)

# Database source configuration
db_config = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600
}

db_source = DatabaseSource(
    name="user_database",
    url="postgresql://user:pass@localhost:5432/users",
    db_type="postgresql",
    **db_config
)
```

### **3. Storage Configuration**

```python
from ragify.storage import VectorDatabase, CacheManager, PrivacyManager

# Vector database configuration
vector_config = {
    'collection_name': 'ragify_contexts',
    'embedding_model': 'all-MiniLM-L6-v2',
    'distance_metric': 'cosine',
    'dimension': 384
}

vector_db = VectorDatabase(
    vector_db_url="chromadb://localhost:8000",
    **vector_config
)

# Cache configuration
cache_config = {
    'ttl': 3600,
    'max_size': 10000,
    'compression': True,
    'serialization': 'json'
}

cache_manager = CacheManager(
    cache_url="redis://localhost:6379",
    **cache_config
)

# Privacy configuration
privacy_config = {
    'default_privacy_level': PrivacyLevel.RESTRICTED,
    'encryption_key': 'your_encryption_key',
    'anonymization_patterns': {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
    }
}

privacy_manager = PrivacyManager(**privacy_config)
```

## 🔒 **Security Configuration**

### **1. Encryption Configuration**

```python
# Generate encryption key
import secrets
encryption_key = secrets.token_urlsafe(32)

# Configure encryption
security_config = {
    'encryption_enabled': True,
    'encryption_key': encryption_key,
    'encryption_algorithm': 'AES-256-GCM',
    'key_rotation_enabled': True,
    'key_rotation_interval': 30  # days
}
```

### **2. Authentication Configuration**

```python
# API authentication
auth_config = {
    'api_key_required': True,
    'api_key_header': 'X-API-Key',
    'jwt_enabled': True,
    'jwt_secret': 'your_jwt_secret',
    'jwt_expiration': 3600,  # seconds
    'oauth2_enabled': True,
    'oauth2_providers': {
        'google': {
            'client_id': 'your_google_client_id',
            'client_secret': 'your_google_client_secret'
        }
    }
}
```

### **3. Privacy Configuration**

```python
# Privacy levels configuration
privacy_config = {
    'default_privacy_level': PrivacyLevel.RESTRICTED,
    'anonymization_enabled': True,
    'pii_detection': True,
    'data_retention_days': 90,
    'audit_logging': True,
    'privacy_patterns': {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
    }
}
```

## 📊 **Performance Configuration**

### **1. Caching Configuration**

```python
# Multi-level caching
cache_config = {
    'levels': {
        'l1': {
            'type': 'memory',
            'max_size': 1000,
            'ttl': 300
        },
        'l2': {
            'type': 'redis',
            'url': 'redis://localhost:6379',
            'max_size': 10000,
            'ttl': 3600
        }
    },
    'compression': True,
    'serialization': 'json'
}
```

### **2. Database Configuration**

```python
# Connection pooling
db_config = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'echo': False,
    'isolation_level': 'READ_COMMITTED'
}
```

### **3. Vector Database Configuration**

```python
# FAISS configuration
faiss_config = {
    'index_type': 'HNSW',
    'dimension': 384,
    'metric': 'cosine',
    'M': 16,
    'efConstruction': 200,
    'efSearch': 100
}

# ChromaDB configuration
chroma_config = {
    'collection_name': 'ragify_contexts',
    'embedding_model': 'all-MiniLM-L6-v2',
    'distance_metric': 'cosine',
    'metadata_indexing': True
}
```

## 🔍 **Monitoring Configuration**

### **1. Logging Configuration**

```python
# Structured logging
logging_config = {
    'level': 'INFO',
    'format': 'json',
    'handlers': {
        'console': {
            'enabled': True,
            'level': 'INFO'
        },
        'file': {
            'enabled': True,
            'filename': 'logs/ragify.log',
            'level': 'DEBUG',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        },
        'syslog': {
            'enabled': True,
            'address': '/dev/log',
            'level': 'WARNING'
        }
    },
    'filters': {
        'sensitive_data': True,
        'performance_metrics': True
    }
}
```

### **2. Metrics Configuration**

```python
# Metrics collection
metrics_config = {
    'enabled': True,
    'collectors': {
        'performance': True,
        'business': True,
        'system': True,
        'security': True
    },
    'exporters': {
        'prometheus': {
            'enabled': True,
            'port': 9090
        },
        'statsd': {
            'enabled': True,
            'host': 'localhost',
            'port': 8125
        }
    },
    'sampling_rate': 0.1  # 10% of requests
}
```

### **3. Health Check Configuration**

```python
# Health checks
health_config = {
    'enabled': True,
    'endpoint': '/health',
    'checks': {
        'database': {
            'enabled': True,
            'timeout': 5
        },
        'cache': {
            'enabled': True,
            'timeout': 2
        },
        'vector_db': {
            'enabled': True,
            'timeout': 3
        },
        'external_apis': {
            'enabled': True,
            'timeout': 10
        }
    },
    'interval': 30  # seconds
}
```

## 🚀 **Deployment Configuration**

### **1. Docker Configuration**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV RAGIFY_ENV=production
ENV RAGIFY_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "ragify.server"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ragify:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAGIFY_ENV=production
      - RAGIFY_DATABASE_URL=postgresql://user:pass@db:5432/ragify
      - RAGIFY_REDIS_URL=redis:redis:6379
    depends_on:
      - db
      - redis
      - chromadb
  
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: ragify
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
  
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  postgres_data:
  redis_data:
  chroma_data:
```

### **2. Kubernetes Configuration**

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragify
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ragify
  template:
    metadata:
      labels:
        app: ragify
    spec:
      containers:
      - name: ragify
        image: ragify:latest
        ports:
        - containerPort: 8000
        env:
        - name: RAGIFY_ENV
          value: "production"
        - name: RAGIFY_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ragify-secrets
              key: database-url
        - name: RAGIFY_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ragify-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### **3. Environment-Specific Configuration**

```python
# config/environments.py
import os
from typing import Dict, Any

class Config:
    def __init__(self):
        self.environment = os.getenv('RAGIFY_ENV', 'development')
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        configs = {
            'development': {
                'debug': True,
                'database_url': 'sqlite:///dev.db',
                'vector_db_url': 'memory://',
                'cache_enabled': False,
                'log_level': 'DEBUG'
            },
            'testing': {
                'debug': True,
                'database_url': 'sqlite:///test.db',
                'vector_db_url': 'memory://',
                'cache_enabled': False,
                'log_level': 'DEBUG'
            },
            'staging': {
                'debug': False,
                'database_url': os.getenv('DATABASE_URL'),
                'vector_db_url': os.getenv('VECTOR_DB_URL'),
                'cache_enabled': True,
                'log_level': 'INFO'
            },
            'production': {
                'debug': False,
                'database_url': os.getenv('DATABASE_URL'),
                'vector_db_url': os.getenv('VECTOR_DB_URL'),
                'cache_enabled': True,
                'encryption_enabled': True,
                'anonymization_enabled': True,
                'log_level': 'WARNING'
            }
        }
        
        return configs.get(self.environment, configs['development'])
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)

# Usage
config = Config()
database_url = config.get('database_url')
```

## 🔧 **Configuration Validation**

### **1. Schema Validation**

```python
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any

class RagifyConfig(BaseModel):
    max_contexts: int = 1000
    default_chunk_size: int = 1000
    default_overlap: int = 200
    database_url: str
    vector_db_url: str
    cache_enabled: bool = True
    encryption_enabled: bool = False
    
    @validator('max_contexts')
    def validate_max_contexts(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('max_contexts must be between 1 and 10000')
        return v
    
    @validator('default_chunk_size')
    def validate_chunk_size(cls, v):
        if v < 100 or v > 5000:
            raise ValueError('default_chunk_size must be between 100 and 5000')
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('sqlite://', 'postgresql://', 'mysql://')):
            raise ValueError('Invalid database URL format')
        return v

# Usage
try:
    config = RagifyConfig(
    database_url='postgresql://user:pass@localhost:5432/ragify',
        vector_db_url='chromadb://localhost:8000'
    )
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### **2. Configuration Testing**

```python
import pytest
from ragify import OrchestratorConfig

def test_configuration():
    # Test valid configuration
    config = OrchestratorConfig(
        max_contexts=1000,
        default_chunk_size=1000,
        default_overlap=200
    )
    
    assert config.max_contexts == 1000
    assert config.default_chunk_size == 1000
    assert config.default_overlap == 200

def test_invalid_configuration():
    # Test invalid configuration
    with pytest.raises(ValueError):
        OrchestratorConfig(
            max_contexts=-1,  # Invalid value
            default_chunk_size=1000,
            default_overlap=200
        )

def test_environment_configuration():
    # Test environment-based configuration
    import os
    os.environ['RAGIFY_MAX_CONTEXTS'] = '2000'
    
    config = OrchestratorConfig()
    assert config.max_contexts == 2000
```

---

## 📚 **Next Steps**

- **[Deployment](deployment.md)** - Production deployment guide
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Troubleshooting](troubleshooting.md)** - Common configuration issues
- **[Performance](performance.md)** - Performance optimization guide
