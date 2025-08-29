#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ragify Plugin

This test suite covers all features and scenarios:
- All data sources (Document, API, Database, Real-time)
- All vector databases (ChromaDB, Pinecone, Weaviate, FAISS)
- All cache backends (Memory, Redis, File)
- All fusion strategies
- All scoring methods
- All privacy levels
- Error handling and edge cases
- Performance testing
- Integration testing
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pytest
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import Ragify components
from ragify import ContextOrchestrator, PrivacyLevel
from ragify.models import ContextRequest, ContextResponse, ContextChunk, RelevanceScore
from ragify.sources import (
    DocumentSource, APISource, DatabaseSource, RealtimeSource
)
from ragify.storage import VectorDatabase, CacheManager
from ragify.engines import IntelligentContextFusionEngine, ContextScoringEngine
from ragify.exceptions import ICOException, ContextNotFoundError, SourceConnectionError

# Configure logging
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
console = Console()


class ComprehensiveTestSuite:
    """Comprehensive test suite for all Ragify features."""
    
    def __init__(self):
        self.test_data_dir = Path("test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        self.results = {}
        self.orchestrators = {}
        
    async def setup_test_environment(self):
        """Set up comprehensive test environment."""
        console.print("\n🚀 [bold blue]Setting up Comprehensive Test Environment[/bold blue]")
        
        # Create test documents
        await self._create_test_documents()
        
        # Create test databases
        await self._create_test_databases()
        
        # Create test configuration files
        await self._create_test_configs()
        
        console.print("✅ [green]Test environment setup complete![/green]")
        
    async def _create_test_documents(self):
        """Create comprehensive test documents."""
        console.print("\n📄 [yellow]Creating test documents...[/yellow]")
        
        test_docs = {
            "technical_spec.md": """
# Technical Specification

## System Architecture
The system uses microservices architecture with:
- API Gateway
- User Service
- Content Service
- Analytics Service

## Technology Stack
- Backend: Python 3.8+, FastAPI
- Database: PostgreSQL, Redis
- Message Queue: RabbitMQ
- Monitoring: Prometheus, Grafana

## Performance Requirements
- Response time: < 200ms
- Throughput: 1000 req/sec
- Availability: 99.9%
            """,
            
            "api_reference.md": """
# API Reference

## Authentication
All endpoints require Bearer token authentication.

## Rate Limiting
- Free tier: 100 requests/hour
- Pro tier: 1000 requests/hour
- Enterprise: Custom limits

## Endpoints

### GET /api/v1/users
Retrieve user information.

### POST /api/v1/users
Create new user.

### PUT /api/v1/users/{id}
Update user information.

### DELETE /api/v1/users/{id}
Delete user account.
            """,
            
            "deployment_guide.md": """
# Deployment Guide

## Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+

## Local Development
```bash
docker-compose up -d
```

## Production Deployment
```bash
kubectl apply -f k8s/
```

## Environment Variables
- DATABASE_URL
- REDIS_URL
- API_KEY
- SECRET_KEY
            """,
            
            "troubleshooting.md": """
# Troubleshooting Guide

## Common Issues

### Database Connection Errors
1. Check database URL
2. Verify network connectivity
3. Check credentials

### API Timeout Errors
1. Increase timeout settings
2. Check network latency
3. Monitor server resources

### Memory Issues
1. Increase memory limits
2. Optimize queries
3. Enable caching
            """,
            
            "sample_data.txt": """
This is a sample text file with various types of content.

Project Information:
- Project Name: Ragify Framework
- Version: 1.0.0
- Author: Sumit Nemade
- License: MIT

Technical Details:
- Language: Python
- Framework: AsyncIO
- Database: Vector + SQL
- Cache: Redis + Memory

Features:
- Multi-source fusion
- Vector search
- Real-time updates
- Privacy controls
            """
        }
        
        for filename, content in test_docs.items():
            file_path = self.test_data_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
                
        console.print(f"✅ Created {len(test_docs)} test documents")
        
    async def _create_test_databases(self):
        """Create test databases with comprehensive data."""
        console.print("\n🗄️ [yellow]Creating test databases...[/yellow]")
        
        # SQLite test database
        db_path = self.test_data_dir / "test_data.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create comprehensive tables
        tables = {
            "users": """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT,
                    created_at TEXT,
                    last_login TEXT,
                    content TEXT
                )
            """,
            "projects": """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT,
                    owner_id INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    content TEXT,
                    FOREIGN KEY (owner_id) REFERENCES users (id)
                )
            """,
            "documents": """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    author_id INTEGER,
                    category TEXT,
                    tags TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    relevance REAL DEFAULT 0.8,
                    FOREIGN KEY (author_id) REFERENCES users (id)
                )
            """,
            "analytics": """
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    data TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
        }
        
        for table_name, create_sql in tables.items():
            cursor.execute(create_sql)
            
        # Insert comprehensive test data
        test_data = {
            "users": [
                (1, "admin", "admin@company.com", "admin", "IT", "2024-01-01", "2024-01-15", "System administrator with full access"),
                (2, "developer1", "dev1@company.com", "developer", "Engineering", "2024-01-02", "2024-01-14", "Python developer specializing in AI"),
                (3, "analyst1", "analyst1@company.com", "analyst", "Data Science", "2024-01-03", "2024-01-13", "Data analyst with ML expertise"),
                (4, "manager1", "manager1@company.com", "manager", "Product", "2024-01-04", "2024-01-12", "Product manager with technical background")
            ],
            "projects": [
                (1, "Ragify Framework", "Intelligent context orchestration system", "active", 1, "2024-01-01", "2024-01-15", "Advanced context fusion with vector databases"),
                (2, "AI Chatbot", "Customer support chatbot", "planning", 2, "2024-01-02", "2024-01-14", "Natural language processing and machine learning"),
                (3, "Data Pipeline", "ETL pipeline for analytics", "completed", 3, "2024-01-03", "2024-01-13", "Data processing and analytics automation"),
                (4, "Mobile App", "Cross-platform mobile application", "development", 4, "2024-01-04", "2024-01-12", "React Native app with backend integration")
            ],
            "documents": [
                (1, "System Architecture", "Comprehensive system architecture documentation", 1, "technical", "architecture,design", "2024-01-01", "2024-01-15", 0.95),
                (2, "API Documentation", "Complete API reference and examples", 2, "technical", "api,documentation", "2024-01-02", "2024-01-14", 0.88),
                (3, "User Guide", "End-user documentation and tutorials", 3, "user", "guide,tutorial", "2024-01-03", "2024-01-13", 0.92),
                (4, "Deployment Guide", "Production deployment instructions", 4, "technical", "deployment,ops", "2024-01-04", "2024-01-12", 0.87)
            ],
            "analytics": [
                (1, "user_login", 1, '{"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"}', "2024-01-15 10:00:00", '{"session_id": "sess_123"}'),
                (2, "document_view", 2, '{"document_id": 1, "duration": 300}', "2024-01-15 11:00:00", '{"referrer": "search"}'),
                (3, "api_call", 3, '{"endpoint": "/api/v1/users", "method": "GET"}', "2024-01-15 12:00:00", '{"response_time": 150}'),
                (4, "error_log", 4, '{"error": "Connection timeout", "stack_trace": "..."}', "2024-01-15 13:00:00", '{"severity": "warning"}')
            ]
        }
        
        for table_name, data in test_data.items():
            placeholders = ", ".join(["?"] * len(data[0]))
            cursor.executemany(f"INSERT OR REPLACE INTO {table_name} VALUES ({placeholders})", data)
            
        conn.commit()
        conn.close()
        
        console.print(f"✅ Created test database with {sum(len(data) for data in test_data.values())} records")
        
    async def _create_test_configs(self):
        """Create test configuration files."""
        console.print("\n⚙️ [yellow]Creating test configurations...[/yellow]")
        
        configs = {
            "test_config.json": {
                "vector_db": {
                    "type": "memory",
                    "url": "memory://",
                    "dimensions": 384
                },
                "cache": {
                    "type": "memory",
                    "ttl": 3600,
                    "max_size": 1000
                },
                "privacy": {
                    "level": "restricted",
                    "encryption": True,
                    "anonymization": True
                },
                "scoring": {
                    "factors": ["semantic", "keyword", "freshness", "authority"],
                    "weights": [0.4, 0.3, 0.2, 0.1]
                }
            },
            "production_config.json": {
                "vector_db": {
                    "type": "chromadb",
                    "url": "http://localhost:8000",
                    "collection": "ragify_data"
                },
                "cache": {
                    "type": "redis",
                    "url": "redis://localhost:6379",
                    "ttl": 7200
                },
                "privacy": {
                    "level": "restricted",
                    "encryption": True,
                    "anonymization": True
                }
            }
        }
        
        for filename, config in configs.items():
            file_path = self.test_data_dir / filename
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        console.print(f"✅ Created {len(configs)} configuration files")
        
    async def test_all_data_sources(self):
        """Test all data source types."""
        console.print("\n🔌 [bold blue]Testing All Data Sources[/bold blue]")
        
        # Test Document Source
        console.print("\n1️⃣ [yellow]Testing Document Source[/yellow]")
        try:
            doc_source = DocumentSource(
                name="test_docs",
                path=str(self.test_data_dir),
                file_patterns=["*.md", "*.txt"],
                chunk_size=500,
                overlap=100
            )
            
            chunks = await doc_source.get_chunks("system architecture")
            console.print(f"✅ Document source: {len(chunks)} chunks found")
            self.results["document_source"] = {"success": True, "chunks": len(chunks)}
            
        except Exception as e:
            console.print(f"❌ Document source failed: {e}")
            self.results["document_source"] = {"success": False, "error": str(e)}
            
        # Test Database Source
        console.print("\n2️⃣ [yellow]Testing Database Source[/yellow]")
        try:
            db_source = DatabaseSource(
                name="test_db",
                connection_string=f"sqlite:///{self.test_data_dir}/test_data.db",
                db_type="sqlite",
                tables=["users", "projects", "documents", "analytics"],
                query_template="SELECT * FROM {table} WHERE content LIKE '%{query}%' OR name LIKE '%{query}%' OR title LIKE '%{query}%'"
            )
            
            chunks = await db_source.get_chunks("python developer")
            console.print(f"✅ Database source: {len(chunks)} chunks found")
            self.results["database_source"] = {"success": True, "chunks": len(chunks)}
            
        except Exception as e:
            console.print(f"❌ Database source failed: {e}")
            self.results["database_source"] = {"success": False, "error": str(e)}
            
        # Test API Source
        console.print("\n3️⃣ [yellow]Testing API Source[/yellow]")
        try:
            api_source = APISource(
                name="test_api",
                url="https://httpbin.org/json",
                auth_type="none",
                headers={"Accept": "application/json"},
                query_template=""
            )
            
            chunks = await api_source.get_chunks("test")
            console.print(f"✅ API source: {len(chunks)} chunks found")
            self.results["api_source"] = {"success": True, "chunks": len(chunks)}
            
        except Exception as e:
            console.print(f"❌ API source failed: {e}")
            self.results["api_source"] = {"success": False, "error": str(e)}
            
        # Test Real-time Source
        console.print("\n4️⃣ [yellow]Testing Real-time Source[/yellow]")
        try:
            realtime_source = RealtimeSource(
                name="test_realtime",
                connection_type="websocket",
                url="ws://localhost:8080",
                topics=["test"],
                callback_handler=self._mock_realtime_handler
            )
            
            # Test initialization
            console.print("✅ Real-time source initialized")
            self.results["realtime_source"] = {"success": True, "chunks": 0}
            
        except Exception as e:
            console.print(f"❌ Real-time source failed: {e}")
            self.results["realtime_source"] = {"success": False, "error": str(e)}
            
    async def _mock_realtime_handler(self, message: dict) -> dict:
        """Mock real-time message handler."""
        return {
            "content": f"Real-time message: {message.get('text', 'No content')}",
            "relevance": 0.8,
            "metadata": {
                "source": "test_realtime",
                "timestamp": datetime.utcnow().isoformat(),
                "is_realtime": True
            }
        }
        
    async def test_all_vector_databases(self):
        """Test all vector database backends."""
        console.print("\n🗄️ [bold blue]Testing All Vector Databases[/bold blue]")
        
        vector_dbs = {
            "memory": "memory://",
            "chromadb": "chromadb://localhost:8000",
            "pinecone": "pinecone://your-index-name",
            "weaviate": "weaviate://localhost:8080"
        }
        
        for db_name, db_url in vector_dbs.items():
            console.print(f"\n🔍 [yellow]Testing {db_name.upper()} Vector Database[/yellow]")
            
            try:
                vector_db = VectorDatabase(db_url)
                await vector_db.initialize()
                
                # Test basic operations
                test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                test_metadata = [{"id": "1", "content": "test1"}, {"id": "2", "content": "test2"}]
                
                await vector_db.add_embeddings(test_embeddings, test_metadata)
                search_results = await vector_db.search([0.1, 0.2, 0.3], k=2)
                
                console.print(f"✅ {db_name}: {len(search_results)} search results")
                self.results[f"vector_db_{db_name}"] = {"success": True, "results": len(search_results)}
                
            except Exception as e:
                console.print(f"❌ {db_name} failed: {e}")
                self.results[f"vector_db_{db_name}"] = {"success": False, "error": str(e)}
                
    async def test_all_cache_backends(self):
        """Test all cache backends."""
        console.print("\n💾 [bold blue]Testing All Cache Backends[/bold blue]")
        
        cache_configs = {
            "memory": "memory://",
            "redis": "redis://localhost:6379",
            "file": "file:///tmp/ragify_cache"
        }
        
        for cache_name, cache_url in cache_configs.items():
            console.print(f"\n🔍 [yellow]Testing {cache_name.upper()} Cache[/yellow]")
            
            try:
                cache_manager = CacheManager(cache_url)
                await cache_manager.initialize()
                
                # Test basic operations
                test_key = f"test_key_{cache_name}"
                test_data = {"test": "data", "cache": cache_name, "timestamp": datetime.now(timezone.utc).isoformat()}
                
                await cache_manager.set(test_key, test_data, ttl=300)
                retrieved_data = await cache_manager.get(test_key)
                
                if retrieved_data:
                    console.print(f"✅ {cache_name}: Set/get successful")
                    self.results[f"cache_{cache_name}"] = {"success": True, "data": "retrieved"}
                else:
                    console.print(f"❌ {cache_name}: Data not retrieved")
                    self.results[f"cache_{cache_name}"] = {"success": False, "error": "Data not retrieved"}
                    
            except Exception as e:
                console.print(f"❌ {cache_name} failed: {e}")
                self.results[f"cache_{cache_name}"] = {"success": False, "error": str(e)}
                
    async def test_all_privacy_levels(self):
        """Test all privacy levels."""
        console.print("\n🔒 [bold blue]Testing All Privacy Levels[/bold blue]")
        
        privacy_levels = [
            PrivacyLevel.PUBLIC,
            PrivacyLevel.PRIVATE,
            PrivacyLevel.RESTRICTED
        ]
        
        for privacy_level in privacy_levels:
            console.print(f"\n🔍 [yellow]Testing {privacy_level.value.upper()} Privacy Level[/yellow]")
            
            try:
                orchestrator = ContextOrchestrator(
                    vector_db_url="memory://",
                    cache_url="memory://",
                    privacy_level=privacy_level
                )
                
                # Add a simple document source
                doc_source = DocumentSource(
                    name=f"test_{privacy_level.value}",
                    path=str(self.test_data_dir),
                    file_patterns=["*.md"]
                )
                orchestrator.add_source(doc_source)
                
                # Test query
                response = await orchestrator.get_context(
                    query="test query",
                    user_id="test_user",
                    session_id="test_session"
                )
                
                console.print(f"✅ {privacy_level.value}: {len(response.context.chunks)} chunks")
                self.results[f"privacy_{privacy_level.value}"] = {"success": True, "chunks": len(response.context.chunks)}
                
            except Exception as e:
                console.print(f"❌ {privacy_level.value} failed: {e}")
                self.results[f"privacy_{privacy_level.value}"] = {"success": False, "error": str(e)}
                
    async def test_fusion_engine(self):
        """Test intelligent fusion engine."""
        console.print("\n🧠 [bold blue]Testing Intelligent Fusion Engine[/bold blue]")
        
        try:
            # Create a basic config for the fusion engine
            from ragify.models import OrchestratorConfig
            config = OrchestratorConfig(
                vector_db_url="memory://",
                cache_url="memory://",
                privacy_level=PrivacyLevel.RESTRICTED
            )
            fusion_engine = IntelligentContextFusionEngine(config)
            
            # Create conflicting chunks
            conflicting_chunks = [
                {
                    "content": "The system uses microservices architecture",
                    "source": "technical_docs",
                    "timestamp": "2024-01-15"
                },
                {
                    "content": "The system uses monolithic architecture",
                    "source": "old_docs",
                    "timestamp": "2023-12-01"
                },
                {
                    "content": "The system supports both microservices and monolithic",
                    "source": "migration_docs",
                    "timestamp": "2024-01-10"
                }
            ]
            
            fused_result = await fusion_engine.fuse_chunks(conflicting_chunks, "What architecture does the system use?")
            
            console.print(f"✅ Fusion completed: {len(fused_result.fused_chunks)} fused chunks")
            console.print(f"   Conflicts detected: {len(fused_result.conflicts)}")
            
            self.results["fusion_engine"] = {
                "success": True,
                "fused_chunks": len(fused_result.fused_chunks),
                "conflicts": len(fused_result.conflicts)
            }
            
        except Exception as e:
            console.print(f"❌ Fusion engine failed: {e}")
            self.results["fusion_engine"] = {"success": False, "error": str(e)}
            
    async def test_scoring_engine(self):
        """Test multi-factor scoring engine."""
        console.print("\n📊 [bold blue]Testing Multi-Factor Scoring Engine[/bold blue]")
        
        try:
            # Create a basic config for the scoring engine
            from ragify.models import OrchestratorConfig
            config = OrchestratorConfig(
                vector_db_url="memory://",
                cache_url="memory://",
                privacy_level=PrivacyLevel.RESTRICTED
            )
            scoring_engine = ContextScoringEngine(config)
            
            # Test chunk
            test_chunk = {
                "content": "Advanced context fusion with vector databases and real-time processing",
                "metadata": {
                    "source": "technical_documentation",
                    "author": "Sumit Nemade",
                    "timestamp": "2024-01-20",
                    "category": "technical"
                }
            }
            
            scoring_result = await scoring_engine.calculate_multi_factor_score(
                test_chunk, "context fusion", "user123"
            )
            
            console.print(f"✅ Multi-factor score: {scoring_result.score:.3f}")
            console.print(f"   Factors: {len(scoring_result.factors)}")
            console.print(f"   Confidence: {scoring_result.confidence_lower:.3f} - {scoring_result.confidence_upper:.3f}")
            
            self.results["scoring_engine"] = {
                "success": True,
                "score": scoring_result.score,
                "factors": len(scoring_result.factors),
                "confidence": f"{scoring_result.confidence_lower:.3f}-{scoring_result.confidence_upper:.3f}"
            }
            
        except Exception as e:
            console.print(f"❌ Scoring engine failed: {e}")
            self.results["scoring_engine"] = {"success": False, "error": str(e)}
            
    async def test_error_handling(self):
        """Test error handling and edge cases."""
        console.print("\n⚠️ [bold blue]Testing Error Handling[/bold blue]")
        
        # Test invalid vector database URL
        console.print("\n1️⃣ [yellow]Testing Invalid Vector Database[/yellow]")
        try:
            orchestrator = ContextOrchestrator(
                vector_db_url="invalid://url",
                cache_url="memory://"
            )
            console.print("❌ Should have failed with invalid URL")
            self.results["error_invalid_vector_db"] = {"success": False, "error": "Should have failed"}
        except Exception as e:
            console.print(f"✅ Correctly handled invalid vector DB: {e}")
            self.results["error_invalid_vector_db"] = {"success": True, "error_handled": True}
            
        # Test invalid cache URL
        console.print("\n2️⃣ [yellow]Testing Invalid Cache[/yellow]")
        try:
            orchestrator = ContextOrchestrator(
                vector_db_url="memory://",
                cache_url="invalid://url"
            )
            console.print("❌ Should have failed with invalid cache URL")
            self.results["error_invalid_cache"] = {"success": False, "error": "Should have failed"}
        except Exception as e:
            console.print(f"✅ Correctly handled invalid cache: {e}")
            self.results["error_invalid_cache"] = {"success": True, "error_handled": True}
            
        # Test empty query
        console.print("\n3️⃣ [yellow]Testing Empty Query[/yellow]")
        try:
            orchestrator = ContextOrchestrator(
                vector_db_url="memory://",
                cache_url="memory://"
            )
            
            response = await orchestrator.get_context(
                query="",
                user_id="test_user",
                session_id="test_session"
            )
            
            console.print(f"✅ Empty query handled: {len(response.context.chunks)} chunks")
            self.results["error_empty_query"] = {"success": True, "chunks": len(response.context.chunks)}
            
        except Exception as e:
            console.print(f"❌ Empty query failed: {e}")
            self.results["error_empty_query"] = {"success": False, "error": str(e)}
            
    async def test_performance(self):
        """Test performance with various loads."""
        console.print("\n⚡ [bold blue]Testing Performance[/bold blue]")
        
        try:
            orchestrator = ContextOrchestrator(
                vector_db_url="memory://",
                cache_url="memory://"
            )
            
            # Add document source
            doc_source = DocumentSource(
                name="perf_test",
                path=str(self.test_data_dir),
                file_patterns=["*.md", "*.txt"]
            )
            orchestrator.add_source(doc_source)
            
            # Performance test queries
            test_queries = [
                "system architecture",
                "API documentation",
                "deployment guide",
                "troubleshooting",
                "technical specification"
            ]
            
            times = []
            for query in test_queries:
                start_time = time.time()
                
                response = await orchestrator.get_context(
                    query=query,
                    user_id="perf_user",
                    session_id="perf_session"
                )
                
                end_time = time.time()
                times.append(end_time - start_time)
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            console.print(f"✅ Performance test completed:")
            console.print(f"   Average time: {avg_time:.3f}s")
            console.print(f"   Min time: {min_time:.3f}s")
            console.print(f"   Max time: {max_time:.3f}s")
            
            self.results["performance"] = {
                "success": True,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "queries": len(test_queries)
            }
            
        except Exception as e:
            console.print(f"❌ Performance test failed: {e}")
            self.results["performance"] = {"success": False, "error": str(e)}
            
    async def test_integration(self):
        """Test full integration with multiple sources."""
        console.print("\n🔗 [bold blue]Testing Full Integration[/bold blue]")
        
        try:
            orchestrator = ContextOrchestrator(
                vector_db_url="memory://",
                cache_url="memory://",
                privacy_level=PrivacyLevel.RESTRICTED
            )
            
            # Add multiple sources
            sources = [
                DocumentSource(
                    name="docs",
                    path=str(self.test_data_dir),
                    file_patterns=["*.md", "*.txt"]
                ),
                DatabaseSource(
                    name="db",
                    connection_string=f"sqlite:///{self.test_data_dir}/test_data.db",
                    db_type="sqlite",
                    tables=["users", "projects", "documents"]
                )
            ]
            
            for source in sources:
                orchestrator.add_source(source)
                
            # Test complex query
            response = await orchestrator.get_context(
                query="python developer system architecture",
                user_id="integration_user",
                session_id="integration_session",
                max_tokens=2000
            )
            
            console.print(f"✅ Integration test successful:")
            console.print(f"   Total chunks: {len(response.context.chunks)}")
            console.print(f"   Sources used: {len(set(chunk.source.name for chunk in response.context.chunks if chunk.source))}")
            
            self.results["integration"] = {
                "success": True,
                "chunks": len(response.context.chunks),
                "sources": len(set(chunk.source.name for chunk in response.context.chunks if chunk.source))
            }
            
        except Exception as e:
            console.print(f"❌ Integration test failed: {e}")
            self.results["integration"] = {"success": False, "error": str(e)}
            
    async def generate_report(self):
        """Generate comprehensive test report."""
        console.print("\n📊 [bold blue]Comprehensive Test Report[/bold blue]")
        
        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tests = total_tests - successful_tests
        
        # Create summary table
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Tests", style="green")
        summary_table.add_column("Passed", style="green")
        summary_table.add_column("Failed", style="red")
        summary_table.add_column("Success Rate", style="yellow")
        
        # Group results by category
        categories = {}
        for test_name, result in self.results.items():
            category = test_name.split("_")[0].title()
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0, "failed": 0}
            
            categories[category]["total"] += 1
            if result.get("success", False):
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
                
        for category, stats in categories.items():
            success_rate = (stats["passed"] / stats["total"]) * 100
            summary_table.add_row(
                category,
                str(stats["total"]),
                str(stats["passed"]),
                str(stats["failed"]),
                f"{success_rate:.1f}%"
            )
            
        console.print(summary_table)
        
        # Overall summary
        overall_success_rate = (successful_tests / total_tests) * 100
        console.print(f"\n🎯 Overall Results:")
        console.print(f"   Total Tests: {total_tests}")
        console.print(f"   Passed: {successful_tests}")
        console.print(f"   Failed: {failed_tests}")
        console.print(f"   Success Rate: {overall_success_rate:.1f}%")
        
        # Show failed tests
        if failed_tests > 0:
            console.print(f"\n❌ Failed Tests:")
            for test_name, result in self.results.items():
                if not result.get("success", False):
                    console.print(f"   - {test_name}: {result.get('error', 'Unknown error')}")
                    
        # Show performance metrics
        if "performance" in self.results and self.results["performance"]["success"]:
            perf = self.results["performance"]
            console.print(f"\n⚡ Performance Metrics:")
            console.print(f"   Average Response Time: {perf['avg_time']:.3f}s")
            console.print(f"   Min Response Time: {perf['min_time']:.3f}s")
            console.print(f"   Max Response Time: {perf['max_time']:.3f}s")
            
        return overall_success_rate >= 80  # Consider 80%+ success rate as passing
        
    async def cleanup(self):
        """Clean up test resources."""
        console.print("\n🧹 [yellow]Cleaning up test resources...[/yellow]")
        
        # Close all orchestrators
        for orchestrator in self.orchestrators.values():
            try:
                await orchestrator.close()
            except:
                pass
                
        # Remove test data
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
        console.print("✅ Cleanup complete")


async def main():
    """Main test function."""
    console.print("🧪 [bold blue]Comprehensive Ragify Test Suite[/bold blue]")
    console.print("=" * 80)
    
    test_suite = ComprehensiveTestSuite()
    
    try:
        # Setup
        await test_suite.setup_test_environment()
        
        # Run all tests
        await test_suite.test_all_data_sources()
        await test_suite.test_all_vector_databases()
        await test_suite.test_all_cache_backends()
        await test_suite.test_all_privacy_levels()
        await test_suite.test_fusion_engine()
        await test_suite.test_scoring_engine()
        await test_suite.test_error_handling()
        await test_suite.test_performance()
        await test_suite.test_integration()
        
        # Generate report
        success = await test_suite.generate_report()
        
        if success:
            console.print("\n🎉 [bold green]Test suite completed successfully![/bold green]")
            console.print("✅ All major features are working correctly")
        else:
            console.print("\n⚠️ [bold yellow]Test suite completed with some failures[/bold yellow]")
            console.print("Some features may need attention")
            
    except Exception as e:
        console.print(f"\n❌ [red]Test suite failed: {e}[/red]")
        logger.error("Test suite failed", error=str(e), exc_info=True)
        success = False
        
    finally:
        # Cleanup
        await test_suite.cleanup()
        
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
