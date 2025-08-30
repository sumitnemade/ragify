"""
Pytest configuration and fixtures for RAGify testing infrastructure.

This file provides:
- Global test configuration
- Shared fixtures for all tests
- Test markers and categorization
- Performance monitoring setup
- Security testing configuration
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Import RAGify components for testing
from src.ragify.core import ContextOrchestrator
from src.ragify.engines.fusion import IntelligentContextFusionEngine
from src.ragify.engines.scoring import ContextScoringEngine
from src.ragify.engines.storage import StorageEngine
from src.ragify.engines.updates import ContextUpdatesEngine
from src.ragify.models import (
    Context,
    ContextChunk,
    ContextRequest,
    ContextResponse,
    ContextSource,
    OrchestratorConfig,
    PrivacyLevel,
    RelevanceScore,
    SourceType,
)
from src.ragify.sources.api import APISource
from src.ragify.sources.database import DatabaseSource
from src.ragify.sources.document import DocumentSource
from src.ragify.sources.realtime import RealtimeSource
from src.ragify.storage.cache import CacheManager
from src.ragify.storage.compliance import ComplianceManager
from src.ragify.storage.privacy import PrivacyManager
from src.ragify.storage.security import SecurityManager
from src.ragify.storage.vector_db import VectorDatabase


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and configuration."""
    
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "security: Security and compliance tests"
    )
    config.addinivalue_line(
        "markers", "load: Load testing and stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers", "memory: Memory usage and leak tests"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )
    config.addinivalue_line(
        "markers", "database: Database integration tests"
    )
    config.addinivalue_line(
        "markers", "vulnerability: Vulnerability and security testing"
    )
    config.addinivalue_line(
        "markers", "configuration: Configuration and setup testing"
    )
    config.addinivalue_line(
        "markers", "regression: Performance regression tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    
    for item in items:
        # Add performance marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add security marker to security tests
        if "security" in item.name.lower() or "compliance" in item.name.lower():
            item.add_marker(pytest.mark.security)
        
        # Add load marker to load tests
        if "load" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.load)
        
        # Add memory marker to memory tests
        if "memory" in item.name.lower() or "leak" in item.name.lower():
            item.add_marker(pytest.mark.memory)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor test performance and collect metrics."""
    
    def __init__(self):
        self.test_times: Dict[str, float] = {}
        self.memory_usage: Dict[str, float] = {}
        self.start_time: Optional[float] = None
    
    def start_test(self, test_name: str):
        """Start timing a test."""
        self.start_time = time.time()
    
    def end_test(self, test_name: str):
        """End timing a test and record results."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.test_times[test_name] = duration
            self.start_time = None
    
    def get_performance_report(self) -> Dict[str, float]:
        """Get performance report for all tests."""
        return self.test_times.copy()


@pytest.fixture(scope="session")
def performance_monitor() -> PerformanceMonitor:
    """Global performance monitor for all tests."""
    return PerformanceMonitor()


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_text() -> str:
    """Sample text for testing document processing."""
    return """
    This is a sample document for testing purposes. It contains multiple sentences
    that can be processed and chunked by the document source. The text includes
    various types of content that might be found in real-world documents.
    
    This paragraph contains additional information about the testing process.
    We can use this text to verify that chunking, processing, and analysis
    work correctly across different document types and sizes.
    """


@pytest.fixture
def sample_documents() -> List[Dict[str, str]]:
    """Sample documents for testing."""
    return [
        {
            "content": "Document 1: Introduction to RAG systems",
            "metadata": {"type": "introduction", "author": "Test Author"}
        },
        {
            "content": "Document 2: Advanced RAG techniques",
            "metadata": {"type": "advanced", "author": "Test Author"}
        },
        {
            "content": "Document 3: RAG implementation examples",
            "metadata": {"type": "examples", "author": "Test Author"}
        }
    ]


@pytest.fixture
def sample_context_chunks() -> List[ContextChunk]:
    """Sample context chunks for testing."""
    return [
        ContextChunk(
            content="Sample chunk 1 content",
            source=ContextSource(
                name="test_source_1",
                source_type=SourceType.DOCUMENT,
                url="file:///test1.txt"
            ),
            metadata={"chunk_id": 1, "position": 0}
        ),
        ContextChunk(
            content="Sample chunk 2 content",
            source=ContextSource(
                name="test_source_2",
                source_type=SourceType.API,
                url="https://api.test.com/data"
            ),
            metadata={"chunk_id": 2, "position": 1}
        )
    ]


@pytest.fixture
def sample_context_request() -> ContextRequest:
    """Sample context request for testing."""
    return ContextRequest(
        query="test query",
        max_chunks=10,
        min_relevance_score=0.5,
        privacy_level=PrivacyLevel.PRIVATE
    )


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Basic orchestrator configuration for testing."""
    return OrchestratorConfig(
        max_context_size=1000,
        cache_ttl=300,
        privacy_level=PrivacyLevel.PRIVATE,
        enable_logging=True
    )


@pytest.fixture
async def context_orchestrator(orchestrator_config) -> AsyncGenerator[ContextOrchestrator, None]:
    """Context orchestrator instance for testing."""
    orchestrator = ContextOrchestrator(config=orchestrator_config)
    try:
        yield orchestrator
    finally:
        await orchestrator.close()


@pytest.fixture
def fusion_engine(orchestrator_config) -> IntelligentContextFusionEngine:
    """Fusion engine instance for testing."""
    return IntelligentContextFusionEngine(config=orchestrator_config)


@pytest.fixture
def scoring_engine(orchestrator_config) -> ContextScoringEngine:
    """Scoring engine instance for testing."""
    return ContextScoringEngine(config=orchestrator_config)


@pytest.fixture
def storage_engine() -> StorageEngine:
    """Storage engine instance for testing."""
    return StorageEngine(
        encryption_key="test_key_12345",
        privacy_level=PrivacyLevel.PRIVATE
    )


@pytest.fixture
def updates_engine() -> ContextUpdatesEngine:
    """Updates engine instance for testing."""
    return ContextUpdatesEngine()


@pytest.fixture
def cache_manager() -> CacheManager:
    """Cache manager instance for testing."""
    return CacheManager(
        cache_url="memory://"
    )


@pytest.fixture
def privacy_manager() -> PrivacyManager:
    """Privacy manager instance for testing."""
    return PrivacyManager(
        default_privacy_level=PrivacyLevel.PRIVATE,
        encryption_key="test_key_12345"
    )


@pytest.fixture
def security_manager() -> SecurityManager:
    """Security manager instance for testing."""
    return SecurityManager(
        encryption_key="test_key_12345",
        security_level="standard"
    )


@pytest.fixture
def compliance_manager(security_manager) -> ComplianceManager:
    """Compliance manager instance for testing."""
    return ComplianceManager(security_manager=security_manager)


@pytest.fixture
async def vector_database() -> AsyncGenerator[VectorDatabase, None]:
    """Vector database instance for testing."""
    db = VectorDatabase(
        vector_db_url="chroma:///tmp/test_chroma"
    )
    try:
        await db.connect()
        yield db
    finally:
        # Cleanup if needed
        pass


# =============================================================================
# SOURCE FIXTURES
# =============================================================================

@pytest.fixture
def document_source() -> DocumentSource:
    """Document source instance for testing."""
    return DocumentSource(
        name="test_document_source",
        source_type=SourceType.DOCUMENT,
        url="file:///test_docs",
        chunk_size=1000,
        overlap=200
    )


@pytest.fixture
def api_source() -> APISource:
    """API source instance for testing."""
    return APISource(
        name="test_api_source",
        source_type=SourceType.API,
        url="https://httpbin.org/json",
        headers={"User-Agent": "RAGify-Test"},
        timeout=30
    )


@pytest.fixture
def database_source() -> DatabaseSource:
    """Database source instance for testing."""
    return DatabaseSource(
        name="test_database_source",
        source_type=SourceType.DATABASE,
        url="sqlite:///test.db",
        query_template="SELECT * FROM test_table WHERE content LIKE :query"
    )


@pytest.fixture
def realtime_source() -> RealtimeSource:
    """Realtime source instance for testing."""
    return RealtimeSource(
        name="test_realtime_source",
        source_type=SourceType.REALTIME,
        url="ws://echo.websocket.org",
        connection_type="websocket"
    )


# =============================================================================
# TEST ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_db_path(temp_dir) -> str:
    """Test database path."""
    return str(temp_dir / "test.db")


@pytest.fixture
def mock_http_response():
    """Mock HTTP response for API testing."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json.return_value = {"test": "data"}
    mock_response.text = '{"test": "data"}'
    return mock_response


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================

@pytest_asyncio.fixture
async def async_context():
    """Async context for testing async functions."""
    # This fixture ensures proper async context for tests
    pass


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Performance thresholds for testing."""
    return {
        "test_execution_time": 1.0,  # 1 second max
        "memory_usage_mb": 100,      # 100 MB max
        "response_time_ms": 100,     # 100ms max
        "throughput_ops_per_sec": 1000  # 1000 ops/sec min
    }


# =============================================================================
# SECURITY TESTING FIXTURES
# =============================================================================

@pytest.fixture
def security_test_data() -> Dict[str, str]:
    """Sensitive data for security testing."""
    return {
        "email": "test@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "ip_address": "192.168.1.1"
    }


# =============================================================================
# LOAD TESTING FIXTURES
# =============================================================================

@pytest.fixture
def load_test_config() -> Dict[str, int]:
    """Configuration for load testing."""
    return {
        "concurrent_users": 100,
        "requests_per_user": 10,
        "test_duration_seconds": 60,
        "ramp_up_time_seconds": 10
    }


# =============================================================================
# TEST UTILITIES
# =============================================================================

def assert_performance_met(actual_time: float, expected_max: float, test_name: str):
    """Assert that performance meets expectations."""
    assert actual_time <= expected_max, (
        f"Performance test failed for {test_name}: "
        f"Expected <= {expected_max}s, got {actual_time}s"
    )


def assert_memory_usage_acceptable(usage_mb: float, max_mb: float, test_name: str):
    """Assert that memory usage is acceptable."""
    assert usage_mb <= max_mb, (
        f"Memory usage test failed for {test_name}: "
        f"Expected <= {max_mb}MB, got {usage_mb}MB"
    )


def create_large_test_data(size_mb: int) -> str:
    """Create large test data for performance testing."""
    # Generate approximately size_mb of test data
    base_text = "This is test data for performance testing. " * 100
    repetitions = (size_mb * 1024 * 1024) // len(base_text)
    return base_text * repetitions


# =============================================================================
# TEST MARKERS AND CATEGORIZATION
# =============================================================================

# Performance test markers
pytest_plugins = ["pytest_asyncio"]

# Note: pytestmark removed to avoid applying asyncio to all tests
# Individual tests should use @pytest.mark.asyncio when needed
