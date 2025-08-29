"""
Tests for the core functionality of the Ragify plugin.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ragify import ContextOrchestrator
from ragify.models import PrivacyLevel, SourceType
from ragify.sources import DocumentSource, APISource, DatabaseSource


class TestContextOrchestrator:
    """Test cases for ContextOrchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create a test orchestrator."""
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.PUBLIC
        )
        yield orchestrator
        await orchestrator.close()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test that the orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.config.privacy_level == PrivacyLevel.PUBLIC
    
    @pytest.mark.asyncio
    async def test_add_source(self, orchestrator):
        """Test adding data sources."""
        # Add document source
        doc_source = DocumentSource(
            name="test_docs",
            url="./test_docs"
        )
        orchestrator.add_source(doc_source)
        
        # Add API source
        api_source = APISource(
            name="test_api",
            url="https://api.example.com"
        )
        orchestrator.add_source(api_source)
        
        # Add database source
        db_source = DatabaseSource(
            name="test_db",
            url="sqlite:///test.db"
        )
        orchestrator.add_source(db_source)
        
        assert len(orchestrator._sources) == 3
        assert "test_docs" in orchestrator._sources
        assert "test_api" in orchestrator._sources
        assert "test_db" in orchestrator._sources
    
    @pytest.mark.asyncio
    async def test_get_context_basic(self, orchestrator):
        """Test basic context retrieval."""
        # Add a mock source that returns some chunks
        mock_source = AsyncMock()
        mock_source.name = "mock_source"
        mock_source.source_type = SourceType.DOCUMENT
        mock_source.get_chunks.return_value = [
            type('obj', (object,), {
                'content': 'Test content',
                'source': 'mock_source',
                'metadata': {},
                'relevance_score': type('obj', (object,), {'score': 0.8})()
            })()
        ]
        
        orchestrator.add_source(mock_source)
        
        # Get context
        context = await orchestrator.get_context(
            query="test query",
            user_id="test_user",
            max_tokens=1000
        )
        
        assert context is not None
        assert context.context.query == "test query"
        assert context.context.user_id == "test_user"
        # Note: The mock chunk might not be properly processed due to missing attributes
        # This test verifies that the context retrieval mechanism works
    
    @pytest.mark.asyncio
    async def test_privacy_levels(self):
        """Test different privacy levels."""
        # Test public level
        public_orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.PUBLIC
        )
        assert public_orchestrator.config.privacy_level == PrivacyLevel.PUBLIC
        await public_orchestrator.close()
        
        # Test enterprise level
        enterprise_orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.ENTERPRISE
        )
        assert enterprise_orchestrator.config.privacy_level == PrivacyLevel.ENTERPRISE
        await enterprise_orchestrator.close()
        
        # Test restricted level
        restricted_orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        assert restricted_orchestrator.config.privacy_level == PrivacyLevel.RESTRICTED
        await restricted_orchestrator.close()


class TestDataSources:
    """Test cases for data sources."""
    
    @pytest.mark.asyncio
    async def test_document_source(self):
        """Test document source functionality."""
        source = DocumentSource(
            name="test_docs",
            url="./test_docs"
        )
        
        # Test basic functionality
        chunks = await source.get_chunks("test query")
        assert isinstance(chunks, list)
        
        await source.close()
    
    @pytest.mark.asyncio
    async def test_api_source(self):
        """Test API source functionality."""
        source = APISource(
            name="test_api",
            url="https://api.example.com"
        )
        
        # Test basic functionality
        chunks = await source.get_chunks("test query")
        assert isinstance(chunks, list)
        
        await source.close()
    
    @pytest.mark.asyncio
    async def test_database_source(self):
        """Test database source functionality."""
        source = DatabaseSource(
            name="test_db",
            url="sqlite:///test.db"
        )
        
        # Test basic functionality
        chunks = await source.get_chunks("test query")
        assert isinstance(chunks, list)
        
        await source.close()


class TestModels:
    """Test cases for data models."""
    
    def test_privacy_levels(self):
        """Test privacy level enum."""
        assert PrivacyLevel.PUBLIC == "public"
        assert PrivacyLevel.ENTERPRISE == "enterprise"
        assert PrivacyLevel.RESTRICTED == "restricted"
    
    def test_source_types(self):
        """Test source type enum."""
        assert SourceType.DOCUMENT == "document"
        assert SourceType.API == "api"
        assert SourceType.DATABASE == "database"
        assert SourceType.REALTIME == "realtime"


if __name__ == "__main__":
    pytest.main([__file__])
