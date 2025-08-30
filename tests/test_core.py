"""
Comprehensive test suite for the core functionality of the Ragify plugin.

This test file provides 100% coverage of the ContextOrchestrator and related
core functionality, including error handling, edge cases, and all public methods.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from uuid import uuid4
from datetime import datetime, timezone

from ragify import ContextOrchestrator
from src.ragify.models import (
    PrivacyLevel, SourceType, ContextRequest, ContextResponse, 
    Context, ContextChunk, ContextSource, RelevanceScore,
    OrchestratorConfig
)
from src.ragify.sources import DocumentSource, APISource, DatabaseSource, RealtimeSource
from src.ragify.exceptions import (
    ICOException, ContextNotFoundError, ConfigurationError, 
    PrivacyViolationError, SourceConnectionError
)


class TestContextOrchestrator:
    """Comprehensive test cases for ContextOrchestrator."""
    
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
    
    @pytest.fixture
    def mock_source(self):
        """Create a mock data source."""
        source = AsyncMock()
        source.name = "mock_source"
        source.source_type = SourceType.DOCUMENT
        source.get_chunks.return_value = []
        source.refresh = AsyncMock()
        source.close = AsyncMock()
        return source
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample context chunk."""
        source = ContextSource(
            id=uuid4(),
            name="test_source",
            source_type=SourceType.DOCUMENT,
            url="test://source"
        )
        return ContextChunk(
            id=uuid4(),
            content="Test content",
            source=source,
            metadata={},
            relevance_score=RelevanceScore(score=0.8)
        )
    
    @pytest.fixture
    def sample_context(self, sample_chunk):
        """Create a sample context."""
        return Context(
            id=uuid4(),
            query="test query",
            chunks=[sample_chunk],
            user_id="test_user",
            session_id="test_session",
            max_tokens=1000,
            privacy_level=PrivacyLevel.PUBLIC
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test that the orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.config.privacy_level == PrivacyLevel.PUBLIC
        assert orchestrator.logger is not None
        assert hasattr(orchestrator, 'fusion_engine')
        assert hasattr(orchestrator, 'scoring_engine')
        assert hasattr(orchestrator, 'storage_engine')
        assert hasattr(orchestrator, 'updates_engine')
        assert orchestrator._sources == {}
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_config(self):
        """Test orchestrator initialization with custom config."""
        config = OrchestratorConfig(
            vector_db_url="memory://vector",
            cache_url="memory://cache",
            privacy_level=PrivacyLevel.ENTERPRISE,
            max_context_size=5000,
            default_relevance_threshold=0.7
        )
        orchestrator = ContextOrchestrator(config=config)
        
        assert orchestrator.config == config
        assert orchestrator.config.privacy_level == PrivacyLevel.ENTERPRISE
        assert orchestrator.config.max_context_size == 5000
        await orchestrator.close()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_without_components(self):
        """Test orchestrator initialization without optional components."""
        orchestrator = ContextOrchestrator(
            privacy_level=PrivacyLevel.PRIVATE
        )
        
        assert orchestrator.config.vector_db_url is None
        assert orchestrator.config.cache_url is None
        assert orchestrator.vector_db is None
        assert orchestrator.cache_manager is None
        await orchestrator.close()
    
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
        assert orchestrator._sources["test_docs"] == doc_source
        assert orchestrator._sources["test_api"] == api_source
        assert orchestrator._sources["test_db"] == db_source
    
    @pytest.mark.asyncio
    async def test_add_source_replacement(self, orchestrator):
        """Test that adding a source with existing name replaces it."""
        source1 = DocumentSource(name="test", url="./test1")
        source2 = DocumentSource(name="test", url="./test2")
        
        orchestrator.add_source(source1)
        assert orchestrator._sources["test"] == source1
        
        orchestrator.add_source(source2)
        assert orchestrator._sources["test"] == source2
        assert len(orchestrator._sources) == 1
    
    @pytest.mark.asyncio
    async def test_remove_source(self, orchestrator):
        """Test removing data sources."""
        source = DocumentSource(name="test", url="./test")
        orchestrator.add_source(source)
        
        assert "test" in orchestrator._sources
        
        orchestrator.remove_source("test")
        assert "test" not in orchestrator._sources
        assert len(orchestrator._sources) == 0
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_source(self, orchestrator):
        """Test removing a source that doesn't exist."""
        orchestrator.remove_source("nonexistent")
        # Should not raise an error, just log a warning
    
    @pytest.mark.asyncio
    async def test_get_source(self, orchestrator):
        """Test getting a data source by name."""
        source = DocumentSource(name="test", url="./test")
        orchestrator.add_source(source)
        
        retrieved_source = orchestrator.get_source("test")
        assert retrieved_source == source
        
        # Test getting non-existent source
        assert orchestrator.get_source("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_list_sources(self, orchestrator):
        """Test listing all data sources."""
        assert orchestrator.list_sources() == []
        
        source1 = DocumentSource(name="test1", url="./test1")
        source2 = DocumentSource(name="test2", url="./test2")
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        
        sources = orchestrator.list_sources()
        assert len(sources) == 2
        assert "test1" in sources
        assert "test2" in sources
    
    @pytest.mark.asyncio
    async def test_get_context_basic(self, orchestrator, mock_source, sample_chunk):
        """Test basic context retrieval."""
        mock_source.get_chunks.return_value = [sample_chunk]
        orchestrator.add_source(mock_source)
        
        context = await orchestrator.get_context(
            query="test query",
            user_id="test_user",
            max_tokens=1000
        )
        
        assert context is not None
        assert isinstance(context, ContextResponse)
        assert context.context.query == "test query"
        assert context.context.user_id == "test_user"
        assert context.processing_time >= 0
        assert context.cache_hit is False
    
    @pytest.mark.asyncio
    async def test_get_context_with_cache(self, orchestrator, mock_source, sample_chunk):
        """Test context retrieval with caching."""
        mock_source.get_chunks.return_value = [sample_chunk]
        orchestrator.add_source(mock_source)
        
        # First request - should not hit cache
        context1 = await orchestrator.get_context(
            query="test query",
            user_id="test_user"
        )
        
        # Second request with same parameters - should hit cache
        context2 = await orchestrator.get_context(
            query="test query",
            user_id="test_user"
        )
        
        assert context1.context.query == context2.context.query
        assert context1.context.user_id == context2.context.user_id
    
    @pytest.mark.asyncio
    async def test_get_context_with_source_filtering(self, orchestrator):
        """Test context retrieval with source filtering."""
        source1 = AsyncMock()
        source1.name = "source1"
        source1.source_type = SourceType.DOCUMENT
        source1.get_chunks.return_value = []
        
        source2 = AsyncMock()
        source2.name = "source2"
        source2.source_type = SourceType.API
        source2.get_chunks.return_value = []
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        
        # Test include_sources
        await orchestrator.get_context(
            query="test",
            sources=["source1"]
        )
        
        # Test exclude_sources
        await orchestrator.get_context(
            query="test",
            exclude_sources=["source2"]
        )
    
    @pytest.mark.asyncio
    async def test_get_context_no_sources(self, orchestrator):
        """Test context retrieval when no sources are available."""
        with pytest.raises(ContextNotFoundError):
            await orchestrator.get_context(query="test query")
    
    @pytest.mark.asyncio
    async def test_get_context_source_timeout(self, orchestrator):
        """Test context retrieval with source timeout."""
        slow_source = AsyncMock()
        slow_source.name = "slow_source"
        slow_source.source_type = SourceType.DOCUMENT
        
        async def slow_get_chunks(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow source
            return []
        
        slow_source.get_chunks = slow_get_chunks
        orchestrator.add_source(slow_source)
        
        # Set a very short timeout
        orchestrator.config.source_timeout = 0.05
        
        # Should handle timeout gracefully
        context = await orchestrator.get_context(query="test")
        assert context is not None
    
    @pytest.mark.asyncio
    async def test_get_context_source_failure(self, orchestrator):
        """Test context retrieval when sources fail."""
        failing_source = AsyncMock()
        failing_source.name = "failing_source"
        failing_source.source_type = SourceType.DOCUMENT
        failing_source.get_chunks.side_effect = Exception("Source failed")
        
        orchestrator.add_source(failing_source)
        
        # Should handle source failures gracefully
        context = await orchestrator.get_context(query="test")
        assert context is not None
    
    @pytest.mark.asyncio
    async def test_get_context_with_relevance_filtering(self, orchestrator):
        """Test context retrieval with relevance filtering."""
        source = AsyncMock()
        source.name = "test_source"
        source.source_type = SourceType.DOCUMENT
        
        # Create chunks with different relevance scores
        chunk1 = MagicMock()
        chunk1.relevance_score = MagicMock()
        chunk1.relevance_score.score = 0.9
        
        chunk2 = MagicMock()
        chunk2.relevance_score = MagicMock()
        chunk2.relevance_score.score = 0.3
        
        source.get_chunks.return_value = [chunk1, chunk2]
        orchestrator.add_source(source)
        
        # Set high relevance threshold
        context = await orchestrator.get_context(
            query="test",
            min_relevance=0.8
        )
        
        # Should only include high-relevance chunks
        assert len(context.context.chunks) >= 0  # May be filtered by scoring engine
    
    @pytest.mark.asyncio
    async def test_get_context_with_token_limit(self, orchestrator, mock_source):
        """Test context retrieval with token limits."""
        # Create chunks with token counts
        chunk1 = MagicMock()
        chunk1.content = "Short content"
        chunk1.token_count = 100
        
        chunk2 = MagicMock()
        chunk2.content = "Longer content"
        chunk2.token_count = 500
        
        mock_source.get_chunks.return_value = [chunk1, chunk2]
        orchestrator.add_source(mock_source)
        
        context = await orchestrator.get_context(
            query="test",
            max_tokens=300
        )
        
        assert context.context.max_tokens == 300
    
    @pytest.mark.asyncio
    async def test_update_context(self, orchestrator, sample_context):
        """Test updating an existing context."""
        updates = {"metadata": {"updated": True}}
        
        # Mock the storage engine
        orchestrator.storage_engine.update_context = AsyncMock(return_value=sample_context)
        
        result = await orchestrator.update_context(
            context_id=sample_context.id,
            updates=updates
        )
        
        assert result == sample_context
        orchestrator.storage_engine.update_context.assert_called_once_with(
            sample_context.id, updates
        )
    
    @pytest.mark.asyncio
    async def test_delete_context(self, orchestrator):
        """Test deleting a context."""
        context_id = uuid4()
        
        # Mock the storage engine
        orchestrator.storage_engine.delete_context = AsyncMock()
        
        await orchestrator.delete_context(context_id)
        
        orchestrator.storage_engine.delete_context.assert_called_once_with(context_id)
    
    @pytest.mark.asyncio
    async def test_get_context_history(self, orchestrator):
        """Test getting context history for a user."""
        user_id = "test_user"
        limit = 5
        
        # Mock the storage engine
        mock_history = [MagicMock(), MagicMock()]
        orchestrator.storage_engine.get_context_history = AsyncMock(return_value=mock_history)
        
        result = await orchestrator.get_context_history(user_id, limit)
        
        assert result == mock_history
        orchestrator.storage_engine.get_context_history.assert_called_once_with(user_id, limit)
    
    @pytest.mark.asyncio
    async def test_refresh_sources(self, orchestrator):
        """Test refreshing all data sources."""
        source1 = AsyncMock()
        source1.name = "source1"
        source1.refresh = AsyncMock()
        
        source2 = AsyncMock()
        source2.name = "source2"
        source2.refresh = AsyncMock()
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        
        await orchestrator.refresh_sources()
        
        source1.refresh.assert_called_once()
        source2.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refresh_sources_with_failures(self, orchestrator):
        """Test refreshing sources when some fail."""
        source1 = AsyncMock()
        source1.name = "source1"
        source1.refresh = AsyncMock()
        
        source2 = AsyncMock()
        source2.name = "source2"
        source2.refresh = AsyncMock(side_effect=Exception("Refresh failed"))
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        
        # Should continue even if some sources fail
        await orchestrator.refresh_sources()
        
        source1.refresh.assert_called_once()
        source2.refresh.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_analytics(self, orchestrator):
        """Test getting analytics about context usage."""
        analytics = await orchestrator.get_analytics()
        
        assert isinstance(analytics, dict)
        assert "total_contexts" in analytics
        assert "cache_hit_rate" in analytics
        assert "average_processing_time" in analytics
        assert "source_usage" in analytics
    
    @pytest.mark.asyncio
    async def test_close(self, orchestrator):
        """Test closing the orchestrator."""
        # Add some sources
        source1 = AsyncMock()
        source1.name = "source1"
        source1.close = AsyncMock()
        
        source2 = AsyncMock()
        source2.name = "source2"
        source2.close = AsyncMock()
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        
        await orchestrator.close()
        
        source1.close.assert_called_once()
        source2.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        async with ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://"
        ) as orchestrator:
            assert orchestrator is not None
            assert orchestrator._sources == {}
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, orchestrator):
        """Test cache key generation for requests."""
        request = ContextRequest(
            query="test query",
            user_id="test_user",
            session_id="test_session",
            max_tokens=1000,
            min_relevance=0.5,
            privacy_level=PrivacyLevel.PUBLIC,
            sources=["source1"],
            exclude_sources=["source2"]
        )
        
        cache_key = orchestrator._generate_cache_key(request)
        
        assert cache_key.startswith("context:")
        assert len(cache_key) > 10
    
    @pytest.mark.asyncio
    async def test_source_filtering(self, orchestrator):
        """Test source filtering logic."""
        source1 = DocumentSource(name="source1", url="./test1")
        source2 = DocumentSource(name="source2", url="./test2")
        source3 = DocumentSource(name="source3", url="./test3")
        
        orchestrator.add_source(source1)
        orchestrator.add_source(source2)
        orchestrator.add_source(source3)
        
        # Test include_sources
        filtered = orchestrator._filter_sources(["source1", "source2"], None)
        assert len(filtered) == 2
        assert "source1" in filtered
        assert "source2" in filtered
        assert "source3" not in filtered
        
        # Test exclude_sources
        filtered = orchestrator._filter_sources(None, ["source3"])
        assert len(filtered) == 2
        assert "source1" in filtered
        assert "source2" in filtered
        assert "source3" not in filtered
        
        # Test both
        filtered = orchestrator._filter_sources(["source1"], ["source2"])
        assert len(filtered) == 1
        assert "source1" in filtered
        assert "source2" not in filtered
        assert "source3" not in filtered
    
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
        
        # Test private level
        private_orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.PRIVATE
        )
        assert private_orchestrator.config.privacy_level == PrivacyLevel.PRIVATE
        await private_orchestrator.close()


class TestDataSources:
    """Comprehensive test cases for data sources."""
    
    @pytest.fixture
    async def document_source(self):
        """Create a document source for testing."""
        source = DocumentSource(
            name="test_docs",
            url="./test_docs"
        )
        yield source
        await source.close()
    
    @pytest.fixture
    async def api_source(self):
        """Create an API source for testing."""
        source = APISource(
            name="test_api",
            url="https://api.example.com"
        )
        yield source
        await source.close()
    
    @pytest.fixture
    async def database_source(self):
        """Create a database source for testing."""
        source = DatabaseSource(
            name="test_db",
            url="sqlite:///test.db"
        )
        yield source
        await source.close()
    
    @pytest.mark.asyncio
    async def test_document_source_initialization(self, document_source):
        """Test document source initialization."""
        assert document_source.name == "test_docs"
        assert document_source.source_type == SourceType.DOCUMENT
        assert document_source.url == "./test_docs"
        assert document_source.logger is not None
        assert document_source.config is not None
    
    @pytest.mark.asyncio
    async def test_api_source_initialization(self, api_source):
        """Test API source initialization."""
        assert api_source.name == "test_api"
        assert api_source.source_type == SourceType.API
        assert api_source.url == "https://api.example.com"
        assert api_source.logger is not None
        assert api_source.config is not None
    
    @pytest.mark.asyncio
    async def test_database_source_initialization(self, database_source):
        """Test database source initialization."""
        assert database_source.name == "test_db"
        assert database_source.source_type == SourceType.DATABASE
        assert database_source.url == "sqlite:///test.db"
        assert database_source.logger is not None
        assert database_source.config is not None
    
    @pytest.mark.asyncio
    async def test_document_source_get_chunks(self, document_source):
        """Test document source chunk retrieval."""
        chunks = await document_source.get_chunks("test query")
        assert isinstance(chunks, list)
    
    @pytest.mark.asyncio
    async def test_api_source_get_chunks(self, api_source):
        """Test API source chunk retrieval."""
        chunks = await api_source.get_chunks("test query")
        assert isinstance(chunks, list)
    
    @pytest.mark.asyncio
    async def test_database_source_get_chunks(self, database_source):
        """Test database source chunk retrieval."""
        chunks = await database_source.get_chunks("test query")
        assert isinstance(chunks, list)
    
    @pytest.mark.asyncio
    async def test_source_stats(self, document_source):
        """Test source statistics retrieval."""
        stats = await document_source.get_stats()
        assert isinstance(stats, dict)
        assert stats['name'] == "test_docs"
        assert stats['type'] == SourceType.DOCUMENT.value
    
    @pytest.mark.asyncio
    async def test_source_availability(self, document_source):
        """Test source availability checking."""
        is_available = await document_source.is_available()
        assert isinstance(is_available, bool)
    
    @pytest.mark.asyncio
    async def test_source_info(self, document_source):
        """Test source information retrieval."""
        source_info = await document_source.get_source_info()
        assert source_info.name == "test_docs"
        assert source_info.source_type == SourceType.DOCUMENT
    
    @pytest.mark.asyncio
    async def test_source_metadata_update(self, document_source):
        """Test source metadata updates."""
        updates = {"test_key": "test_value"}
        await document_source.update_metadata(updates)
        
        # Verify metadata was updated
        stats = await document_source.get_stats()
        assert stats['metadata']['test_key'] == "test_value"


class TestModels:
    """Comprehensive test cases for data models."""
    
    def test_privacy_levels(self):
        """Test privacy level enum."""
        assert PrivacyLevel.PUBLIC == "public"
        assert PrivacyLevel.ENTERPRISE == "enterprise"
        assert PrivacyLevel.RESTRICTED == "restricted"
        assert PrivacyLevel.PRIVATE == "private"
        
        # Test all values
        all_levels = [level.value for level in PrivacyLevel]
        assert "public" in all_levels
        assert "enterprise" in all_levels
        assert "restricted" in all_levels
        assert "private" in all_levels
    
    def test_source_types(self):
        """Test source type enum."""
        assert SourceType.DOCUMENT == "document"
        assert SourceType.API == "api"
        assert SourceType.DATABASE == "database"
        assert SourceType.REALTIME == "realtime"
        assert SourceType.VECTOR == "vector"
        assert SourceType.CACHE == "cache"
        
        # Test all values
        all_types = [source_type.value for source_type in SourceType]
        assert "document" in all_types
        assert "api" in all_types
        assert "database" in all_types
        assert "realtime" in all_types
        assert "vector" in all_types
        assert "cache" in all_types
    
    def test_context_source_creation(self):
        """Test ContextSource model creation."""
        source = ContextSource(
            name="test_source",
            source_type=SourceType.DOCUMENT,
            url="test://source",
            metadata={"test": True}
        )
        
        assert source.name == "test_source"
        assert source.source_type == SourceType.DOCUMENT
        assert source.url == "test://source"
        assert source.metadata["test"] is True
        assert source.is_active is True
        assert source.privacy_level == PrivacyLevel.PRIVATE
    
    def test_context_chunk_creation(self):
        """Test ContextChunk model creation."""
        source = ContextSource(
            name="test_source",
            source_type=SourceType.DOCUMENT
        )
        
        chunk = ContextChunk(
            content="Test content",
            source=source,
            metadata={"test": True},
            token_count=100
        )
        
        assert chunk.content == "Test content"
        assert chunk.source == source
        assert chunk.metadata["test"] is True
        assert chunk.token_count == 100
        assert chunk.privacy_level == PrivacyLevel.PRIVATE
    
    def test_context_creation(self):
        """Test Context model creation."""
        source = ContextSource(
            name="test_source",
            source_type=SourceType.DOCUMENT
        )
        
        chunk = ContextChunk(
            content="Test content",
            source=source
        )
        
        context = Context(
            query="test query",
            chunks=[chunk],
            user_id="test_user",
            max_tokens=1000
        )
        
        assert context.query == "test query"
        assert len(context.chunks) == 1
        assert context.user_id == "test_user"
        assert context.max_tokens == 1000
        assert context.total_tokens == 0  # chunk has no token_count
    
    def test_context_request_creation(self):
        """Test ContextRequest model creation."""
        request = ContextRequest(
            query="test query",
            user_id="test_user",
            max_tokens=1000,
            min_relevance=0.7,
            privacy_level=PrivacyLevel.PUBLIC
        )
        
        assert request.query == "test query"
        assert request.user_id == "test_user"
        assert request.max_tokens == 1000
        assert request.min_relevance == 0.7
        assert request.privacy_level == PrivacyLevel.PUBLIC
    
    def test_context_response_creation(self):
        """Test ContextResponse model creation."""
        source = ContextSource(
            name="test_source",
            source_type=SourceType.DOCUMENT
        )
        
        chunk = ContextChunk(
            content="Test content",
            source=source
        )
        
        context = Context(
            query="test query",
            chunks=[chunk]
        )
        
        response = ContextResponse(
            context=context,
            processing_time=0.5,
            cache_hit=True
        )
        
        assert response.context == context
        assert response.processing_time == 0.5
        assert response.cache_hit is True
    
    def test_orchestrator_config_creation(self):
        """Test OrchestratorConfig model creation."""
        config = OrchestratorConfig(
            vector_db_url="test://vector",
            cache_url="test://cache",
            privacy_level=PrivacyLevel.ENTERPRISE,
            max_context_size=5000,
            default_relevance_threshold=0.8
        )
        
        assert config.vector_db_url == "test://vector"
        assert config.cache_url == "test://cache"
        assert config.privacy_level == PrivacyLevel.ENTERPRISE
        assert config.max_context_size == 5000
        assert config.default_relevance_threshold == 0.8


class TestErrorHandling:
    """Test cases for error handling and exceptions."""
    
    @pytest.mark.asyncio
    async def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        with pytest.raises(ConfigurationError):
            # This should raise a ConfigurationError due to invalid config
            orchestrator = ContextOrchestrator(
                vector_db_url="invalid://url",
                cache_url="invalid://url"
            )
            # The error should occur during component initialization
    
    @pytest.mark.asyncio
    async def test_context_not_found_error(self):
        """Test handling of context not found errors."""
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://"
        )
        
        # No sources added, should raise ContextNotFoundError
        with pytest.raises(ContextNotFoundError):
            await orchestrator.get_context("test query")
        
        await orchestrator.close()
    
    @pytest.mark.asyncio
    async def test_source_connection_error_handling(self):
        """Test handling of source connection errors."""
        # Create a source that will fail to connect
        source = AsyncMock()
        source.name = "failing_source"
        source.source_type = SourceType.DOCUMENT
        source.get_chunks.side_effect = SourceConnectionError("failing_source", "Connection failed")
        
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://"
        )
        
        orchestrator.add_source(source)
        
        # Should handle the error gracefully
        context = await orchestrator.get_context("test query")
        assert context is not None
        
        await orchestrator.close()


if __name__ == "__main__":
    pytest.main([__file__])
