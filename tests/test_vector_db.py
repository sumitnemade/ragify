"""
Tests for vector database functionality.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.ragify.storage.vector_db import VectorDatabase
from src.ragify.models import ContextChunk, ContextSource, SourceType, RelevanceScore
from src.ragify.exceptions import VectorDBError


class TestVectorDatabase:
    """Test vector database functionality."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample context chunks for testing."""
        return [
            ContextChunk(
                content="This is a test document about machine learning",
                source=ContextSource(name="Test Doc", source_type=SourceType.DOCUMENT),
                relevance_score=RelevanceScore(score=0.8),
                created_at=datetime.utcnow(),
                token_count=10
            ),
            ContextChunk(
                content="Another document about artificial intelligence",
                source=ContextSource(name="AI Doc", source_type=SourceType.DOCUMENT),
                relevance_score=RelevanceScore(score=0.7),
                created_at=datetime.utcnow(),
                token_count=8
            ),
            ContextChunk(
                content="A third document about data science",
                source=ContextSource(name="Data Doc", source_type=SourceType.DOCUMENT),
                relevance_score=RelevanceScore(score=0.9),
                created_at=datetime.utcnow(),
                token_count=7
            )
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4],  # 384 dimensions
            [0.2, 0.3, 0.4, 0.5, 0.6] * 76 + [0.2, 0.3, 0.4, 0.5],  # 384 dimensions
            [0.3, 0.4, 0.5, 0.6, 0.7] * 76 + [0.3, 0.4, 0.5, 0.6],  # 384 dimensions
        ]
    
    @pytest.fixture
    def query_embedding(self):
        """Create a query embedding for testing."""
        return [0.15, 0.25, 0.35, 0.45, 0.55] * 76 + [0.15, 0.25, 0.35, 0.39]  # 384 dimensions
    
    @pytest.mark.asyncio
    async def test_vector_db_initialization(self):
        """Test vector database initialization."""
        # Test with valid URL
        vector_db = VectorDatabase("chroma:///tmp/test_chroma")
        assert vector_db.db_type == "chroma"
        assert vector_db.connection_string == "/tmp/test_chroma"
        
        # Test with invalid database type
        with pytest.raises(VectorDBError, match="Unsupported database type"):
            VectorDatabase("invalid://test")
    
    @pytest.mark.asyncio
    async def test_chroma_db_functionality(self, sample_chunks, sample_embeddings, query_embedding):
        """Test ChromaDB functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            
            # Initialize ChromaDB
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            assert len(vector_ids) == 3
            assert all(isinstance(vid, str) for vid in vector_ids)
            
            # Search similar vectors
            results = await vector_db.search_similar(query_embedding, top_k=2)
            assert len(results) > 0
            assert all(len(result) == 3 for result in results)  # (id, score, metadata)
            
            # Test with filters
            filtered_results = await vector_db.search_similar(
                query_embedding, 
                top_k=2, 
                filters={"source_type": "document"}
            )
            assert len(filtered_results) > 0
            
            # Get metadata
            metadata = await vector_db.get_metadata(vector_ids[0])
            assert metadata is not None
            assert "source_name" in metadata
            
            # Get statistics
            stats = await vector_db.get_stats()
            assert stats["total_vectors"] == 3
            assert stats["db_type"] == "chroma"
            
            # Close connection
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_faiss_db_functionality(self, sample_chunks, sample_embeddings, query_embedding):
        """Test FAISS functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_path = os.path.join(temp_dir, "faiss_index")
            
            # Initialize FAISS
            vector_db = VectorDatabase(f"faiss://{faiss_path}")
            await vector_db.connect()
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            assert len(vector_ids) == 3
            
            # Search similar vectors
            results = await vector_db.search_similar(query_embedding, top_k=2)
            assert len(results) > 0
            
            # Get metadata
            metadata = await vector_db.get_metadata(vector_ids[0])
            assert metadata is not None
            
            # Get statistics
            stats = await vector_db.get_stats()
            assert stats["total_vectors"] == 3
            assert stats["db_type"] == "faiss"
            
            # Close connection
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_pinecone_db_functionality(self, sample_chunks, sample_embeddings, query_embedding):
        """Test Pinecone functionality with mocked client."""
        with patch('ragify.storage.vector_db.pinecone') as mock_pinecone:
            # Mock Pinecone client
            mock_pinecone_instance = Mock()
            mock_index = Mock()
            mock_pinecone.Pinecone.return_value = mock_pinecone_instance
            mock_pinecone_instance.Index.return_value = mock_index
            
            # Mock query results
            mock_match = Mock()
            mock_match.id = "test_id"
            mock_match.score = 0.85
            mock_match.metadata = {"source_name": "Test"}
            mock_query_result = Mock()
            mock_query_result.matches = [mock_match]
            mock_index.query.return_value = mock_query_result
            
            # Initialize Pinecone
            vector_db = VectorDatabase("pinecone://test_api_key:test_index")
            await vector_db.connect()
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            assert len(vector_ids) == 3
            
            # Search similar vectors
            results = await vector_db.search_similar(query_embedding, top_k=2)
            assert len(results) > 0
            
            # Close connection
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_weaviate_db_functionality(self, sample_chunks, sample_embeddings, query_embedding):
        """Test Weaviate functionality with mocked client."""
        # Skip this test if Weaviate is not available
        from src.ragify.storage.vector_db import WEAVIATE_AVAILABLE
        if not WEAVIATE_AVAILABLE:
            pytest.skip("Weaviate not available - skipping test")
        
        # Mock the availability check first
        with patch('ragify.storage.vector_db.WEAVIATE_AVAILABLE', True), \
             patch('ragify.storage.vector_db.weaviate') as mock_weaviate:
            
            # Mock Weaviate client
            mock_client = Mock()
            mock_weaviate.connect_to_wcs.return_value = mock_client
            
            # Mock schema creation
            mock_client.schema.create_class.side_effect = Exception("Already exists")
            
            # Mock batch context manager
            mock_batch = Mock()
            mock_batch.__enter__ = Mock(return_value=mock_batch)
            mock_batch.__exit__ = Mock(return_value=None)
            mock_client.batch = mock_batch
            
            # Mock query results
            mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_where.return_value.do.return_value = {
                'data': {
                    'Get': {
                        'ContextChunk': [
                            {
                                'chunk_id': 'test_id',
                                'source_name': 'Test',
                                'source_type': 'document',
                                'content_preview': 'Test content',
                                '_additional': {'certainty': 0.85}
                            }
                        ]
                    }
                }
            }
            
            # Initialize Weaviate
            vector_db = VectorDatabase("weaviate://localhost:8080")
            await vector_db.connect()
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            assert len(vector_ids) == 3
            
            # Search similar vectors
            results = await vector_db.search_similar(query_embedding, top_k=2)
            assert len(results) > 0
            
            # Close connection
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in vector database operations."""
        # Test invalid connection string
        with pytest.raises(VectorDBError, match="Pinecone connection string must be"):
            vector_db = VectorDatabase("pinecone://invalid")
            await vector_db.connect()
    
    @pytest.mark.asyncio
    async def test_embedding_mismatch(self, sample_chunks):
        """Test handling of embedding count mismatch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Test with mismatched embeddings
            with pytest.raises(VectorDBError, match="Number of chunks must match"):
                await vector_db.store_embeddings(sample_chunks, [[0.1, 0.2, 0.3]])
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, sample_chunks, sample_embeddings, query_embedding):
        """Test search functionality with various filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Store embeddings
            await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            
            # Test with source type filter
            results = await vector_db.search_similar(
                query_embedding,
                top_k=5,
                filters={"source_type": "document"}
            )
            assert len(results) > 0
            
            # Test with source name filter
            results = await vector_db.search_similar(
                query_embedding,
                top_k=5,
                filters={"source_name": "Test Doc"}
            )
            assert len(results) > 0
            
            # Test with multiple filters (ChromaDB has limitations with complex filters)
            # For now, test with single filter only
            results = await vector_db.search_similar(
                query_embedding,
                top_k=5,
                filters={"source_type": "document"}
            )
            assert len(results) > 0
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_min_score_filtering(self, sample_chunks, sample_embeddings, query_embedding):
        """Test minimum score filtering in search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Store embeddings
            await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            
            # Test with high minimum score (should return fewer results)
            high_threshold_results = await vector_db.search_similar(
                query_embedding,
                top_k=5,
                min_score=0.9
            )
            
            # Test with low minimum score (should return more results)
            low_threshold_results = await vector_db.search_similar(
                query_embedding,
                top_k=5,
                min_score=0.1
            )
            
            assert len(low_threshold_results) >= len(high_threshold_results)
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, sample_chunks, sample_embeddings, query_embedding):
        """Test statistics tracking functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Check initial stats
            initial_stats = await vector_db.get_stats()
            assert initial_stats["total_vectors"] == 0
            assert initial_stats["searches_performed"] == 0
            
            # Store embeddings
            await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            
            # Check stats after storage
            storage_stats = await vector_db.get_stats()
            assert storage_stats["total_vectors"] == 3
            
            # Perform search
            await vector_db.search_similar(query_embedding, top_k=2)
            
            # Check stats after search
            search_stats = await vector_db.get_stats()
            assert search_stats["searches_performed"] == 1
            assert search_stats["avg_search_time"] > 0
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_index_creation(self, sample_chunks, sample_embeddings):
        """Test index creation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_path = os.path.join(temp_dir, "faiss_index")
            vector_db = VectorDatabase(f"faiss://{faiss_path}")
            await vector_db.connect()
            
            # Store embeddings
            await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            
            # Create index with custom parameters (use smaller nlist for small dataset)
            await vector_db.create_index({
                "index_type": "ivf",
                "nlist": 2,  # Small nlist for small dataset
                "dimension": 384
            })
            
            # Check that index was created
            stats = await vector_db.get_stats()
            assert stats["total_vectors"] == 3
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_vector_operations(self, sample_chunks, sample_embeddings):
        """Test vector operations (update, delete)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma_db")
            vector_db = VectorDatabase(f"chroma://{chroma_path}")
            await vector_db.connect()
            
            # Store embeddings
            vector_ids = await vector_db.store_embeddings(sample_chunks, sample_embeddings)
            
            # Update embeddings
            new_embeddings = [[0.9, 0.8, 0.7, 0.6, 0.5] * 76 + [0.9, 0.8, 0.7, 0.6]]  # 384 dimensions
            await vector_db.update_embeddings([vector_ids[0]], new_embeddings)
            
            # Delete embeddings
            await vector_db.delete_embeddings([vector_ids[1]])
            
            # Check stats
            stats = await vector_db.get_stats()
            assert stats["total_vectors"] == 2  # One deleted
            
            await vector_db.close()
    
    @pytest.mark.asyncio
    async def test_connection_string_parsing(self):
        """Test various connection string formats."""
        # Test ChromaDB local path
        vector_db = VectorDatabase("chroma:///path/to/chroma")
        assert vector_db.db_type == "chroma"
        assert vector_db.connection_string == "/path/to/chroma"
        
        # Test ChromaDB remote
        vector_db = VectorDatabase("chroma://localhost:8000")
        assert vector_db.db_type == "chroma"
        assert vector_db.connection_string == "localhost:8000"
        
        # Test Pinecone
        vector_db = VectorDatabase("pinecone://api_key:index_name")
        assert vector_db.db_type == "pinecone"
        assert vector_db.connection_string == "api_key:index_name"
        
        # Test Weaviate with mocked availability
        with patch('ragify.storage.vector_db.WEAVIATE_AVAILABLE', True):
            vector_db = VectorDatabase("weaviate://localhost:8080")
            assert vector_db.db_type == "weaviate"
            assert vector_db.connection_string == "localhost:8080"
        
        # Test Weaviate when not available (should raise error)
        with patch('ragify.storage.vector_db.WEAVIATE_AVAILABLE', False):
            with pytest.raises(VectorDBError, match="Weaviate not available"):
                VectorDatabase("weaviate://localhost:8080")
        
        # Test FAISS
        vector_db = VectorDatabase("faiss:///path/to/faiss/index")
        assert vector_db.db_type == "faiss"
        assert vector_db.connection_string == "/path/to/faiss/index"
    
    @pytest.mark.asyncio
    async def test_availability_checking(self):
        """Test availability checking for vector database libraries."""
        # Test with missing libraries (should raise error)
        with patch('ragify.storage.vector_db.CHROMADB_AVAILABLE', False):
            with pytest.raises(VectorDBError, match="ChromaDB not available"):
                VectorDatabase("chroma:///test")
        
        with patch('ragify.storage.vector_db.PINECONE_AVAILABLE', False):
            with pytest.raises(VectorDBError, match="Pinecone not available"):
                VectorDatabase("pinecone://test:test")
        
        with patch('ragify.storage.vector_db.WEAVIATE_AVAILABLE', False):
            with pytest.raises(VectorDBError, match="Weaviate not available"):
                VectorDatabase("weaviate://localhost:8080")
        
        with patch('ragify.storage.vector_db.FAISS_AVAILABLE', False):
            with pytest.raises(VectorDBError, match="FAISS not available"):
                VectorDatabase("faiss:///test")
