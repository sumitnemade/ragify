"""
Base data source class for the Intelligent Context Orchestration plugin.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import structlog

from ..models import ContextChunk, SourceType, ContextSource


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.
    
    Defines the interface that all data sources must implement
    for integration with the context orchestrator.
    """
    
    def __init__(self, name: str, source_type: SourceType, **kwargs):
        """
        Initialize the base data source.
        
        Args:
            name: Name of the data source
            source_type: Type of data source
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.source_type = source_type
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
        # Source configuration
        self.config = kwargs
        
        # Source metadata
        self.metadata = {
            'created_at': None,
            'last_updated': None,
            'total_chunks': 0,
            'is_active': True,
        }
        
        # Create source object
        self.source = ContextSource(
            name=name,
            source_type=source_type,
            metadata=self.metadata,
        )
    
    @abstractmethod
    async def get_chunks(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Get context chunks from this data source.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            min_relevance: Minimum relevance threshold
            user_id: Optional user ID for personalization
            session_id: Optional session ID for continuity
            
        Returns:
            List of context chunks
        """
        pass
    
    @abstractmethod
    async def refresh(self) -> None:
        """
        Refresh the data source.
        
        This method should update the source's data and metadata.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Close the data source and clean up resources.
        """
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the data source.
        
        Returns:
            Dictionary with source statistics
        """
        return {
            'name': self.name,
            'type': self.source_type.value,
            'metadata': self.metadata,
            'config': self.config,
        }
    
    async def is_available(self) -> bool:
        """
        Check if the data source is available.
        
        Returns:
            True if available, False otherwise
        """
        return self.metadata.get('is_active', True)
    
    async def get_source_info(self) -> ContextSource:
        """
        Get information about this data source.
        
        Returns:
            ContextSource object with source information
        """
        return self.source
    
    async def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Update source metadata.
        
        Args:
            updates: Dictionary of metadata updates
        """
        self.metadata.update(updates)
        self.source.metadata = self.metadata
        self.logger.info(f"Updated metadata: {updates}")
    
    async def _create_chunk(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None,
    ) -> ContextChunk:
        """
        Create a context chunk from this source.
        
        Args:
            content: Chunk content
            metadata: Optional chunk metadata
            token_count: Optional token count
            
        Returns:
            ContextChunk object
        """
        return ContextChunk(
            content=content,
            source=self.source,
            metadata=metadata or {},
            token_count=token_count,
        )
    
    async def _validate_query(self, query: str) -> str:
        """
        Validate and normalize a query.
        
        Args:
            query: Query to validate
            
        Returns:
            Normalized query
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        return query.strip()
    
    async def _apply_filters(
        self,
        chunks: List[ContextChunk],
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
    ) -> List[ContextChunk]:
        """
        Apply filters to chunks.
        
        Args:
            chunks: List of chunks to filter
            max_chunks: Maximum number of chunks
            min_relevance: Minimum relevance threshold
            
        Returns:
            Filtered list of chunks
        """
        # Filter by relevance
        if min_relevance > 0.0:
            chunks = [
                chunk for chunk in chunks
                if chunk.relevance_score and chunk.relevance_score.score >= min_relevance
            ]
        
        # Limit number of chunks
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        return chunks
    
    async def _update_stats(self, chunks_count: int) -> None:
        """
        Update source statistics.
        
        Args:
            chunks_count: Number of chunks processed
        """
        self.metadata['total_chunks'] = chunks_count
        self.metadata['last_updated'] = None  # Would be set to current time
        self.source.metadata = self.metadata
