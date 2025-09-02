"""
Base data source class for the Intelligent Context Orchestration plugin.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import structlog
from datetime import datetime, timezone
import asyncio

from ..models import ContextChunk, SourceType, ContextSource, RelevanceScore


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
            'created_at': datetime.now(timezone.utc),
            'last_updated': datetime.now(timezone.utc),
            'total_chunks': 0,
            'is_active': True,
            'connection_status': 'disconnected',
            'last_refresh': None,
            'refresh_count': 0,
            'error_count': 0,
            'total_queries': 0,
            'successful_queries': 0,
        }
        
        # Create source object
        self.source = ContextSource(
            name=name,
            source_type=source_type,
            metadata=self.metadata,
        )
        
        # Internal state
        self._is_connected = False
        self._is_refreshing = False
        self._last_error = None
        self._connection_lock = asyncio.Lock()
    
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
        
        This is a base implementation that should be overridden by subclasses
        for specific data source logic.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            min_relevance: Minimum relevance threshold
            user_id: Optional user ID for personalization
            session_id: Optional session ID for continuity
            
        Returns:
            List of context chunks
        """
        try:
            # Update query statistics
            self.metadata['total_queries'] += 1
            
            # Validate query
            validated_query = await self._validate_query(query)
            
            # Check if source is available
            if not await self.is_available():
                self.logger.warning(f"Source {self.name} is not available")
                return []
            
            # Ensure connection
            if not self._is_connected:
                await self._ensure_connection()
            
            # Get chunks from subclass implementation
            chunks = await self._get_chunks_impl(
                validated_query, max_chunks, min_relevance, user_id, session_id
            )
            
            # Apply filters
            filtered_chunks = await self._apply_filters(chunks, max_chunks, min_relevance)
            
            # Update statistics
            self.metadata['successful_queries'] += 1
            self.metadata['last_updated'] = datetime.now(timezone.utc)
            await self._update_stats(len(filtered_chunks))
            
            self.logger.info(
                f"Retrieved {len(filtered_chunks)} chunks from {self.name}",
                query=validated_query[:100],
                user_id=user_id,
                chunks_count=len(filtered_chunks)
            )
            
            return filtered_chunks
            
        except Exception as e:
            # Update error statistics
            self.metadata['error_count'] += 1
            self._last_error = str(e)
            self.logger.error(f"Failed to get chunks from {self.name}: {e}")
            raise
    
    async def refresh(self) -> None:
        """
        Refresh the data source.
        
        This method should update the source's data and metadata.
        """
        if self._is_refreshing:
            self.logger.info(f"Refresh already in progress for {self.name}")
            return
        
        try:
            self._is_refreshing = True
            self.metadata['refresh_count'] += 1
            
            self.logger.info(f"Starting refresh of data source: {self.name}")
            
            # Perform refresh implementation
            await self._refresh_impl()
            
            # Update metadata
            self.metadata['last_refresh'] = datetime.now(timezone.utc)
            self.metadata['last_updated'] = datetime.now(timezone.utc)
            self.metadata['connection_status'] = 'connected'
            
            self.logger.info(f"Successfully refreshed data source: {self.name}")
            
        except Exception as e:
            self.metadata['error_count'] += 1
            self._last_error = str(e)
            self.metadata['connection_status'] = 'error'
            self.logger.error(f"Failed to refresh data source {self.name}: {e}")
            raise
        finally:
            self._is_refreshing = False
    
    async def close(self) -> None:
        """
        Close the data source and clean up resources.
        """
        try:
            self.logger.info(f"Closing data source: {self.name}")
            
            # Perform cleanup implementation
            await self._close_impl()
            
            # Update metadata
            self.metadata['is_active'] = False
            self.metadata['connection_status'] = 'disconnected'
            self._is_connected = False
            
            self.logger.info(f"Successfully closed data source: {self.name}")
            
        except Exception as e:
            self.metadata['error_count'] += 1
            self._last_error = str(e)
            self.logger.error(f"Error closing data source {self.name}: {e}")
            raise
    
    async def _ensure_connection(self) -> None:
        """Ensure the data source is connected."""
        async with self._connection_lock:
            if not self._is_connected:
                try:
                    await self._connect_impl()
                    self._is_connected = True
                    self.metadata['connection_status'] = 'connected'
                    self.logger.info(f"Connected to data source: {self.name}")
                except Exception as e:
                    self.metadata['connection_status'] = 'error'
                    self._last_error = str(e)
                    self.logger.error(f"Failed to connect to {self.name}: {e}")
                    raise
    
    async def _get_chunks_impl(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Implementation of chunk retrieval.
        
        This method should be overridden by subclasses to provide
        specific data source logic.
        
        Args:
            query: Validated search query
            max_chunks: Maximum number of chunks
            min_relevance: Minimum relevance threshold
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            List of context chunks
        """
        # Base implementation returns empty list
        # Subclasses should override this method
        return []
    
    async def _refresh_impl(self) -> None:
        """
        Implementation of data source refresh.
        
        This method should be overridden by subclasses to provide
        specific refresh logic.
        """
        # Base implementation does nothing
        # Subclasses should override this method
        pass
    
    async def _close_impl(self) -> None:
        """
        Implementation of data source cleanup.
        
        This method should be overridden by subclasses to provide
        specific cleanup logic.
        """
        # Base implementation does nothing
        # Subclasses should override this method
        pass
    
    async def _connect_impl(self) -> None:
        """
        Implementation of data source connection.
        
        This method should be overridden by subclasses to provide
        specific connection logic.
        """
        # Base implementation does nothing
        # Subclasses should override this method
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
            'connection_status': self._is_connected,
            'is_refreshing': self._is_refreshing,
            'last_error': self._last_error,
        }
    
    async def is_available(self) -> bool:
        """
        Check if the data source is available.
        
        Returns:
            True if available, False otherwise
        """
        return (
            self.metadata.get('is_active', True) and
            self.metadata.get('connection_status') != 'error'
        )
    
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
        relevance_score: Optional[float] = None,
    ) -> ContextChunk:
        """
        Create a context chunk from this source.
        
        Args:
            content: Chunk content
            metadata: Optional chunk metadata
            token_count: Optional token count
            relevance_score: Optional relevance score
            
        Returns:
            ContextChunk object
        """
        # Create relevance score if provided
        relevance = None
        if relevance_score is not None:
            relevance = RelevanceScore(score=relevance_score)
        
        return ContextChunk(
            content=content,
            source=self.source,
            metadata=metadata or {},
            token_count=token_count,
            relevance_score=relevance,
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
        self.metadata['last_updated'] = datetime.now(timezone.utc)
        self.source.metadata = self.metadata
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the data source.
        
        Returns:
            Dictionary with health information
        """
        return {
            'name': self.name,
            'is_active': self.metadata.get('is_active', False),
            'connection_status': self.metadata.get('connection_status', 'unknown'),
            'is_connected': self._is_connected,
            'is_refreshing': self._is_refreshing,
            'last_error': self._last_error,
            'total_queries': self.metadata.get('total_queries', 0),
            'successful_queries': self.metadata.get('successful_queries', 0),
            'error_count': self.metadata.get('error_count', 0),
            'last_refresh': self.metadata.get('last_refresh'),
            'last_updated': self.metadata.get('last_updated'),
        }
    
    async def reset_stats(self) -> None:
        """Reset all statistics for the data source."""
        self.metadata.update({
            'total_queries': 0,
            'successful_queries': 0,
            'error_count': 0,
            'refresh_count': 0,
            'total_chunks': 0,
        })
        self._last_error = None
        self.logger.info(f"Reset statistics for data source: {self.name}")
