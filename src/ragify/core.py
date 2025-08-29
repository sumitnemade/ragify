"""
Core Context Orchestrator implementation.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import structlog
from pydantic import ValidationError

from .engines import (
    IntelligentContextFusionEngine,
    ContextScoringEngine,
    ContextStorageEngine,
    ContextUpdatesEngine,
)
from .exceptions import (
    ICOException,
    ContextNotFoundError,
    ConfigurationError,
    PrivacyViolationError,
)
from .models import (
    Context,
    ContextRequest,
    ContextResponse,
    ContextSource,
    ContextChunk,
    OrchestratorConfig,
    PrivacyLevel,
)
from .sources import BaseDataSource
from .storage import CacheManager, PrivacyManager, VectorDatabase


class ContextOrchestrator:
    """
    Main orchestrator for intelligent context management.
    
    Coordinates context fusion, scoring, storage, and updates across multiple
    data sources with privacy controls and real-time synchronization.
    """
    
    def __init__(
        self,
        vector_db_url: Optional[str] = None,
        cache_url: Optional[str] = None,
        privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize the context orchestrator.
        
        Args:
            vector_db_url: URL for vector database connection
            cache_url: URL for cache connection
            privacy_level: Default privacy level for operations
            config: Full configuration object
        """
        self.config = config or OrchestratorConfig(
            vector_db_url=vector_db_url,
            cache_url=cache_url,
            privacy_level=privacy_level,
        )
        
        # Initialize logging
        self.logger = structlog.get_logger(__name__)
        
        # Initialize core components
        self._initialize_components()
        
        # Data sources registry
        self._sources: Dict[str, BaseDataSource] = {}
        
        # Engines
        self.fusion_engine = IntelligentContextFusionEngine(self.config)
        self.scoring_engine = ContextScoringEngine(self.config)
        self.storage_engine = ContextStorageEngine(self.config)
        self.updates_engine = ContextUpdatesEngine(self.config)
        
        self.logger.info("Context Orchestrator initialized", config=self.config.model_dump())
    
    def _initialize_components(self) -> None:
        """Initialize core components."""
        try:
            # Vector database
            if self.config.vector_db_url:
                self.vector_db = VectorDatabase(self.config.vector_db_url)
            else:
                self.vector_db = None
                self.logger.warning("No vector database configured")
            
            # Cache manager
            if self.config.cache_url:
                self.cache_manager = CacheManager(self.config.cache_url)
            else:
                self.cache_manager = None
                self.logger.warning("No cache configured")
            
            # Privacy manager
            self.privacy_manager = PrivacyManager(self.config.privacy_level)
            
        except Exception as e:
            raise ConfigurationError("component_initialization", str(e), str(e))
    
    def add_source(self, source: BaseDataSource) -> None:
        """
        Add a data source to the orchestrator.
        
        Args:
            source: Data source instance
        """
        if source.name in self._sources:
            self.logger.warning(f"Source '{source.name}' already exists, replacing")
        
        self._sources[source.name] = source
        self.logger.info(f"Added data source: {source.name}", source_type=source.source_type)
    
    def remove_source(self, source_name: str) -> None:
        """
        Remove a data source from the orchestrator.
        
        Args:
            source_name: Name of the source to remove
        """
        if source_name in self._sources:
            del self._sources[source_name]
            self.logger.info(f"Removed data source: {source_name}")
        else:
            self.logger.warning(f"Source '{source_name}' not found")
    
    def get_source(self, source_name: str) -> Optional[BaseDataSource]:
        """
        Get a data source by name.
        
        Args:
            source_name: Name of the source
            
        Returns:
            Data source instance or None if not found
        """
        return self._sources.get(source_name)
    
    def list_sources(self) -> List[str]:
        """
        List all registered data sources.
        
        Returns:
            List of source names
        """
        return list(self._sources.keys())
    
    async def get_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_chunks: Optional[int] = None,
        min_relevance: Optional[float] = None,
        privacy_level: Optional[PrivacyLevel] = None,
        include_metadata: bool = True,
        sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> ContextResponse:
        """
        Get intelligent context for a query.
        
        Args:
            query: The user query
            user_id: User identifier for personalization
            session_id: Session identifier for continuity
            max_tokens: Maximum tokens in context
            max_chunks: Maximum number of chunks
            min_relevance: Minimum relevance threshold
            privacy_level: Privacy level for this request
            include_metadata: Whether to include metadata
            sources: Specific sources to include
            exclude_sources: Sources to exclude
            
        Returns:
            ContextResponse with intelligent context
        """
        start_time = time.time()
        
        try:
            # Create request object
            request = ContextRequest(
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_tokens=max_tokens or self.config.max_context_size,
                max_chunks=max_chunks,
                min_relevance=min_relevance or self.config.default_relevance_threshold,
                privacy_level=privacy_level or self.config.privacy_level,
                include_metadata=include_metadata,
                sources=sources,
                exclude_sources=exclude_sources,
            )
            
            self.logger.info(
                "Processing context request",
                query=query,
                user_id=user_id,
                session_id=session_id,
                max_tokens=request.max_tokens,
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                return ContextResponse(
                    context=cached_response,
                    processing_time=time.time() - start_time,
                    cache_hit=True,
                )
            
            # Get context from sources
            context = await self._retrieve_context(request)
            
            # Apply privacy controls
            context = await self._apply_privacy_controls(context, request.privacy_level)
            
            # Store in cache
            await self._store_in_cache(cache_key, context)
            
            processing_time = time.time() - start_time
            self.logger.info(
                "Context retrieved successfully",
                processing_time=processing_time,
                chunks_count=len(context.chunks),
                total_tokens=context.total_tokens,
            )
            
            return ContextResponse(
                context=context,
                processing_time=processing_time,
                cache_hit=False,
            )
            
        except Exception as e:
            self.logger.error("Failed to get context", error=str(e), query=query)
            raise
    
    async def _retrieve_context(self, request: ContextRequest) -> Context:
        """Retrieve context from all sources using concurrent processing."""
        # Get relevant sources
        sources = self._filter_sources(request.sources, request.exclude_sources)
        
        if not sources:
            raise ContextNotFoundError(request.query, request.user_id)
        
        # Retrieve chunks from all sources concurrently
        all_chunks = await self._retrieve_context_concurrent(request, sources)
        
        if not all_chunks:
            raise ContextNotFoundError(request.query, request.user_id)
        
        # Score relevance
        scored_chunks = await self.scoring_engine.score_chunks(
            chunks=all_chunks,
            query=request.query,
        )
        
        # Filter by relevance
        relevant_chunks = [
            chunk for chunk in scored_chunks
            if chunk.relevance_score and chunk.relevance_score.score >= request.min_relevance
        ]
        
        # Create context
        context = Context(
            query=request.query,
            chunks=relevant_chunks,
            user_id=request.user_id,
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            privacy_level=request.privacy_level,
        )
        
        # Optimize for token limit
        if request.max_tokens:
            context.optimize_for_tokens(request.max_tokens)
        
        return context
    
    async def _retrieve_context_concurrent(self, request: ContextRequest, sources: Dict[str, BaseDataSource]) -> List[ContextChunk]:
        """Retrieve context from all sources concurrently for optimal performance."""
        if not sources:
            return []
        
        # Create concurrent tasks for all sources
        tasks = []
        source_names = []
        
        for source_name, source in sources.items():
            task = self._get_chunks_from_source(
                source_name=source_name,
                source=source,
                request=request
            )
            tasks.append(task)
            source_names.append(source_name)
        
        # Execute all sources concurrently with timeout
        try:
            self.logger.info(f"Processing {len(sources)} sources concurrently")
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.source_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Source processing timeout, using partial results")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and handle errors gracefully
        all_chunks = []
        successful_sources = 0
        failed_sources = 0
        
        for i, result in enumerate(results):
            source_name = source_names[i]
            if isinstance(result, list):
                all_chunks.extend(result)
                successful_sources += 1
                self.logger.info(f"Source {source_name} returned {len(result)} chunks")
            else:
                failed_sources += 1
                if isinstance(result, Exception):
                    self.logger.warning(f"Source {source_name} failed: {result}")
                else:
                    self.logger.warning(f"Source {source_name} returned unexpected result type: {type(result)}")
        
        self.logger.info(f"Concurrent processing completed: {successful_sources} successful, {failed_sources} failed, total chunks: {len(all_chunks)}")
        return all_chunks
    
    async def _get_chunks_from_source(
        self, 
        source_name: str, 
        source: BaseDataSource, 
        request: ContextRequest
    ) -> List[ContextChunk]:
        """Get chunks from a single source with error handling."""
        try:
            chunks = await source.get_chunks(
                query=request.query,
                max_chunks=request.max_chunks,
                min_relevance=request.min_relevance,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            return chunks if chunks else []
        except Exception as e:
            self.logger.warning(
                f"Failed to get chunks from source {source_name}",
                error=str(e),
            )
            # Return empty list instead of raising to allow other sources to continue
            return []
    
    def _filter_sources(
        self,
        include_sources: Optional[List[str]],
        exclude_sources: Optional[List[str]],
    ) -> Dict[str, BaseDataSource]:
        """Filter sources based on include/exclude lists."""
        sources = self._sources.copy()
        
        if include_sources:
            sources = {k: v for k, v in sources.items() if k in include_sources}
        
        if exclude_sources:
            sources = {k: v for k, v in sources.items() if k not in exclude_sources}
        
        return sources
    
    async def _apply_privacy_controls(
        self,
        context: Context,
        privacy_level: PrivacyLevel,
    ) -> Context:
        """Apply privacy controls to context."""
        return await self.privacy_manager.apply_privacy_controls(context, privacy_level)
    
    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate cache key for request."""
        # Create a deterministic key based on request parameters
        key_parts = [
            request.query,
            request.user_id or "",
            request.session_id or "",
            str(request.max_tokens or ""),
            str(request.min_relevance),
            str(request.privacy_level),
        ]
        
        if request.sources:
            key_parts.extend(sorted(request.sources))
        
        if request.exclude_sources:
            key_parts.extend(sorted(request.exclude_sources))
        
        return f"context:{hash(tuple(key_parts))}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Context]:
        """Get context from cache."""
        if not self.cache_manager:
            return None
        
        try:
            return await self.cache_manager.get(cache_key)
        except Exception as e:
            self.logger.warning("Cache get failed", error=str(e))
            return None
    
    async def _store_in_cache(self, cache_key: str, context: Context) -> None:
        """Store context in cache."""
        if not self.cache_manager:
            return
        
        try:
            await self.cache_manager.set(
                cache_key,
                context,
                ttl=self.config.cache_ttl,
            )
        except Exception as e:
            self.logger.warning("Cache set failed", error=str(e))
    
    async def update_context(
        self,
        context_id: UUID,
        updates: Dict[str, Any],
    ) -> Context:
        """
        Update an existing context.
        
        Args:
            context_id: ID of the context to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated context
        """
        return await self.storage_engine.update_context(context_id, updates)
    
    async def delete_context(self, context_id: UUID) -> None:
        """
        Delete a context.
        
        Args:
            context_id: ID of the context to delete
        """
        await self.storage_engine.delete_context(context_id)
    
    async def get_context_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Context]:
        """
        Get context history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of contexts to return
            
        Returns:
            List of historical contexts
        """
        return await self.storage_engine.get_context_history(user_id, limit)
    
    async def refresh_sources(self) -> None:
        """Refresh all data sources."""
        self.logger.info("Refreshing all data sources")
        
        for source_name, source in self._sources.items():
            try:
                await source.refresh()
                self.logger.info(f"Refreshed source: {source_name}")
            except Exception as e:
                self.logger.error(f"Failed to refresh source {source_name}", error=str(e))
    
    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about context usage.
        
        Returns:
            Dictionary with analytics data
        """
        # Integrate with analytics engine for performance tracking
        if hasattr(self, 'analytics_engine'):
            try:
                await self.analytics_engine.track_context_retrieval(
                    query=request.query,
                    user_id=request.user_id,
                    chunk_count=len(response.chunks),
                    processing_time=response.processing_time,
                    sources_used=[chunk.source.name for chunk in response.chunks]
                )
            except Exception as e:
                self.logger.warning(f"Failed to track analytics: {e}")
        return {
            "total_contexts": len(self._sources),
            "cache_hit_rate": 0.0,  # Would be calculated from cache stats
            "average_processing_time": 0.0,  # Would be calculated from metrics
            "source_usage": {name: 0 for name in self._sources.keys()},
        }
    
    async def close(self) -> None:
        """Close the orchestrator and clean up resources."""
        self.logger.info("Closing Context Orchestrator")
        
        # Close all sources
        for source in self._sources.values():
            try:
                await source.close()
            except Exception as e:
                self.logger.warning(f"Error closing source {source.name}", error=str(e))
        
        # Close components
        if self.vector_db:
            await self.vector_db.close()
        
        if self.cache_manager:
            await self.cache_manager.close()
        
        self.logger.info("Context Orchestrator closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
