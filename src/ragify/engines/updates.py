"""
Context Updates Engine for real-time context synchronization.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
import structlog

from ..models import Context, ContextChunk, OrchestratorConfig
from ..exceptions import ICOException


class ContextUpdatesEngine:
    """
    Real-time context updates engine for live synchronization.
    
    Handles real-time updates from external sources, change detection,
    and automatic context refresh.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the updates engine.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Update subscriptions
        self.subscriptions: Dict[str, List[Callable]] = {}
        
        # Update queues
        self.update_queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks = set()
        
        # Update policies
        self.update_policies = {
            'max_concurrent_updates': 10,
            'update_timeout': 30,  # seconds
            'retry_attempts': 3,
            'retry_delay': 5,  # seconds
            'batch_size': 100,
            'refresh_interval_minutes': 30,
        }
        
        # Caches for tracking updates
        self._context_cache = {}
        self._source_modification_cache = {}
        self._source_check_cache = {}
        self._monitored_sources = {}
    
    async def start(self) -> None:
        """Start the updates engine."""
        self.logger.info("Starting Context Updates Engine")
        
        # Start background tasks
        self.background_tasks.add(
            asyncio.create_task(self._process_update_queue())
        )
        self.background_tasks.add(
            asyncio.create_task(self._monitor_sources())
        )
        
        self.logger.info("Context Updates Engine started")
    
    async def stop(self) -> None:
        """Stop the updates engine."""
        self.logger.info("Stopping Context Updates Engine")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        self.logger.info("Context Updates Engine stopped")
    
    async def subscribe_to_updates(
        self,
        source_name: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to updates from a specific source.
        
        Args:
            source_name: Name of the source to subscribe to
            callback: Callback function to call when updates occur
        """
        if source_name not in self.subscriptions:
            self.subscriptions[source_name] = []
        
        self.subscriptions[source_name].append(callback)
        self.logger.info(f"Subscribed to updates from {source_name}")
    
    async def unsubscribe_from_updates(
        self,
        source_name: str,
        callback: Callable,
    ) -> None:
        """
        Unsubscribe from updates from a specific source.
        
        Args:
            source_name: Name of the source to unsubscribe from
            callback: Callback function to remove
        """
        if source_name in self.subscriptions:
            try:
                self.subscriptions[source_name].remove(callback)
                self.logger.info(f"Unsubscribed from updates from {source_name}")
            except ValueError:
                self.logger.warning(f"Callback not found in subscriptions for {source_name}")
    
    async def trigger_update(
        self,
        source_name: str,
        update_data: Dict[str, Any],
    ) -> None:
        """
        Trigger an update from a source.
        
        Args:
            source_name: Name of the source
            update_data: Update data to process
        """
        try:
            # Add to update queue
            await self.update_queue.put({
                'source_name': source_name,
                'update_data': update_data,
                'timestamp': datetime.now(timezone.utc),
            })
            
            self.logger.info(f"Queued update from {source_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger update from {source_name}: {e}")
    
    async def refresh_context(
        self,
        context: Context,
        force: bool = False,
    ) -> Context:
        """
        Refresh a context with latest data from sources.
        
        Args:
            context: Context to refresh
            force: Force refresh even if not needed
            
        Returns:
            Refreshed context
        """
        try:
            self.logger.info(f"Refreshing context {context.id}")
            
            # Check if refresh is needed
            if not force and not await self._needs_refresh(context):
                self.logger.info(f"Context {context.id} does not need refresh")
                return context
            
            # Get updated chunks from sources
            updated_chunks = []
            for chunk in context.chunks:
                updated_chunk = await self._refresh_chunk(chunk)
                if updated_chunk:
                    updated_chunks.append(updated_chunk)
                else:
                    updated_chunks.append(chunk)  # Keep original if no update
            
            # Create refreshed context
            refreshed_context = Context(
                id=context.id,
                query=context.query,
                chunks=updated_chunks,
                user_id=context.user_id,
                session_id=context.session_id,
                relevance_score=context.relevance_score,
                total_tokens=sum(c.token_count or 0 for c in updated_chunks),
                max_tokens=context.max_tokens,
                created_at=context.created_at,
                expires_at=context.expires_at,
                privacy_level=context.privacy_level,
                metadata={
                    **context.metadata,
                    'refreshed_at': datetime.now(timezone.utc).isoformat(),
                    'refresh_count': context.metadata.get('refresh_count', 0) + 1,
                },
            )
            
            self.logger.info(f"Successfully refreshed context {context.id}")
            return refreshed_context
            
        except Exception as e:
            self.logger.error(f"Failed to refresh context {context.id}: {e}")
            raise ICOException(f"Context refresh failed: {e}")
    
    async def _process_update_queue(self) -> None:
        """Process updates from the queue."""
        while True:
            try:
                # Get update from queue
                update = await self.update_queue.get()
                
                # Process update
                await self._process_update(update)
                
                # Mark as done
                self.update_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing update: {e}")
    
    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single update."""
        try:
            source_name = update['source_name']
            update_data = update['update_data']
            
            self.logger.info(f"Processing update from {source_name}")
            
            # Notify subscribers
            if source_name in self.subscriptions:
                for callback in self.subscriptions[source_name]:
                    try:
                        await callback(update_data)
                    except Exception as e:
                        self.logger.error(f"Error in update callback: {e}")
            
            # Process update data
            await self._apply_update_data(source_name, update_data)
            
            self.logger.info(f"Successfully processed update from {source_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process update: {e}")
    
    async def _apply_update_data(
        self,
        source_name: str,
        update_data: Dict[str, Any],
    ) -> None:
        """Apply update data to affected contexts."""
        try:
            self.logger.info(f"Applying update data from {source_name}")
            
            # Get affected contexts based on update data
            affected_contexts = await self._find_affected_contexts(source_name, update_data)
            
            # Update each affected context
            for context_id in affected_contexts:
                try:
                    # Get the context
                    context = await self._get_context_by_id(context_id)
                    if not context:
                        continue
                    
                    # Apply the update
                    updated_context = await self._apply_context_update(context, update_data)
                    
                    # Store the updated context
                    await self._store_updated_context(updated_context)
                    
                    self.logger.info(f"Updated context {context_id} with data from {source_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to update context {context_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply update data from {source_name}: {e}")
    
    async def _find_affected_contexts(self, source_name: str, update_data: Dict[str, Any]) -> List[str]:
        """Find contexts affected by an update."""
        try:
            affected_contexts = []
            
            # Search for contexts that contain chunks from this source
            # This is a simplified implementation - in production, you'd use a proper search index
            for context_id, context in self._context_cache.items():
                for chunk in context.chunks:
                    if chunk.source.name == source_name:
                        # Check if the update affects this chunk
                        if await self._chunk_affected_by_update(chunk, update_data):
                            affected_contexts.append(context_id)
                            break
            
            return affected_contexts
            
        except Exception as e:
            self.logger.error(f"Failed to find affected contexts: {e}")
            return []
    
    async def _chunk_affected_by_update(self, chunk: ContextChunk, update_data: Dict[str, Any]) -> bool:
        """Check if a chunk is affected by an update."""
        try:
            # Check if the update affects this specific chunk
            # This could be based on content similarity, metadata matching, etc.
            
            # Simple check: if the update contains keywords from the chunk
            update_content = str(update_data.get('content', ''))
            chunk_content = chunk.content.lower()
            
            # Extract keywords from update content
            update_keywords = set(update_content.lower().split())
            chunk_keywords = set(chunk_content.split())
            
            # If there's significant keyword overlap, consider it affected
            overlap = len(update_keywords.intersection(chunk_keywords))
            if overlap > 0:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check chunk affected by update: {e}")
            return False
    
    async def _monitor_sources(self) -> None:
        """Monitor sources for changes."""
        while True:
            try:
                # Check for source changes
                await self._check_source_changes()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring sources: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _check_source_changes(self) -> None:
        """Check for changes in monitored sources."""
        try:
            self.logger.debug("Checking for source changes")
            
            # Check each monitored source for changes
            for source_name, source in self._monitored_sources.items():
                try:
                    if hasattr(source, 'has_updates'):
                        has_updates = await source.has_updates()
                        if has_updates:
                            self.logger.info(f"Source {source_name} has updates")
                            # Trigger update processing
                            await self.trigger_update(source_name, {
                                'type': 'content_update',
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'source': source_name
                            })
                    elif hasattr(source, 'get_last_modified'):
                        last_modified = await source.get_last_modified()
                        cached_time = self._source_modification_cache.get(source_name)
                        if cached_time and last_modified > cached_time:
                            self.logger.info(f"Source {source_name} modified at {last_modified}")
                            self._source_modification_cache[source_name] = last_modified
                            # Trigger update processing
                            await self.trigger_update(source_name, {
                                'type': 'modification',
                                'timestamp': last_modified.isoformat(),
                                'source': source_name
                            })
                except Exception as e:
                    self.logger.warning(f"Error checking source {source_name}: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error checking source changes: {e}")
    
    async def _needs_refresh(self, context: Context) -> bool:
        """Check if a context needs refresh."""
        try:
            # Check last refresh time
            last_refresh = context.metadata.get('refreshed_at')
            if last_refresh:
                last_refresh_time = datetime.fromisoformat(last_refresh)
                if datetime.now(timezone.utc) - last_refresh_time < timedelta(minutes=5):
                    return False
            
            # Check if any sources have updates
            for chunk in context.chunks:
                if await self._source_has_updates(chunk.source):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking if context needs refresh: {e}")
            return False
    
    async def _source_has_updates(self, source) -> bool:
        """Check if a source has updates."""
        try:
            # Check source-specific update indicators
            if hasattr(source, 'get_last_modified'):
                last_modified = await source.get_last_modified()
                if last_modified:
                    # Compare with cached last modified time
                    cached_time = self._source_modification_cache.get(source.name)
                    if cached_time and last_modified > cached_time:
                        self._source_modification_cache[source.name] = last_modified
                        return True
            
            # Check for real-time updates if source supports it
            if hasattr(source, 'has_realtime_updates'):
                return await source.has_realtime_updates()
            
            # Default: check based on time-based refresh policy
            last_check = self._source_check_cache.get(source.name)
            if last_check:
                time_since_check = datetime.now(timezone.utc) - last_check
                refresh_interval = self.update_policies.get('refresh_interval_minutes', 30)
                if time_since_check.total_seconds() < refresh_interval * 60:
                    return False
            
            # Update check time
            self._source_check_cache[source.name] = datetime.now(timezone.utc)
            
            # Conservative approach: assume no updates unless explicitly detected
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking source updates: {e}")
            return False
    
    async def _refresh_chunk(self, chunk: ContextChunk) -> Optional[ContextChunk]:
        """Refresh a single chunk."""
        try:
            # Get the source for this chunk
            source = chunk.source
            
            # Try to refresh the chunk content
            if hasattr(source, 'refresh_chunk'):
                refreshed_chunk = await source.refresh_chunk(chunk)
                if refreshed_chunk and refreshed_chunk.content != chunk.content:
                    self.logger.info(f"Refreshed chunk {chunk.id} from source {source.name}")
                    return refreshed_chunk
            
            # Fallback: re-query the source with the original query
            if hasattr(source, 'get_chunks'):
                # Extract original query from chunk metadata or use a default
                original_query = chunk.metadata.get('original_query', '')
                if original_query:
                    new_chunks = await source.get_chunks(original_query, max_chunks=1)
                    if new_chunks:
                        new_chunk = new_chunks[0]
                        if new_chunk.content != chunk.content:
                            self.logger.info(f"Refreshed chunk {chunk.id} via re-query")
                            return new_chunk
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error refreshing chunk {chunk.id}: {e}")
            return None
    
    async def _get_context_by_id(self, context_id: str) -> Optional[Context]:
        """Get context by ID from cache or storage."""
        try:
            # First check cache
            if context_id in self._context_cache:
                return self._context_cache[context_id]
            
            # Try to retrieve from storage backends
            if hasattr(self, 'storage_backends'):
                for backend_name, backend in self.storage_backends.items():
                    try:
                        if hasattr(backend, 'retrieve'):
                            context = await backend.retrieve(context_id)
                            if context:
                                # Cache the retrieved context
                                self._context_cache[context_id] = context
                                return context
                    except Exception as e:
                        self.logger.warning(f"Failed to retrieve context from {backend_name}: {e}")
                        continue
            
            # Context not found
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get context by ID {context_id}: {e}")
            return None
    
    async def _apply_context_update(self, context: Context, update_data: Dict[str, Any]) -> Context:
        """Apply update data to a context."""
        try:
            # Create updated context
            updated_context = context.copy()
            
            # Update metadata
            updated_context.metadata['last_updated'] = datetime.now(timezone.utc).isoformat()
            updated_context.metadata['update_source'] = update_data.get('source', 'unknown')
            updated_context.metadata['update_type'] = update_data.get('type', 'content')
            
            # Update chunks if new content is provided
            if 'content' in update_data:
                # Find and update relevant chunks
                for i, chunk in enumerate(updated_context.chunks):
                    if chunk.source.name == update_data.get('source_name'):
                        # Update chunk content
                        updated_chunk = chunk.copy()
                        updated_chunk.content = update_data['content']
                        updated_chunk.metadata['updated_at'] = datetime.now(timezone.utc).isoformat()
                        updated_context.chunks[i] = updated_chunk
                        break
            
            return updated_context
            
        except Exception as e:
            self.logger.error(f"Failed to apply context update: {e}")
            return context
    
    async def _store_updated_context(self, context: Context) -> None:
        """Store updated context."""
        try:
            # Update cache
            self._context_cache[context.id] = context
            
            # Store in persistent storage backends
            if hasattr(self, 'storage_backends'):
                for backend_name, backend in self.storage_backends.items():
                    try:
                        if hasattr(backend, 'store'):
                            await backend.store(context)
                            self.logger.info(f"Stored updated context {context.id} in {backend_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to store context in {backend_name}: {e}")
                        continue
            
            self.logger.info(f"Successfully stored updated context {context.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store updated context: {e}")
    
    async def get_update_stats(self) -> Dict[str, Any]:
        """
        Get update engine statistics.
        
        Returns:
            Dictionary with update statistics
        """
        try:
            return {
                'queue_size': self.update_queue.qsize(),
                'active_subscriptions': len(self.subscriptions),
                'background_tasks': len(self.background_tasks),
                'update_policies': self.update_policies,
            }
            
        except Exception as e:
            self.logger.error(f"Error getting update stats: {e}")
            return {}
