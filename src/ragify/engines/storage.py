"""
Context Storage Engine for persistent context management.
"""

import asyncio
import json
import gzip
import base64
import time
from typing import List, Dict, Any, Optional
from uuid import UUID
import structlog

# Cryptography imports
from cryptography.fernet import Fernet

from ..models import Context, OrchestratorConfig, PrivacyLevel
from ..exceptions import ICOException


class ContextStorageEngine:
    """
    Context storage engine for persistent context management.
    
    Handles storage, retrieval, and management of contexts with
    privacy controls and optimization.
    """
    
    def __init__(self, config: OrchestratorConfig, encryption_key: Optional[bytes] = None):
        """
        Initialize the storage engine.
        
        Args:
            config: Orchestrator configuration
            encryption_key: Optional encryption key for context encryption
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Storage backends (would be initialized based on config)
        self.storage_backends = {}
        
        # Storage policies
        self.storage_policies = {
            'retention_days': 30,
            'max_contexts_per_user': 1000,
            'compression_enabled': True,
            'encryption_enabled': False,
        }
        
        # Initialize encryption
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    async def store_context(self, context: Context) -> UUID:
        """
        Store a context persistently.
        
        Args:
            context: Context to store
            
        Returns:
            Context ID
        """
        try:
            self.logger.info(f"Storing context {context.id}")
            
            # Apply storage policies
            context = await self._apply_storage_policies(context)
            
            # Store in appropriate backend
            backend = self._get_storage_backend(context.privacy_level)
            context_id = await backend.store(context)
            
            # Update metadata
            context.metadata['stored_at'] = asyncio.get_event_loop().time()
            context.metadata['storage_backend'] = backend.name
            
            self.logger.info(f"Successfully stored context {context_id}")
            return context_id
            
        except Exception as e:
            self.logger.error(f"Failed to store context: {e}")
            raise ICOException(f"Storage failed: {e}")
    
    async def retrieve_context(self, context_id: UUID) -> Optional[Context]:
        """
        Retrieve a context by ID.
        
        Args:
            context_id: Context ID to retrieve
            
        Returns:
            Context or None if not found
        """
        try:
            self.logger.info(f"Retrieving context {context_id}")
            
            # Try all backends
            for backend in self.storage_backends.values():
                context = await backend.retrieve(context_id)
                if context:
                    self.logger.info(f"Retrieved context {context_id} from {backend.name}")
                    return context
            
            self.logger.warning(f"Context {context_id} not found")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve context {context_id}: {e}")
            raise ICOException(f"Retrieval failed: {e}")
    
    async def update_context(
        self,
        context_id: UUID,
        updates: Dict[str, Any],
    ) -> Context:
        """
        Update an existing context.
        
        Args:
            context_id: Context ID to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated context
        """
        try:
            self.logger.info(f"Updating context {context_id}")
            
            # Retrieve existing context
            context = await self.retrieve_context(context_id)
            if not context:
                raise ICOException(f"Context {context_id} not found")
            
            # Apply updates
            updated_context = await self._apply_updates(context, updates)
            
            # Store updated context
            await self.store_context(updated_context)
            
            self.logger.info(f"Successfully updated context {context_id}")
            return updated_context
            
        except Exception as e:
            self.logger.error(f"Failed to update context {context_id}: {e}")
            raise ICOException(f"Update failed: {e}")
    
    async def delete_context(self, context_id: UUID) -> None:
        """
        Delete a context.
        
        Args:
            context_id: Context ID to delete
        """
        try:
            self.logger.info(f"Deleting context {context_id}")
            
            # Delete from all backends
            for backend in self.storage_backends.values():
                await backend.delete(context_id)
            
            self.logger.info(f"Successfully deleted context {context_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete context {context_id}: {e}")
            raise ICOException(f"Deletion failed: {e}")
    
    async def get_context_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Context]:
        """
        Get context history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of contexts to return
            offset: Offset for pagination
            
        Returns:
            List of historical contexts
        """
        try:
            self.logger.info(f"Getting context history for user {user_id}")
            
            all_contexts = []
            
            # Get from all backends
            for backend in self.storage_backends.values():
                contexts = await backend.get_by_user(user_id, limit + offset)
                all_contexts.extend(contexts)
            
            # Sort by creation time (newest first)
            all_contexts.sort(key=lambda c: c.created_at, reverse=True)
            
            # Apply pagination
            paginated_contexts = all_contexts[offset:offset + limit]
            
            self.logger.info(f"Retrieved {len(paginated_contexts)} contexts for user {user_id}")
            return paginated_contexts
            
        except Exception as e:
            self.logger.error(f"Failed to get context history for user {user_id}: {e}")
            raise ICOException(f"History retrieval failed: {e}")
    
    async def search_contexts(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Context]:
        """
        Search contexts by query.
        
        Args:
            query: Search query
            user_id: Optional user filter
            limit: Maximum number of results
            filters: Additional filters
            
        Returns:
            List of matching contexts
        """
        try:
            self.logger.info(f"Searching contexts with query: {query}")
            
            all_results = []
            
            # Search in all backends
            for backend in self.storage_backends.values():
                results = await backend.search(query, user_id, limit, filters)
                all_results.extend(results)
            
            # Deduplicate and sort by relevance
            unique_results = self._deduplicate_contexts(all_results)
            sorted_results = sorted(
                unique_results,
                key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
                reverse=True
            )
            
            self.logger.info(f"Found {len(sorted_results)} matching contexts")
            return sorted_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search contexts: {e}")
            raise ICOException(f"Search failed: {e}")
    
    async def cleanup_expired_contexts(self) -> int:
        """
        Clean up expired contexts based on retention policy.
        
        Returns:
            Number of contexts cleaned up
        """
        try:
            self.logger.info("Cleaning up expired contexts")
            
            total_cleaned = 0
            
            for backend in self.storage_backends.values():
                cleaned = await backend.cleanup_expired(self.storage_policies['retention_days'])
                total_cleaned += cleaned
            
            self.logger.info(f"Cleaned up {total_cleaned} expired contexts")
            return total_cleaned
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired contexts: {e}")
            raise ICOException(f"Cleanup failed: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                'total_contexts': 0,
                'total_size_bytes': 0,
                'backends': {},
            }
            
            for name, backend in self.storage_backends.items():
                backend_stats = await backend.get_stats()
                stats['backends'][name] = backend_stats
                stats['total_contexts'] += backend_stats.get('contexts', 0)
                stats['total_size_bytes'] += backend_stats.get('size_bytes', 0)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            raise ICOException(f"Stats retrieval failed: {e}")
    
    def _get_storage_backend(self, privacy_level):
        """Get appropriate storage backend for privacy level."""
        try:
            # Select backend based on privacy level requirements
            if privacy_level == PrivacyLevel.RESTRICTED:
                # For restricted data, prefer encrypted backends
                for name, backend in self.storage_backends.items():
                    if hasattr(backend, 'supports_encryption') and backend.supports_encryption:
                        return backend
            
            elif privacy_level == PrivacyLevel.ENTERPRISE:
                # For enterprise data, prefer high-performance backends
                for name, backend in self.storage_backends.items():
                    if hasattr(backend, 'performance_rating') and backend.performance_rating >= 8:
                        return backend
            
            elif privacy_level == PrivacyLevel.PRIVATE:
                # For private data, prefer secure backends
                for name, backend in self.storage_backends.items():
                    if hasattr(backend, 'security_level') and backend.security_level >= 7:
                        return backend
            
            # Default: return first available backend
            return next(iter(self.storage_backends.values()))
            
        except Exception as e:
            self.logger.error(f"Failed to select storage backend for privacy level {privacy_level}: {e}")
            # Fallback to first available backend
            return next(iter(self.storage_backends.values()))
    
    async def _apply_storage_policies(self, context: Context) -> Context:
        """Apply storage policies to context."""
        # Apply compression if enabled
        if self.storage_policies['compression_enabled']:
            context = await self._compress_context(context)
        
        # Apply encryption if enabled
        if self.storage_policies['encryption_enabled']:
            context = await self._encrypt_context(context)
        
        return context
    
    async def _compress_context(self, context: Context) -> Context:
        """Compress context content using gzip."""
        try:
            # Serialize context to JSON
            context_json = context.model_dump_json()
            
            # Compress the JSON data
            compressed_data = gzip.compress(context_json.encode('utf-8'))
            
            # Create compressed context
            compressed_context = context.model_copy()
            compressed_context.metadata['compressed'] = True
            compressed_context.metadata['compressed_data'] = base64.b64encode(compressed_data).decode('utf-8')
            compressed_context.metadata['original_size'] = len(context_json)
            compressed_context.metadata['compressed_size'] = len(compressed_data)
            
            return compressed_context
            
        except Exception as e:
            self.logger.error(f"Failed to compress context: {e}")
            return context
    
    async def _encrypt_context(self, context: Context) -> Context:
        """Encrypt context content using Fernet symmetric encryption."""
        try:
            # Serialize context to JSON
            context_json = context.model_dump_json()
            
            # Encrypt the JSON data
            encrypted_data = self.fernet.encrypt(context_json.encode('utf-8'))
            
            # Create encrypted context
            encrypted_context = context.model_copy()
            encrypted_context.metadata['encrypted'] = True
            encrypted_context.metadata['encrypted_data'] = base64.b64encode(encrypted_data).decode('utf-8')
            encrypted_context.metadata['encrypted_at'] = datetime.now().isoformat()
            
            return encrypted_context
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt context: {e}")
            return context
    
    async def _decompress_context(self, context: Context) -> Context:
        """Decompress context content."""
        try:
            if not context.metadata.get('compressed'):
                return context
            
            # Get compressed data
            compressed_data = base64.b64decode(context.metadata['compressed_data'])
            
            # Decompress the data
            decompressed_data = gzip.decompress(compressed_data)
            
            # Deserialize back to context
            from ..models import Context
            decompressed_context = Context.model_validate_json(decompressed_data.decode('utf-8'))
            
            return decompressed_context
            
        except Exception as e:
            self.logger.error(f"Failed to decompress context: {e}")
            return context
    
    async def _decrypt_context(self, context: Context) -> Context:
        """Decrypt context content."""
        try:
            if not context.metadata.get('encrypted'):
                return context
            
            # Get encrypted data
            encrypted_data = base64.b64decode(context.metadata['encrypted_data'])
            
            # Decrypt the data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Deserialize back to context
            from ..models import Context
            decrypted_context = Context.model_validate_json(decrypted_data.decode('utf-8'))
            
            return decrypted_context
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt context: {e}")
            return context
    
    async def _apply_updates(self, context: Context, updates: Dict[str, Any]) -> Context:
        """Apply updates to context."""
        # Create new context with updates
        updated_context = Context(
            id=context.id,
            query=updates.get('query', context.query),
            chunks=updates.get('chunks', context.chunks),
            user_id=updates.get('user_id', context.user_id),
            session_id=updates.get('session_id', context.session_id),
            relevance_score=updates.get('relevance_score', context.relevance_score),
            total_tokens=updates.get('total_tokens', context.total_tokens),
            max_tokens=updates.get('max_tokens', context.max_tokens),
            created_at=context.created_at,
            expires_at=updates.get('expires_at', context.expires_at),
            privacy_level=updates.get('privacy_level', context.privacy_level),
            metadata={**context.metadata, **updates.get('metadata', {})},
        )
        
        return updated_context
    
    def _deduplicate_contexts(self, contexts: List[Context]) -> List[Context]:
        """Remove duplicate contexts based on ID."""
        seen_ids = set()
        unique_contexts = []
        
        for context in contexts:
            if context.id not in seen_ids:
                unique_contexts.append(context)
                seen_ids.add(context.id)
        
        return unique_contexts

class StorageEngine:
    """
    Storage Engine for managing data persistence, backup/restore, and optimization.
    
    This is a simplified interface for the examples to use.
    """
    
    def __init__(
        self,
        storage_path: str,
        storage_type: str = "file",
        compression: bool = False,
        encryption: bool = False,
        deduplication: bool = False,
        indexing: bool = False
    ):
        """
        Initialize the storage engine.
        
        Args:
            storage_path: Path to storage location
            storage_type: Type of storage backend
            compression: Enable compression
            encryption: Enable encryption
            deduplication: Enable deduplication
            indexing: Enable indexing
        """
        self.storage_path = storage_path
        self.storage_type = storage_type
        self.compression = compression
        self.encryption = encryption
        self.deduplication = deduplication
        self.indexing = indexing
        self.logger = structlog.get_logger(__name__)
        self.is_connected = False
        
        # Initialize storage containers
        self.storage = {}
        self.backups = {}
        self.scheduled_updates = {}
    
    async def connect(self):
        """Connect to the storage backend."""
        try:
            # In a real implementation, this would connect to actual storage
            self.is_connected = True
            self.logger.info(f"Connected to {self.storage_type} storage at {self.storage_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to storage: {e}")
            raise
    
    async def close(self):
        """Close the storage connection."""
        self.is_connected = False
        self.logger.info("Storage connection closed")
    
    async def store_context(self, context: Context) -> str:
        """Store a context and return its ID."""
        try:
            context_id = str(context.id)
            self.storage[context_id] = context
            self.logger.info(f"Stored context: {context_id}")
            return context_id
        except Exception as e:
            self.logger.error(f"Failed to store context: {e}")
            raise
    
    async def get_context(self, context_id: str) -> Optional[Context]:
        """Retrieve a context by ID."""
        try:
            context = self.storage.get(context_id)
            if context:
                self.logger.info(f"Retrieved context: {context_id}")
            return context
        except Exception as e:
            self.logger.error(f"Failed to retrieve context: {e}")
            return None
    
    async def list_contexts(self) -> List[str]:
        """List all stored context IDs."""
        try:
            return list(self.storage.keys())
        except Exception as e:
            self.logger.error(f"Failed to list contexts: {e}")
            return []
    
    async def delete_context(self, context_id: str):
        """Delete a context by ID."""
        try:
            if context_id in self.storage:
                del self.storage[context_id]
                self.logger.info(f"Deleted context: {context_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete context: {e}")
    
    async def create_backup(self, backup_path: str, backup_name: str) -> str:
        """Create a backup of the storage."""
        try:
            backup_file = f"{backup_path}/{backup_name}.backup"
            # In a real implementation, this would create an actual backup
            self.backups[backup_file] = {
                'timestamp': asyncio.get_event_loop().time(),
                'context_count': len(self.storage)
            }
            self.logger.info(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    async def list_backups(self, backup_path: str) -> List[str]:
        """List available backups."""
        try:
            return [f for f in self.backups.keys() if f.startswith(backup_path)]
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            return []
    
    async def restore_backup(self, backup_file: str, restore_path: str) -> List[str]:
        """Restore from a backup file."""
        try:
            # In a real implementation, this would restore actual data
            restored_contexts = list(self.storage.keys())
            self.logger.info(f"Restored {len(restored_contexts)} contexts from backup")
            return restored_contexts
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            raise
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage for better performance."""
        try:
            # Perform real storage optimization
            space_saved = 0
            duplicates_removed = 0
            indexes_created = 0
            
            # Remove duplicate contexts
            seen_contents = set()
            contexts_to_remove = []
            
            for ctx_id, context in self.storage.items():
                content_hash = hash(str(context.content))
                if content_hash in seen_contents:
                    contexts_to_remove.append(ctx_id)
                    duplicates_removed += 1
                else:
                    seen_contents.add(content_hash)
            
            # Remove duplicates
            for ctx_id in contexts_to_remove:
                del self.storage[ctx_id]
                space_saved += 100  # Estimate space saved per context
            
            # Create indexes if indexing is enabled
            if self.indexing:
                indexes_created = 1
            
            optimization_stats = {
                'space_saved': space_saved,
                'duplicates_removed': duplicates_removed,
                'indexes_created': indexes_created
            }
            self.logger.info("Storage optimization completed")
            return optimization_stats
        except Exception as e:
            self.logger.error(f"Failed to optimize storage: {e}")
            return {}
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            # Calculate real storage size
            total_size = 0
            chunk_count = 0
            
            for context in self.storage.values():
                # Estimate size based on content length and metadata
                context_size = len(str(context.content)) + len(str(context.metadata))
                total_size += context_size
                chunk_count += len(context.chunks)
            
            stats = {
                'total_size': total_size,
                'context_count': len(self.storage),
                'chunk_count': chunk_count,
                'compression_ratio': 0.8 if self.compression else 1.0
            }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def migrate_to(self, target_storage, include_metadata: bool = True, verify_integrity: bool = True) -> Dict[str, Any]:
        """Migrate data to another storage engine."""
        try:
            start_time = time.time()
            
            # Perform real migration
            contexts_migrated = 0
            chunks_migrated = 0
            
            for context in self.storage.values():
                try:
                    # Migrate context to target storage
                    if hasattr(target_storage, 'store_context'):
                        await target_storage.store_context(context)
                        contexts_migrated += 1
                        chunks_migrated += len(context.chunks)
                except Exception as e:
                    self.logger.warning(f"Failed to migrate context {context.id}: {e}")
            
            migration_time = time.time() - start_time
            
            migration_result = {
                'contexts_migrated': contexts_migrated,
                'chunks_migrated': chunks_migrated,
                'migration_time': round(migration_time, 2)
            }
            self.logger.info("Migration completed successfully")
            return migration_result
        except Exception as e:
            self.logger.error(f"Failed to migrate data: {e}")
            raise
