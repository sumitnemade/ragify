"""
Cache Manager for intelligent context caching.
"""

import asyncio
import json
import pickle
import gzip
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta
from collections import OrderedDict
import structlog

# Cache backends
import redis.asyncio as redis
import aiomcache
from cachetools import TTLCache, LRUCache

from ..models import Context
from ..exceptions import CacheError


class CacheManager:
    """
    Intelligent cache manager for context storage and retrieval.
    
    Provides caching with TTL, compression, and intelligent eviction
    policies for optimal performance.
    """
    
    def __init__(self, cache_url: str):
        """
        Initialize the cache manager.
        
        Args:
            cache_url: Cache connection URL
        """
        self.cache_url = cache_url
        self.logger = structlog.get_logger(__name__)
        
        # Cache client (would be initialized based on URL)
        self.cache_client = None
        
        # Cache configuration
        self.config = {
            'default_ttl': 3600,  # 1 hour
            'max_size': 1000,  # Maximum number of items
            'compression_enabled': True,
            'serialization_format': 'pickle',  # 'json' or 'pickle'
            'compression_threshold': 1024,  # Only compress data larger than this
            'retry_attempts': 3,  # Number of retry attempts for failed operations
            'retry_delay': 0.1,  # Delay between retries in seconds
        }
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
        }
    
    async def initialize(self) -> None:
        """Initialize the cache manager (alias for connect)."""
        await self.connect()
    
    async def connect(self) -> None:
        """Connect to the cache backend."""
        try:
            # Initialize cache client based on URL
            if self.cache_url.startswith('redis://'):
                await self._init_redis_client()
            elif self.cache_url.startswith('memcached://'):
                await self._init_memcached_client()
            else:
                await self._init_memory_client()
            
            self.logger.info(f"Connected to cache: {self.cache_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to cache: {e}")
            raise CacheError("connection", str(e))
    
    async def get(self, key: str) -> Optional[Context]:
        """
        Get a context from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Context or None if not found
        """
        try:
            # Ensure cache client is initialized
            if self.cache_client is None:
                await self.connect()
            
            # Get raw data from cache
            raw_data = await self._get_raw(key)
            if raw_data is None:
                self.stats['misses'] += 1
                return None
            
            # Decompress if needed
            if self.config['compression_enabled']:
                try:
                    raw_data = await self._decompress(raw_data)
                except Exception as e:
                    self.logger.warning(f"Failed to decompress data, using as-is: {e}")
            
            # Deserialize data
            context = await self._deserialize(raw_data)
            
            self.stats['hits'] += 1
            self.logger.debug(f"Cache hit for key: {key}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get from cache: {e}")
            self.stats['misses'] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Context,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a context in cache.
        
        Args:
            key: Cache key
            value: Context to store
            ttl: Time to live in seconds
        """
        try:
            # Ensure cache client is initialized
            if self.cache_client is None:
                await self.connect()
            
            # Serialize context
            serialized_data = await self._serialize(value)
            
            # Compress if enabled and data is large enough
            if self.config['compression_enabled'] and len(serialized_data) > self.config['compression_threshold']:
                serialized_data = await self._compress(serialized_data)
            
            # Store in cache
            await self._set_raw(key, serialized_data, ttl or self.config['default_ttl'])
            
            self.stats['sets'] += 1
            self.logger.debug(f"Cached context with key: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to set in cache: {e}")
            raise CacheError("set", str(e))
    
    async def delete(self, key: str) -> None:
        """
        Delete a context from cache.
        
        Args:
            key: Cache key to delete
        """
        try:
            await self._delete_raw(key)
            self.stats['deletes'] += 1
            self.logger.debug(f"Deleted cache key: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete from cache: {e}")
            raise CacheError("delete", str(e))
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return await self._exists_raw(key)
        except Exception as e:
            self.logger.error(f"Failed to check cache existence: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Remaining TTL in seconds or None if key doesn't exist
        """
        try:
            return await self._get_ttl_raw(key)
        except Exception as e:
            self.logger.error(f"Failed to get TTL: {e}")
            return None
    
    async def clear(self) -> None:
        """Clear all cached data."""
        try:
            await self._clear_raw()
            self.logger.info("Cleared all cached data")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise CacheError("clear", str(e))
    
    async def get_many(self, keys: list[str]) -> Dict[str, Optional[Context]]:
        """
        Get multiple contexts from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to contexts (None if not found)
        """
        try:
            results = {}
            
            if isinstance(self.cache_client, redis.Redis):
                # Redis - use mget for efficiency
                encoded_keys = [key.encode('utf-8') for key in keys]
                raw_data_list = await self.cache_client.mget(encoded_keys)
                
                for key, raw_data in zip(keys, raw_data_list):
                    if raw_data is None:
                        results[key] = None
                        self.stats['misses'] += 1
                    else:
                        # Decompress and deserialize
                        if self.config['compression_enabled']:
                            try:
                                raw_data = await self._decompress(raw_data)
                            except Exception as e:
                                self.logger.warning(f"Failed to decompress data for {key}: {e}")
                        
                        try:
                            context = await self._deserialize(raw_data)
                            results[key] = context
                            self.stats['hits'] += 1
                        except Exception as e:
                            self.logger.error(f"Failed to deserialize data for {key}: {e}")
                            results[key] = None
                            self.stats['misses'] += 1
                            
            else:
                # Other backends - get individually
                for key in keys:
                    results[key] = await self.get(key)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get multiple items from cache: {e}")
            return {key: None for key in keys}
    
    async def set_many(self, items: Dict[str, Context], ttl: Optional[int] = None) -> None:
        """
        Set multiple contexts in cache.
        
        Args:
            items: Dictionary mapping keys to contexts
            ttl: Time to live in seconds
        """
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis - use pipeline for efficiency
                pipeline = self.cache_client.pipeline()
                
                for key, context in items.items():
                    # Serialize and compress
                    serialized_data = await self._serialize(context)
                    
                    if self.config['compression_enabled'] and len(serialized_data) > self.config['compression_threshold']:
                        serialized_data = await self._compress(serialized_data)
                    
                    pipeline.setex(
                        key.encode('utf-8'),
                        ttl or self.config['default_ttl'],
                        serialized_data
                    )
                
                await pipeline.execute()
                
            else:
                # Other backends - set individually
                for key, context in items.items():
                    await self.set(key, context, ttl)
            
            self.stats['sets'] += len(items)
            self.logger.debug(f"Cached {len(items)} contexts")
            
        except Exception as e:
            self.logger.error(f"Failed to set multiple items in cache: {e}")
            raise CacheError("set_many", str(e))
    
    async def delete_many(self, keys: list[str]) -> None:
        """
        Delete multiple contexts from cache.
        
        Args:
            keys: List of cache keys to delete
        """
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis - use pipeline for efficiency
                pipeline = self.cache_client.pipeline()
                
                for key in keys:
                    pipeline.delete(key.encode('utf-8'))
                
                await pipeline.execute()
                
            else:
                # Other backends - delete individually
                for key in keys:
                    await self.delete(key)
            
            self.stats['deletes'] += len(keys)
            self.logger.debug(f"Deleted {len(keys)} cache keys")
            
        except Exception as e:
            self.logger.error(f"Failed to delete multiple items from cache: {e}")
            raise CacheError("delete_many", str(e))
    
    async def get_keys(self, pattern: str = "*") -> list[str]:
        """
        Get cache keys matching a pattern.
        
        Args:
            pattern: Key pattern (Redis-style patterns supported)
            
        Returns:
            List of matching keys
        """
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis - use scan for efficiency
                keys = []
                async for key in self.cache_client.scan_iter(match=pattern):
                    keys.append(key.decode('utf-8'))
                return keys
                
            elif hasattr(self.cache_client, '__iter__'):
                # In-memory cache
                async with self._cache_lock:
                    return [key for key in self.cache_client.keys() if self._matches_pattern(key, pattern)]
                    
            else:
                self.logger.warning("Key pattern matching not supported for this cache backend")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to get cache keys: {e}")
            return []
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple glob matching)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_stats = await self._get_cache_stats()
            
            return {
                **self.stats,
                'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0.0,
                'cache_stats': cache_stats,
                'config': self.config,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return self.stats
    
    async def _init_redis_client(self) -> None:
        """Initialize Redis client."""
        try:
            # Parse Redis URL
            if self.cache_url.startswith('redis://'):
                url = self.cache_url
            else:
                url = f"redis://{self.cache_url}"
            
            # Create Redis client
            self.cache_client = redis.from_url(url, decode_responses=False)
            
            # Test connection
            await self.cache_client.ping()
            self.logger.info("Redis client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            raise CacheError("redis_init", str(e))
    
    async def _init_memcached_client(self) -> None:
        """Initialize Memcached client."""
        try:
            # Parse Memcached URL (format: memcached://host:port)
            url_parts = self.cache_url.replace('memcached://', '').split(':')
            host = url_parts[0]
            port = int(url_parts[1]) if len(url_parts) > 1 else 11211
            
            # Create Memcached client
            self.cache_client = aiomcache.Client(host, port)
            
            # Test connection
            await self.cache_client.set(b'test', b'test', exptime=1)
            self.logger.info("Memcached client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Memcached client: {e}")
            raise CacheError("memcached_init", str(e))
    
    async def _init_memory_client(self) -> None:
        """Initialize in-memory cache client."""
        try:
            # Create in-memory cache with TTL and LRU eviction
            max_size = self.config.get('max_size', 1000)
            default_ttl = self.config.get('default_ttl', 3600)
            
            self.cache_client = TTLCache(
                maxsize=max_size,
                ttl=default_ttl
            )
            
            # Add thread safety
            self._cache_lock = asyncio.Lock()
            
            self.logger.info("In-memory cache client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize in-memory cache: {e}")
            raise CacheError("memory_init", str(e))
    
    async def _get_raw(self, key: str) -> Optional[bytes]:
        """Get raw data from cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                data = await self.cache_client.get(key.encode('utf-8'))
                return data if data else None
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached
                data = await self.cache_client.get(key.encode('utf-8'))
                return data if data else None
                
            elif hasattr(self.cache_client, '__getitem__'):
                # In-memory cache (TTLCache)
                async with self._cache_lock:
                    try:
                        return self.cache_client[key]
                    except KeyError:
                        return None
                    
            else:
                self.logger.warning(f"Unknown cache client type: {type(self.cache_client)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get raw data from cache: {e}")
            return None
    
    async def _set_raw(self, key: str, value: bytes, ttl: int) -> None:
        """Set raw data in cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                await self.cache_client.setex(
                    key.encode('utf-8'),
                    ttl,
                    value
                )
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached
                await self.cache_client.set(
                    key.encode('utf-8'),
                    value,
                    exptime=ttl
                )
                
            elif hasattr(self.cache_client, '__setitem__'):
                # In-memory cache
                async with self._cache_lock:
                    self.cache_client[key] = value
                    
            else:
                self.logger.error("Unknown cache client type")
                
        except Exception as e:
            self.logger.error(f"Failed to set raw data in cache: {e}")
            raise CacheError("set_raw", str(e))
    
    async def _delete_raw(self, key: str) -> None:
        """Delete raw data from cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                await self.cache_client.delete(key.encode('utf-8'))
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached
                await self.cache_client.delete(key.encode('utf-8'))
                
            elif hasattr(self.cache_client, '__delitem__'):
                # In-memory cache
                async with self._cache_lock:
                    if key in self.cache_client:
                        del self.cache_client[key]
                    
            else:
                self.logger.error("Unknown cache client type")
                
        except Exception as e:
            self.logger.error(f"Failed to delete raw data from cache: {e}")
            raise CacheError("delete_raw", str(e))
    
    async def _exists_raw(self, key: str) -> bool:
        """Check if key exists in cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                return await self.cache_client.exists(key.encode('utf-8')) > 0
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached - get and check if None
                data = await self.cache_client.get(key.encode('utf-8'))
                return data is not None
                
            elif hasattr(self.cache_client, '__contains__'):
                # In-memory cache
                async with self._cache_lock:
                    return key in self.cache_client
                    
            else:
                self.logger.error("Unknown cache client type")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to check cache existence: {e}")
            return False
    
    async def _get_ttl_raw(self, key: str) -> Optional[int]:
        """Get TTL from cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                ttl = await self.cache_client.ttl(key.encode('utf-8'))
                return ttl if ttl > 0 else None
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached doesn't support TTL queries
                return None
                
            elif hasattr(self.cache_client, 'get'):
                # In-memory cache - check if key exists
                async with self._cache_lock:
                    if key in self.cache_client:
                        # For TTLCache, we can't easily get remaining TTL
                        return None
                    return None
                    
            else:
                self.logger.error("Unknown cache client type")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get TTL from cache: {e}")
            return None
    
    async def _clear_raw(self) -> None:
        """Clear all data from cache backend."""
        try:
            if isinstance(self.cache_client, redis.Redis):
                # Redis
                await self.cache_client.flushdb()
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached - flush all
                await self.cache_client.flush_all()
                
            elif hasattr(self.cache_client, 'clear'):
                # In-memory cache
                async with self._cache_lock:
                    self.cache_client.clear()
                    
            else:
                self.logger.error("Unknown cache client type")
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise CacheError("clear_raw", str(e))
    
    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics from cache backend."""
        try:
            stats = {}
            
            if isinstance(self.cache_client, redis.Redis):
                # Redis info
                info = await self.cache_client.info()
                stats.update({
                    'redis_version': info.get('redis_version'),
                    'connected_clients': info.get('connected_clients'),
                    'used_memory_human': info.get('used_memory_human'),
                    'keyspace_hits': info.get('keyspace_hits'),
                    'keyspace_misses': info.get('keyspace_misses'),
                })
                
            elif isinstance(self.cache_client, aiomcache.Client):
                # Memcached stats
                stats_data = await self.cache_client.stats()
                if stats_data:
                    stats.update({
                        'memcached_version': stats_data.get(b'version', b'').decode('utf-8', errors='ignore'),
                        'total_connections': stats_data.get(b'total_connections', 0),
                        'curr_connections': stats_data.get(b'curr_connections', 0),
                        'get_hits': stats_data.get(b'get_hits', 0),
                        'get_misses': stats_data.get(b'get_misses', 0),
                    })
                
            elif hasattr(self.cache_client, 'currsize'):
                # In-memory cache stats
                stats.update({
                    'cache_type': 'in_memory',
                    'current_size': self.cache_client.currsize,
                    'max_size': getattr(self.cache_client, 'maxsize', 'unknown'),
                })
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def _serialize(self, context: Context) -> bytes:
        """Serialize context to bytes."""
        try:
            if self.config['serialization_format'] == 'json':
                return json.dumps(context.model_dump()).encode('utf-8')
            else:  # pickle
                return pickle.dumps(context)
        except Exception as e:
            self.logger.error(f"Failed to serialize context: {e}")
            raise CacheError("serialization", str(e))
    
    async def _deserialize(self, data: bytes) -> Context:
        """Deserialize bytes to context."""
        try:
            if self.config['serialization_format'] == 'json':
                context_dict = json.loads(data.decode('utf-8'))
                return Context(**context_dict)
            else:  # pickle
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to deserialize context: {e}")
            raise CacheError("deserialization", str(e))
    
    async def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        try:
            import gzip
            return gzip.compress(data)
        except ImportError:
            self.logger.warning("gzip compression not available, storing uncompressed data")
            return data  # Return uncompressed data on error
        except Exception as e:
            self.logger.warning(f"Compression failed, storing uncompressed data: {e}")
            return data  # Return uncompressed data on error
    
    async def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        try:
            import gzip
            return gzip.decompress(data)
        except ImportError:
            self.logger.warning("gzip decompression not available, returning data as-is")
            return data  # Return data as-is on error
        except Exception as e:
            self.logger.warning(f"Decompression failed, returning data as-is: {e}")
            return data  # Return data as-is on error
    
    async def close(self) -> None:
        """Close the cache connection."""
        try:
            if self.cache_client:
                if isinstance(self.cache_client, redis.Redis):
                    # Close Redis connection
                    await self.cache_client.close()
                elif isinstance(self.cache_client, aiomcache.Client):
                    # Close Memcached connection
                    self.cache_client.close()
                elif hasattr(self.cache_client, 'clear'):
                    # Clear in-memory cache
                    async with self._cache_lock:
                        self.cache_client.clear()
            
            self.logger.info("Cache connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing cache connection: {e}")
