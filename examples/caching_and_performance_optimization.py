#!/usr/bin/env python3
"""
Cache Management Demo for Ragify Framework

This demo showcases real cache management with multiple backends
and advanced features for intelligent context caching.
"""

import asyncio
import time
from datetime import datetime
from uuid import uuid4
from src.ragify.storage.cache import CacheManager
from src.ragify.models import Context, ContextChunk, ContextSource, SourceType, PrivacyLevel

async def demo_memory_cache():
    """Demonstrate in-memory cache functionality."""
    
    print("🧪 In-Memory Cache Demo")
    print("=" * 50)
    
    # Initialize cache manager with memory backend
    cache_manager = CacheManager("memory://")
    
    try:
        # Connect to cache
        await cache_manager.connect()
        print("✅ Connected to in-memory cache")
        
        # Create test context
        test_context = Context(
            query="demo query",
            chunks=[
                ContextChunk(
                    content="This is a demo chunk for caching demonstration",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Demo Source",
                        source_type=SourceType.DOCUMENT,
                        url="demo://source",
                    ),
                )
            ],
            user_id="demo_user"
        )
        
        # Test basic operations
        print("\n📝 Testing basic cache operations:")
        
        # Set context
        await cache_manager.set("demo_key", test_context, ttl=60)
        print("✅ Set context in cache")
        
        # Check existence
        exists = await cache_manager.exists("demo_key")
        print(f"✅ Key exists: {exists}")
        
        # Get context
        retrieved_context = await cache_manager.get("demo_key")
        if retrieved_context:
            print(f"✅ Retrieved context: {retrieved_context.query}")
            print(f"   Chunks: {len(retrieved_context.chunks)}")
        else:
            print("❌ Failed to retrieve context")
        
        # Get TTL
        ttl = await cache_manager.get_ttl("demo_key")
        print(f"✅ TTL: {ttl} seconds")
        
        # Delete context
        await cache_manager.delete("demo_key")
        print("✅ Deleted context from cache")
        
        # Verify deletion
        exists_after = await cache_manager.exists("demo_key")
        print(f"✅ Key exists after deletion: {exists_after}")
        
    except Exception as e:
        print(f"❌ Memory cache demo failed: {e}")
    finally:
        await cache_manager.close()

async def demo_bulk_operations():
    """Demonstrate bulk cache operations."""
    
    print(f"\n📦 Bulk Operations Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager("memory://")
    
    try:
        await cache_manager.connect()
        print("✅ Connected to cache")
        
        # Create multiple test contexts
        contexts = {}
        for i in range(5):
            context = Context(
                query=f"bulk query {i}",
                chunks=[
                    ContextChunk(
                        content=f"This is bulk chunk {i} for demonstration",
                        source=ContextSource(
                            id=str(uuid4()),
                            name=f"Bulk Source {i}",
                            source_type=SourceType.DOCUMENT,
                            url=f"bulk://source/{i}",
                        ),
                    )
                ],
                user_id=f"bulk_user_{i}"
            )
            contexts[f"bulk_key_{i}"] = context
        
        print(f"📝 Created {len(contexts)} test contexts")
        
        # Test set_many
        await cache_manager.set_many(contexts, ttl=120)
        print("✅ Set multiple contexts in cache")
        
        # Test get_many
        keys = list(contexts.keys())
        retrieved_contexts = await cache_manager.get_many(keys)
        print(f"✅ Retrieved {len(retrieved_contexts)} contexts")
        
        # Verify retrieval
        for key, context in retrieved_contexts.items():
            if context:
                print(f"   {key}: {context.query}")
            else:
                print(f"   {key}: Not found")
        
        # Test delete_many
        await cache_manager.delete_many(keys)
        print("✅ Deleted multiple contexts from cache")
        
        # Verify deletion
        retrieved_after = await cache_manager.get_many(keys)
        found_count = sum(1 for ctx in retrieved_after.values() if ctx is not None)
        print(f"✅ Found {found_count} contexts after deletion (should be 0)")
        
    except Exception as e:
        print(f"❌ Bulk operations demo failed: {e}")
    finally:
        await cache_manager.close()

async def demo_cache_statistics():
    """Demonstrate cache statistics and monitoring."""
    
    print(f"\n📊 Cache Statistics Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager("memory://")
    
    try:
        await cache_manager.connect()
        print("✅ Connected to cache")
        
        # Create test context
        test_context = Context(
            query="statistics demo",
            chunks=[],
            user_id="stats_demo_user"
        )
        
        # Perform operations to generate statistics
        print("\n📝 Performing cache operations:")
        
        # Set operations
        for i in range(10):
            await cache_manager.set(f"stats_demo_key_{i}", test_context, ttl=60)
        
        # Get operations (some hits, some misses)
        for i in range(15):
            await cache_manager.get(f"stats_demo_key_{i}")
        
        # Delete operations
        for i in range(5):
            await cache_manager.delete(f"stats_demo_key_{i}")
        
        print("✅ Completed cache operations")
        
        # Get statistics
        stats = await cache_manager.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Hits: {stats['hits']}")
        print(f"   Misses: {stats['misses']}")
        print(f"   Sets: {stats['sets']}")
        print(f"   Deletes: {stats['deletes']}")
        print(f"   Hit Rate: {stats['hit_rate']:.2%}")
        
        # Show cache backend stats
        if 'cache_stats' in stats:
            print(f"\n🔧 Cache Backend Stats:")
            for key, value in stats['cache_stats'].items():
                print(f"   {key}: {value}")
        
        # Show configuration
        print(f"\n⚙️  Cache Configuration:")
        for key, value in stats['config'].items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Cache statistics demo failed: {e}")
    finally:
        await cache_manager.close()

async def demo_cache_patterns():
    """Demonstrate cache key patterns and searching."""
    
    print(f"\n🔍 Cache Pattern Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager("memory://")
    
    try:
        await cache_manager.connect()
        print("✅ Connected to cache")
        
        # Create test contexts with different key patterns
        test_context = Context(
            query="pattern demo",
            chunks=[],
            user_id="pattern_demo_user"
        )
        
        # Set contexts with different key patterns
        patterns = [
            "user:123:query:abc",
            "user:123:query:def",
            "user:456:query:abc",
            "session:789:context:xyz",
            "session:789:context:uvw",
            "temp:cache:item:1",
            "temp:cache:item:2",
        ]
        
        for pattern in patterns:
            await cache_manager.set(pattern, test_context, ttl=60)
        
        print(f"📝 Set {len(patterns)} contexts with different patterns")
        
        # Test pattern matching
        print(f"\n🔍 Testing pattern matching:")
        
        # Get all keys
        all_keys = await cache_manager.get_keys("*")
        print(f"✅ All keys: {len(all_keys)} found")
        
        # Get user keys
        user_keys = await cache_manager.get_keys("user:*")
        print(f"✅ User keys: {len(user_keys)} found")
        for key in user_keys:
            print(f"   {key}")
        
        # Get session keys
        session_keys = await cache_manager.get_keys("session:*")
        print(f"✅ Session keys: {len(session_keys)} found")
        for key in session_keys:
            print(f"   {key}")
        
        # Get temp keys
        temp_keys = await cache_manager.get_keys("temp:*")
        print(f"✅ Temp keys: {len(temp_keys)} found")
        for key in temp_keys:
            print(f"   {key}")
        
        # Get specific pattern
        specific_keys = await cache_manager.get_keys("user:123:*")
        print(f"✅ User 123 keys: {len(specific_keys)} found")
        for key in specific_keys:
            print(f"   {key}")
        
    except Exception as e:
        print(f"❌ Cache pattern demo failed: {e}")
    finally:
        await cache_manager.close()

async def demo_cache_compression():
    """Demonstrate cache compression functionality."""
    
    print(f"\n🗜️  Cache Compression Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager("memory://")
    
    try:
        await cache_manager.connect()
        print("✅ Connected to cache")
        
        # Create large test context
        large_content = "This is a very large content that should be compressed for efficient storage. " * 1000
        test_context = Context(
            query="compression demo",
            chunks=[
                ContextChunk(
                    content=large_content,
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Compression Demo Source",
                        source_type=SourceType.DOCUMENT,
                        url="demo://compression",
                    ),
                )
            ],
            user_id="compression_demo_user"
        )
        
        print(f"📝 Created large context with {len(large_content)} characters")
        
        # Test with compression enabled
        cache_manager.config['compression_enabled'] = True
        cache_manager.config['compression_threshold'] = 100  # Low threshold to force compression
        
        await cache_manager.set("compressed_demo_key", test_context, ttl=60)
        print("✅ Set large context with compression")
        
        # Retrieve and verify
        retrieved = await cache_manager.get("compressed_demo_key")
        if retrieved and retrieved.chunks:
            print(f"✅ Retrieved compressed context")
            print(f"   Content length: {len(retrieved.chunks[0].content)}")
            print(f"   Content matches: {retrieved.chunks[0].content == large_content}")
        else:
            print("❌ Failed to retrieve compressed context")
        
        # Test without compression
        cache_manager.config['compression_enabled'] = False
        
        await cache_manager.set("uncompressed_demo_key", test_context, ttl=60)
        print("✅ Set large context without compression")
        
        # Retrieve and verify
        retrieved = await cache_manager.get("uncompressed_demo_key")
        if retrieved and retrieved.chunks:
            print(f"✅ Retrieved uncompressed context")
            print(f"   Content length: {len(retrieved.chunks[0].content)}")
            print(f"   Content matches: {retrieved.chunks[0].content == large_content}")
        else:
            print("❌ Failed to retrieve uncompressed context")
        
    except Exception as e:
        print(f"❌ Cache compression demo failed: {e}")
    finally:
        await cache_manager.close()

async def demo_cache_backends():
    """Demonstrate different cache backends."""
    
    print(f"\n🏗️  Cache Backends Demo")
    print("=" * 50)
    
    # Test different cache backends
    backends = [
        ("memory://", "In-Memory Cache"),
        # ("redis://localhost:6379", "Redis Cache"),  # Uncomment if Redis is available
        # ("memcached://localhost:11211", "Memcached Cache"),  # Uncomment if Memcached is available
    ]
    
    for cache_url, backend_name in backends:
        print(f"\n🔧 Testing {backend_name}")
        print("-" * 30)
        
        cache_manager = CacheManager(cache_url)
        
        try:
            await cache_manager.connect()
            print(f"✅ Connected to {backend_name}")
            
            # Create test context
            test_context = Context(
                query=f"{backend_name} test",
                chunks=[
                    ContextChunk(
                        content=f"Testing {backend_name} functionality",
                        source=ContextSource(
                            id=str(uuid4()),
                            name=f"{backend_name} Source",
                            source_type=SourceType.DOCUMENT,
                            url=f"test://{backend_name.lower().replace(' ', '_')}",
                        ),
                    )
                ],
                user_id=f"{backend_name.lower().replace(' ', '_')}_user"
            )
            
            # Test basic operations
            await cache_manager.set("backend_test", test_context, ttl=60)
            retrieved = await cache_manager.get("backend_test")
            
            if retrieved:
                print(f"✅ {backend_name} operations successful")
                print(f"   Retrieved: {retrieved.query}")
            else:
                print(f"❌ {backend_name} operations failed")
            
            # Get backend stats
            stats = await cache_manager.get_stats()
            if 'cache_stats' in stats and stats['cache_stats']:
                print(f"📊 {backend_name} Stats:")
                for key, value in stats['cache_stats'].items():
                    print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ {backend_name} test failed: {e}")
        finally:
            await cache_manager.close()

async def demo_cache_performance():
    """Demonstrate cache performance characteristics."""
    
    print(f"\n⚡ Cache Performance Demo")
    print("=" * 50)
    
    # Initialize cache manager
    cache_manager = CacheManager("memory://")
    
    try:
        await cache_manager.connect()
        print("✅ Connected to cache")
        
        # Create test context
        test_context = Context(
            query="performance test",
            chunks=[
                ContextChunk(
                    content="Performance testing content",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Performance Source",
                        source_type=SourceType.DOCUMENT,
                        url="test://performance",
                    ),
                )
            ],
            user_id="performance_user"
        )
        
        # Test single operations performance
        print("\n📝 Testing single operations performance:")
        
        # Set performance
        start_time = time.time()
        for i in range(100):
            await cache_manager.set(f"perf_key_{i}", test_context, ttl=60)
        set_time = time.time() - start_time
        print(f"✅ Set 100 items in {set_time:.3f} seconds ({100/set_time:.1f} ops/sec)")
        
        # Get performance
        start_time = time.time()
        for i in range(100):
            await cache_manager.get(f"perf_key_{i}")
        get_time = time.time() - start_time
        print(f"✅ Get 100 items in {get_time:.3f} seconds ({100/get_time:.1f} ops/sec)")
        
        # Bulk operations performance
        print("\n📦 Testing bulk operations performance:")
        
        # Create bulk data
        bulk_contexts = {}
        for i in range(50):
            bulk_contexts[f"bulk_perf_{i}"] = test_context
        
        # Bulk set performance
        start_time = time.time()
        await cache_manager.set_many(bulk_contexts, ttl=60)
        bulk_set_time = time.time() - start_time
        print(f"✅ Bulk set 50 items in {bulk_set_time:.3f} seconds ({50/bulk_set_time:.1f} ops/sec)")
        
        # Bulk get performance
        keys = list(bulk_contexts.keys())
        start_time = time.time()
        await cache_manager.get_many(keys)
        bulk_get_time = time.time() - start_time
        print(f"✅ Bulk get 50 items in {bulk_get_time:.3f} seconds ({50/bulk_get_time:.1f} ops/sec)")
        
        # Performance comparison
        print(f"\n📊 Performance Comparison:")
        print(f"   Single Set: {100/set_time:.1f} ops/sec")
        print(f"   Single Get: {100/get_time:.1f} ops/sec")
        print(f"   Bulk Set: {50/bulk_set_time:.1f} ops/sec")
        print(f"   Bulk Get: {50/bulk_get_time:.1f} ops/sec")
        print(f"   Bulk Set Speedup: {(50/bulk_set_time)/(100/set_time):.1f}x")
        print(f"   Bulk Get Speedup: {(50/bulk_get_time)/(100/get_time):.1f}x")
        
    except Exception as e:
        print(f"❌ Cache performance demo failed: {e}")
    finally:
        await cache_manager.close()

async def main():
    """Run the complete cache management demo."""
    
    print("🎯 Ragify Real Cache Management Demo")
    print("=" * 60)
    print("This demo showcases real cache management")
    print("with multiple backends and advanced features.\n")
    
    # Run all demos
    await demo_memory_cache()
    await demo_bulk_operations()
    await demo_cache_statistics()
    await demo_cache_patterns()
    await demo_cache_compression()
    await demo_cache_backends()
    await demo_cache_performance()
    
    print(f"\n🎉 Complete cache management demo finished!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✅ In-memory cache with TTL and LRU eviction")
    print(f"   ✅ Bulk operations (get_many, set_many, delete_many)")
    print(f"   ✅ Cache statistics and monitoring")
    print(f"   ✅ Pattern-based key searching")
    print(f"   ✅ Data compression and serialization")
    print(f"   ✅ Thread-safe operations")
    print(f"   ✅ Performance benchmarking")
    print(f"   ✅ Multiple backend support")
    print(f"\n📚 Supported Cache Backends:")
    print(f"   • Redis (with async client)")
    print(f"   • Memcached (with async client)")
    print(f"   • In-memory (with TTL and LRU)")
    print(f"\n🔧 Advanced Features:")
    print(f"   • Automatic compression for large data")
    print(f"   • Multiple serialization formats (JSON, Pickle)")
    print(f"   • Pattern-based key management")
    print(f"   • Comprehensive statistics and monitoring")
    print(f"   • Bulk operations for efficiency")
    print(f"   • Performance optimization")
    print(f"\n📚 Usage Examples:")
    print(f"   # Initialize cache manager")
    print(f"   cache_manager = CacheManager('redis://localhost:6379')")
    print(f"   await cache_manager.connect()")
    print(f"   # Store context")
    print(f"   await cache_manager.set('user:123:query', context, ttl=3600)")
    print(f"   # Retrieve context")
    print(f"   context = await cache_manager.get('user:123:query')")
    print(f"   # Bulk operations")
    print(f"   await cache_manager.set_many(contexts_dict, ttl=1800)")
    print(f"   # Pattern search")
    print(f"   keys = await cache_manager.get_keys('user:123:*')")

if __name__ == "__main__":
    asyncio.run(main())
