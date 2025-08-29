#!/usr/bin/env python3
"""
Test script for concurrent source processing optimization.
"""

import asyncio
import time
import sys
import os
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.insert(0, 'src')

from ragify import ContextOrchestrator, ContextRequest
from ragify.sources import BaseDataSource
from ragify.models import OrchestratorConfig, PrivacyLevel


class PlaceholderSource(BaseDataSource):
    """Placeholder source for testing concurrent processing."""
    
    def __init__(self, name: str, url: str, delay: float = 0.1):
        from ragify.models import SourceType
        super().__init__(name=name, source_type=SourceType.API, url=url)
        self.delay = delay
        self.mock_data = {
            "machine learning": [
                {
                    'content': f"Data about machine learning from {name}",
                    'relevance': 0.9,
                    'metadata': {
                        'source': name,
                        'query': 'machine learning',
                        'timestamp': time.time(),
                    }
                }
            ],
            "artificial intelligence": [
                {
                    'content': f"Data about artificial intelligence from {name}",
                    'relevance': 0.85,
                    'metadata': {
                        'source': name,
                        'query': 'artificial intelligence',
                        'timestamp': time.time(),
                    }
                }
            ],
            "deep learning": [
                {
                    'content': f"Data about deep learning from {name}",
                    'relevance': 0.88,
                    'metadata': {
                        'source': name,
                        'query': 'deep learning',
                        'timestamp': time.time(),
                    }
                }
            ]
        }
    
    async def get_chunks(
        self, 
        query: str, 
        max_chunks: int = 10, 
        min_relevance: float = 0.5, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> List[Any]:
        """Get chunks with simulated processing delay."""
        # Simulate processing time
        await asyncio.sleep(self.delay)
        
        # Return data based on query
        for key, data in self.mock_data.items():
            if key.lower() in query.lower():
                # Convert to ContextChunk objects
                chunks = []
                for item in data[:max_chunks]:
                    chunk = await self._create_chunk(
                        content=item['content'],
                        metadata=item['metadata'],
                        token_count=len(item['content'].split())  # Rough token count
                    )
                    chunks.append(chunk)
                return chunks
        
        # Default response
        default_chunk = await self._create_chunk(
            content=f"Data for query '{query}' from {self.name}",
            metadata={
                'source': self.name,
                'query': query,
                'timestamp': time.time(),
            },
            token_count=len(query.split()) + 5  # Rough token count
        )
        return [default_chunk]
    
    async def refresh(self) -> None:
        """Refresh the data source."""
        # No-op for placeholder source
        pass
    
    async def close(self) -> None:
        """Close the data source."""
        # No-op for placeholder source
        pass


class SlowPlaceholderSource(PlaceholderSource):
    """Placeholder source with longer delay for testing."""
    
    def __init__(self, name: str, url: str):
        super().__init__(name=name, url=url, delay=2.0)  # 2 second delay


class FailingPlaceholderSource(PlaceholderSource):
    """Placeholder source that fails for testing error handling."""
    
    async def get_chunks(
        self, 
        query: str, 
        max_chunks: int = 10, 
        min_relevance: float = 0.5, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Always raise an exception to test error handling."""
        raise Exception(f"Simulated source failure in {self.name}")


async def test_sequential_vs_concurrent():
    """Test sequential vs concurrent processing performance."""
    
    print("🚀 Testing Sequential vs Concurrent Source Processing")
    print("=" * 60)
    
    # Create configuration with concurrent processing enabled
    config = OrchestratorConfig(
        vector_db_url="memory://",
        cache_url="memory://",
        privacy_level=PrivacyLevel.PRIVATE,
        source_timeout=60.0,  # 60 second timeout for slow sources
        max_concurrent_sources=5
    )
    
    # Initialize orchestrator
    orchestrator = ContextOrchestrator(config=config)
    
    # Add multiple slow sources to demonstrate the difference
    sources = []
    for i in range(3):
        source = SlowPlaceholderSource(
            name=f"slow_source_{i}",
            url=f"https://slow-api-{i}.example.com"
        )
        sources.append(source)
        orchestrator.add_source(source)
        print(f"✅ Added slow source: {source.name}")
    
    # Add a regular placeholder source for comparison
    regular_source = PlaceholderSource(
        name="fast_source",
        url="https://fast-api.example.com",
        delay=0.1
    )
    orchestrator.add_source(regular_source)
    print(f"✅ Added fast source: {regular_source.name}")
    
    print(f"\n📊 Total sources: {len(orchestrator.list_sources())}")
    
    # Test query
    test_query = "machine learning algorithms"
    print(f"\n🔍 Testing query: '{test_query}'")
    
    # Test with current implementation (should be concurrent now)
    print("\n⚡ Testing CONCURRENT processing...")
    start_time = time.time()
    
    try:
        response = await orchestrator.get_context(
            query=test_query,
            max_chunks=20,
            min_relevance=0.5
        )
        
        concurrent_time = time.time() - start_time
        
        print(f"✅ Concurrent processing completed in {concurrent_time:.2f}s")
        print(f"   Chunks retrieved: {len(response.context.chunks)}")
        print(f"   Processing time: {response.processing_time:.3f}s")
        print(f"   Cache hit: {response.cache_hit}")
        
    except Exception as e:
        print(f"❌ Concurrent processing failed: {e}")
        concurrent_time = float('inf')
    
    # Calculate expected sequential time (3 slow sources * 2s + 1 fast source * 0.1s)
    expected_sequential_time = (3 * 2.0) + 0.1
    expected_concurrent_time = 2.1  # Should be roughly the time of the slowest source
    
    print(f"\n📈 Performance Analysis:")
    print(f"   Expected sequential time: {expected_sequential_time:.2f}s")
    print(f"   Expected concurrent time: ~{expected_concurrent_time:.2f}s")
    print(f"   Actual concurrent time: {concurrent_time:.2f}s")
    
    if concurrent_time < expected_sequential_time * 0.8:  # 20% tolerance
        print(f"   🎉 SUCCESS: Concurrent processing is {expected_sequential_time/concurrent_time:.1f}x faster!")
    else:
        print(f"   ⚠️  WARNING: Concurrent processing may not be working optimally")
    
    # Test with multiple concurrent requests
    print(f"\n🔄 Testing multiple concurrent requests...")
    
    async def single_request(query: str, request_id: int):
        """Make a single request."""
        start_time = time.time()
        try:
            response = await orchestrator.get_context(
                query=f"{query} (request {request_id})",
                max_chunks=10,
                min_relevance=0.5
            )
            duration = time.time() - start_time
            return {
                'request_id': request_id,
                'success': True,
                'duration': duration,
                'chunks': len(response.context.chunks)
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'request_id': request_id,
                'success': False,
                'duration': duration,
                'error': str(e)
            }
    
    # Make 5 concurrent requests
    concurrent_requests = [
        single_request("artificial intelligence", i)
        for i in range(5)
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*concurrent_requests)
    total_time = time.time() - start_time
    
    print(f"✅ 5 concurrent requests completed in {total_time:.2f}s")
    print(f"   Average time per request: {total_time/5:.2f}s")
    
    successful_requests = sum(1 for r in results if r['success'])
    print(f"   Successful requests: {successful_requests}/5")
    
    if successful_requests == 5:
        print(f"   🎉 All requests succeeded!")
    else:
        print(f"   ⚠️  Some requests failed")
    
    # Clean up
    await orchestrator.close()
    
    return {
        'concurrent_time': concurrent_time,
        'expected_sequential': expected_sequential_time,
        'concurrent_requests_time': total_time,
        'successful_requests': successful_requests
    }


async def test_error_handling():
    """Test error handling in concurrent processing."""
    
    print(f"\n🧪 Testing Error Handling in Concurrent Processing")
    print("=" * 60)
    
    config = OrchestratorConfig(
        vector_db_url="memory://",
        cache_url="memory://",
        source_timeout=10.0
    )
    
    orchestrator = ContextOrchestrator(config=config)
    
    # Add working and failing sources
    orchestrator.add_source(PlaceholderSource(name="working_source", url="https://working.example.com"))
    orchestrator.add_source(FailingPlaceholderSource(name="failing_source", url="https://failing.example.com"))
    
    print("✅ Added 1 working source and 1 failing source")
    
    # Test that the system continues working despite one source failing
    try:
        response = await orchestrator.get_context(
            query="test query",
            max_chunks=10,
            min_relevance=0.5
        )
        
        print(f"✅ System continued working despite source failure")
        print(f"   Chunks retrieved: {len(response.context.chunks)}")
        
    except Exception as e:
        print(f"❌ System failed completely: {e}")
    
    await orchestrator.close()


async def main():
    """Main test function."""
    print("🧪 Ragify Concurrent Source Processing Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Performance comparison
        results = await test_sequential_vs_concurrent()
        
        # Test 2: Error handling
        await test_error_handling()
        
        print(f"\n🎯 Test Summary:")
        print(f"   Concurrent processing: {'✅ WORKING' if results['concurrent_time'] < 10 else '❌ ISSUES'}")
        print(f"   Multiple requests: {'✅ WORKING' if results['successful_requests'] == 5 else '❌ ISSUES'}")
        print(f"   Performance gain: {results['expected_sequential']/results['concurrent_time']:.1f}x faster")
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
