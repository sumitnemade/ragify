#!/usr/bin/env python3
"""
Real-time Synchronization Demo for Ragify Framework

This demo showcases the comprehensive real-time synchronization capabilities
including WebSocket, MQTT, Redis, and Kafka connections.
"""

import asyncio
import json
from datetime import datetime
from ragify.sources.realtime import RealtimeSource
from ragify.models import SourceType, PrivacyLevel
from ragify.core import ContextOrchestrator
from ragify.exceptions import ICOException

async def demo_websocket_sync():
    """Demonstrate WebSocket real-time synchronization."""
    
    print("🔌 WebSocket Real-time Synchronization Demo")
    print("=" * 50)
    
    # Initialize WebSocket real-time source
    websocket_source = RealtimeSource(
        name="websocket_demo",
        source_type=SourceType.REALTIME,
        url="ws://echo.websocket.org",  # Public echo server
        connection_type="websocket",
        subscription_topics=["demo/topic"],
        ping_interval=30,
        ping_timeout=10
    )
    
    print(f"📡 WebSocket source configured:")
    print(f"   - URL: {websocket_source.url}")
    print(f"   - Connection type: {websocket_source.connection_type}")
    print(f"   - Topics: {websocket_source.subscription_topics}")
    
    try:
        # Test connection
        print(f"\n🔌 Attempting connection...")
        await websocket_source._connect()
        print(f"✅ Connection status: {websocket_source.is_connected}")
        
        if websocket_source.is_connected:
            # Test data retrieval
            print(f"\n📡 Testing real-time data retrieval...")
            chunks = await websocket_source.get_chunks(
                query="live data",
                max_chunks=2,
                min_relevance=0.1
            )
            
            print(f"✅ Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {chunk.content[:60]}...")
                if chunk.relevance_score:
                    print(f"     Relevance: {chunk.relevance_score.score:.2f}")
            
            # Test message publishing
            print(f"\n📤 Testing message publishing...")
            test_message = {
                'content': 'Hello from Ragify WebSocket demo!',
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'ragify_demo'
            }
            
            await websocket_source.publish_message("demo/topic", test_message)
            print(f"✅ Message published successfully")
        
    except Exception as e:
        print(f"❌ WebSocket demo failed: {e}")
        print(f"   This is expected if the echo server is not available")
    
    finally:
        await websocket_source.close()
        print(f"🔌 WebSocket connection closed")

async def demo_mqtt_sync():
    """Demonstrate MQTT real-time synchronization."""
    
    print(f"\n📡 MQTT Real-time Synchronization Demo")
    print("=" * 50)
    
    # Initialize MQTT real-time source
    mqtt_source = RealtimeSource(
        name="mqtt_demo",
        source_type=SourceType.REALTIME,
        url="mqtt://test.mosquitto.org:1883",  # Public MQTT broker
        connection_type="mqtt",
        subscription_topics=["ragify/demo"],
        keepalive=60,
        clean_session=True
    )
    
    print(f"📡 MQTT source configured:")
    print(f"   - URL: {mqtt_source.url}")
    print(f"   - Connection type: {mqtt_source.connection_type}")
    print(f"   - Topics: {mqtt_source.subscription_topics}")
    
    try:
        # Test connection
        print(f"\n🔌 Attempting connection...")
        await mqtt_source._connect()
        print(f"✅ Connection status: {mqtt_source.is_connected}")
        
        if mqtt_source.is_connected:
            # Test data retrieval
            print(f"\n📡 Testing real-time data retrieval...")
            chunks = await mqtt_source.get_chunks(
                query="mqtt data",
                max_chunks=2,
                min_relevance=0.1
            )
            
            print(f"✅ Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {chunk.content[:60]}...")
                if chunk.relevance_score:
                    print(f"     Relevance: {chunk.relevance_score.score:.2f}")
            
            # Test message publishing
            print(f"\n📤 Testing message publishing...")
            test_message = {
                'content': 'Hello from Ragify MQTT demo!',
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'ragify_demo'
            }
            
            await mqtt_source.publish_message("ragify/demo", test_message)
            print(f"✅ Message published successfully")
        
    except Exception as e:
        print(f"❌ MQTT demo failed: {e}")
        print(f"   This is expected if the MQTT broker is not available")
    
    finally:
        await mqtt_source.close()
        print(f"🔌 MQTT connection closed")

async def demo_redis_sync():
    """Demonstrate Redis real-time synchronization."""
    
    print(f"\n🔴 Redis Real-time Synchronization Demo")
    print("=" * 50)
    
    # Initialize Redis real-time source
    redis_source = RealtimeSource(
        name="redis_demo",
        source_type=SourceType.REALTIME,
        url="redis://localhost:6379",  # Local Redis
        connection_type="redis",
        subscription_topics=["ragify:demo"],
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    print(f"📡 Redis source configured:")
    print(f"   - URL: {redis_source.url}")
    print(f"   - Connection type: {redis_source.connection_type}")
    print(f"   - Topics: {redis_source.subscription_topics}")
    
    try:
        # Test connection
        print(f"\n🔌 Attempting connection...")
        await redis_source._connect()
        print(f"✅ Connection status: {redis_source.is_connected}")
        
        if redis_source.is_connected:
            # Test data retrieval
            print(f"\n📡 Testing real-time data retrieval...")
            chunks = await redis_source.get_chunks(
                query="redis data",
                max_chunks=2,
                min_relevance=0.1
            )
            
            print(f"✅ Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {chunk.content[:60]}...")
                if chunk.relevance_score:
                    print(f"     Relevance: {chunk.relevance_score.score:.2f}")
            
            # Test message publishing
            print(f"\n📤 Testing message publishing...")
            test_message = {
                'content': 'Hello from Ragify Redis demo!',
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'ragify_demo'
            }
            
            await redis_source.publish_message("ragify:demo", test_message)
            print(f"✅ Message published successfully")
        
    except Exception as e:
        print(f"❌ Redis demo failed: {e}")
        print(f"   This is expected if Redis is not running locally")
    
    finally:
        await redis_source.close()
        print(f"🔌 Redis connection closed")

async def demo_websocket_sync():
    """Demonstrate actual WebSocket real-time synchronization."""
    
    print(f"\n🌐 WebSocket Real-time Synchronization Demo")
    print("=" * 50)
    
    # Initialize WebSocket real-time source
    websocket_source = RealtimeSource(
        name="websocket_demo",
        source_type=SourceType.REALTIME,
        url="ws://echo.websocket.org",
        connection_type="websocket",
        subscription_topics=["demo/topic"]
    )
    
    print(f"📡 WebSocket source configured:")
    print(f"   - URL: {websocket_source.url}")
    print(f"   - Connection type: {websocket_source.connection_type}")
    print(f"   - Topics: {websocket_source.subscription_topics}")
    
    try:
        # Test connection
        print(f"\n🔌 Attempting WebSocket connection...")
        await websocket_source.connect()
        print(f"✅ Connection status: {websocket_source.is_connected}")
        
        # Test data retrieval
        print(f"\n📡 Testing real-time data retrieval...")
        chunks = await websocket_source.get_chunks(
            query="websocket data",
            max_chunks=2,
            min_relevance=0.1
        )
        
        print(f"✅ Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk.content[:60]}...")
            if chunk.relevance_score:
                print(f"     Relevance: {chunk.relevance_score.score:.2f}")
        
        # Test message publishing
        print(f"\n📤 Testing message publishing...")
        test_message = {
            'content': 'Hello from Ragify WebSocket demo!',
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'ragify_demo'
        }
        
        await websocket_source.publish_message("demo/topic", test_message)
        print(f"✅ Message published successfully")
        
    except Exception as e:
        print(f"❌ WebSocket demo failed: {e}")
    
    finally:
        await websocket_source.close()
        print(f"🔌 WebSocket connection closed")

async def demo_callback_system():
    """Demonstrate callback handler system."""
    
    print(f"\n🔄 Callback Handler System Demo")
    print("=" * 50)
    
    # Initialize real-time source
    source = RealtimeSource(
        name="callback_demo",
        source_type=SourceType.REALTIME,
        url="ws://echo.websocket.org",
        connection_type="websocket"
    )
    
    # Define callback handlers
    async def message_logger(message):
        print(f"📝 Logging message: {message.get('content', 'No content')}")
    
    async def alert_detector(message):
        if 'alert' in message.get('content', '').lower():
            print(f"🚨 ALERT DETECTED: {message.get('content')}")
    
    async def data_processor(message):
        if 'data' in message.get('content', '').lower():
            print(f"⚙️  Processing data: {message.get('content')}")
    
    # Add callback handlers
    source.add_callback(message_logger)
    source.add_callback(alert_detector)
    source.add_callback(data_processor)
    
    print(f"✅ Added {len(source.callback_handlers)} callback handlers")
    
    # Test different message types
    test_messages = [
        {
            'content': 'This is a normal message',
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'content': 'This is an alert message',
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'content': 'This contains data for processing',
            'timestamp': datetime.utcnow().isoformat()
        }
    ]
    
    print(f"\n📤 Testing callback execution...")
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Message {i} ---")
        for callback in source.callback_handlers:
            await callback(message)
    
    await source.close()
    print(f"🔌 Callback demo completed")

async def demo_with_orchestrator():
    """Demonstrate real-time sources with Context Orchestrator."""
    
    print(f"\n🚀 Real-time Sources with Context Orchestrator Demo")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = ContextOrchestrator(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    
    # Add real-time sources
    websocket_source = RealtimeSource(
        name="orchestrator_demo",
        source_type=SourceType.REALTIME,
        url="ws://echo.websocket.org",
        connection_type="websocket",
        subscription_topics=["orchestrator/topic"]
    )
    
    orchestrator.add_source(websocket_source)
    print(f"✅ Added real-time source to orchestrator")
    
    # Test context retrieval with real-time data
    print(f"\n🔍 Testing context retrieval with real-time data...")
    context_response = await orchestrator.get_context(
        query="What is the latest real-time information?",
        user_id="demo_user",
        max_tokens=1000,
        min_relevance=0.3
    )
    
    print(f"Context retrieved:")
    print(f"  - Processing time: {context_response.processing_time:.3f}s")
    print(f"  - Total chunks: {len(context_response.context.chunks)}")
    print(f"  - Sources: {[s.name for s in context_response.context.sources]}")
    
    if context_response.context.chunks:
        print(f"  - Sample content: {context_response.context.chunks[0].content[:100]}...")
    
    # Close orchestrator
    await orchestrator.close()
    print(f"🔌 Orchestrator demo completed")

async def main():
    """Run the complete real-time synchronization demo."""
    
    print("🎯 Ragify Real-time Synchronization Demo")
    print("=" * 60)
    print("This demo showcases comprehensive real-time synchronization capabilities")
    print("including WebSocket, MQTT, and Redis connections.\n")
    
    # Demo different real-time sources
    await demo_websocket_sync()  # WebSocket demo
    await demo_callback_system()
    await demo_with_orchestrator()
    
    # Demo external services (may fail if not available)
    await demo_mqtt_sync()
    await demo_redis_sync()
    
    print(f"\n🎉 Complete real-time synchronization demo finished!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✅ Multiple connection types (WebSocket, MQTT, Redis)")
    print(f"   ✅ Real-time data retrieval and processing")
    print(f"   ✅ Message publishing capabilities")
    print(f"   ✅ Callback handler system")
    print(f"   ✅ Integration with Context Orchestrator")
    print(f"   ✅ Connection management and error handling")
    print(f"   ✅ Graceful error handling and fallbacks")
    print(f"\n📚 Usage Examples:")
    print(f"   # WebSocket connection")
    print(f"   source = RealtimeSource(url='ws://server:port', connection_type='websocket')")
    print(f"   # MQTT connection")
    print(f"   source = RealtimeSource(url='mqtt://broker:port', connection_type='mqtt')")
    print(f"   # Redis connection")
    print(f"   source = RealtimeSource(url='redis://server:port', connection_type='redis')")
    print(f"   # Error handling and fallbacks")
    print(f"   source = RealtimeSource(url='ws://localhost:8080', connection_type='websocket')")

if __name__ == "__main__":
    asyncio.run(main())
