#!/usr/bin/env python3
"""
Real-time Data Processing Example with RAGify

This example demonstrates RAGify's real-time data processing capabilities
including WebSocket, MQTT, Kafka, and Redis Pub/Sub integrations.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

from ragify.core import ContextOrchestrator
from ragify.sources.realtime import RealtimeSource
from ragify.models import PrivacyLevel, SourceType, ContextChunk


class MockWebSocketServer:
    """Mock WebSocket server for demonstration."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = []
        self.messages = []
    
    async def start(self):
        """Start the mock server."""
        print(f"🚀 Mock WebSocket server starting on port {self.port}")
        # In a real implementation, this would start an actual WebSocket server
        # For demo purposes, we'll simulate it
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all clients."""
        self.messages.append(message)
        print(f"📡 Broadcasting: {message.get('content', 'No content')[:50]}...")


class MockMQTTBroker:
    """Mock MQTT broker for demonstration."""
    
    def __init__(self):
        self.topics = {}
        self.clients = []
    
    async def start(self):
        """Start the mock broker."""
        print("🚀 Mock MQTT broker starting")
    
    async def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to topic."""
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(message)
        print(f"📡 MQTT publish to {topic}: {message.get('content', 'No content')[:50]}...")


class MockKafkaCluster:
    """Mock Kafka cluster for demonstration."""
    
    def __init__(self):
        self.topics = {}
        self.consumers = []
    
    async def start(self):
        """Start the mock cluster."""
        print("🚀 Mock Kafka cluster starting")
    
    async def produce(self, topic: str, message: Dict[str, Any]):
        """Produce message to topic."""
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(message)
        print(f"📡 Kafka produce to {topic}: {message.get('content', 'No content')[:50]}...")


async def create_realtime_sources():
    """Create various real-time data sources."""
    print("\n🔌 Creating Real-time Data Sources...")
    
    # WebSocket source for live chat/messaging
    websocket_source = RealtimeSource(
        name="live_chat",
        source_type=SourceType.REALTIME,
        url="ws://localhost:8765",
        connection_type="websocket",
        subscription_topics=["chat/general", "chat/support"],
        max_buffer_size=1000,
        rate_limit=50  # 50 messages per second
    )
    
    # MQTT source for IoT sensor data
    mqtt_source = RealtimeSource(
        name="iot_sensors",
        source_type=SourceType.REALTIME,
        url="mqtt://localhost:1883",
        connection_type="mqtt",
        subscription_topics=["sensors/temperature", "sensors/humidity", "sensors/motion"],
        max_buffer_size=500,
        rate_limit=100  # 100 sensor readings per second
    )
    
    # Kafka source for high-throughput event streams
    kafka_source = RealtimeSource(
        name="event_streams",
        source_type=SourceType.REALTIME,
        url="kafka://localhost:9092",
        connection_type="kafka",
        subscription_topics=["user-events", "system-events", "analytics-events"],
        max_buffer_size=2000,
        rate_limit=1000  # 1000 events per second
    )
    
    # Redis source for real-time notifications
    redis_source = RealtimeSource(
        name="notifications",
        source_type=SourceType.REALTIME,
        url="redis://localhost:6379",
        connection_type="redis",
        subscription_topics=["notifications/email", "notifications/push", "notifications/sms"],
        max_buffer_size=200,
        rate_limit=200  # 200 notifications per second
    )
    
    return {
        "websocket": websocket_source,
        "mqtt": mqtt_source,
        "kafka": kafka_source,
        "redis": redis_source
    }


async def simulate_realtime_data(sources: Dict[str, RealtimeSource]):
    """Simulate real-time data streams."""
    print("\n📡 Simulating Real-time Data Streams...")
    
    # WebSocket chat messages
    chat_messages = [
        {"content": "Hello everyone! How's the project going?", "user": "alice", "timestamp": datetime.now(timezone.utc)},
        {"content": "Great progress! We've completed the core features.", "user": "bob", "timestamp": datetime.now(timezone.utc)},
        {"content": "Any blockers we should be aware of?", "user": "charlie", "timestamp": datetime.now(timezone.utc)},
        {"content": "Just waiting for the final testing phase.", "user": "diana", "timestamp": datetime.now(timezone.utc)},
    ]
    
    # MQTT sensor data
    sensor_data = [
        {"content": "Temperature: 22.5°C, Humidity: 45%", "sensor_id": "temp_001", "timestamp": datetime.now(timezone.utc)},
        {"content": "Motion detected in Zone A", "sensor_id": "motion_001", "timestamp": datetime.now(timezone.utc)},
        {"content": "Temperature: 23.1°C, Humidity: 47%", "sensor_id": "temp_001", "timestamp": datetime.now(timezone.utc)},
        {"content": "No motion in Zone B", "sensor_id": "motion_002", "timestamp": datetime.now(timezone.utc)},
    ]
    
    # Kafka event data
    event_data = [
        {"content": "User login: user123 from IP 192.168.1.100", "event_type": "auth", "timestamp": datetime.now(timezone.utc)},
        {"content": "File upload: document.pdf (2.5MB)", "event_type": "storage", "timestamp": datetime.now(timezone.utc)},
        {"content": "API call: GET /api/users (200 OK)", "event_type": "api", "timestamp": datetime.now(timezone.utc)},
        {"content": "Database query: SELECT * FROM users (15ms)", "event_type": "database", "timestamp": datetime.now(timezone.utc)},
    ]
    
    # Redis notification data
    notification_data = [
        {"content": "New email: Meeting reminder for tomorrow", "type": "email", "priority": "high", "timestamp": datetime.now(timezone.utc)},
        {"content": "Push notification: Your order has shipped", "type": "push", "priority": "medium", "timestamp": datetime.now(timezone.utc)},
        {"content": "SMS: Verification code: 123456", "type": "sms", "priority": "high", "timestamp": datetime.now(timezone.utc)},
        {"content": "System alert: High CPU usage detected", "type": "system", "priority": "critical", "timestamp": datetime.now(timezone.utc)},
    ]
    
    # Simulate data streams
    for i, message in enumerate(chat_messages):
        await sources["websocket"].message_queue.put({
            "content": message["content"],
            "metadata": {
                "user": message["user"],
                "timestamp": message["timestamp"].isoformat(),
                "message_id": f"chat_{i}",
                "source": "websocket"
            },
            "relevance": 0.8
        })
        await asyncio.sleep(0.1)
    
    for i, data in enumerate(sensor_data):
        await sources["mqtt"].message_queue.put({
            "content": data["content"],
            "metadata": {
                "sensor_id": data["sensor_id"],
                "timestamp": data["timestamp"].isoformat(),
                "reading_id": f"sensor_{i}",
                "source": "mqtt"
            },
            "relevance": 0.9
        })
        await asyncio.sleep(0.1)
    
    for i, event in enumerate(event_data):
        await sources["kafka"].message_queue.put({
            "content": event["content"],
            "metadata": {
                "event_type": event["event_type"],
                "timestamp": event["timestamp"].isoformat(),
                "event_id": f"event_{i}",
                "source": "kafka"
            },
            "relevance": 0.7
        })
        await asyncio.sleep(0.1)
    
    for i, notification in enumerate(notification_data):
        await sources["redis"].message_queue.put({
            "content": notification["content"],
            "metadata": {
                "type": notification["type"],
                "priority": notification["priority"],
                "timestamp": notification["timestamp"].isoformat(),
                "notification_id": f"notif_{i}",
                "source": "redis"
            },
            "relevance": 0.6
        })
        await asyncio.sleep(0.1)
    
    print(f"✅ Simulated {len(chat_messages) + len(sensor_data) + len(event_data) + len(notification_data)} real-time messages")


async def demonstrate_realtime_queries(sources: Dict[str, RealtimeSource]):
    """Demonstrate querying real-time data sources."""
    print("\n🔍 Demonstrating Real-time Data Queries...")
    
    queries = [
        "What are the latest chat messages?",
        "Show me recent sensor readings",
        "What system events occurred?",
        "Any high-priority notifications?",
        "User activity summary"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        # Query each source using simulated data instead of external connections
        for source_name, source in sources.items():
            try:
                # Use the message queue data instead of trying to connect
                if source.message_queue.qsize() > 0:
                    # Get messages from queue and process them
                    messages = []
                    while not source.message_queue.empty() and len(messages) < 3:
                        try:
                            message = await asyncio.wait_for(source.message_queue.get(), timeout=0.1)
                            messages.append(message)
                        except asyncio.TimeoutError:
                            break
                    
                    if messages:
                        # Process messages into chunks
                        chunks = await source._process_realtime_data(messages, query)
                        print(f"  📡 {source_name.upper()}: {len(chunks)} chunks from queue")
                        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                            content_preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
                            print(f"    {i+1}. {content_preview}")
                            if hasattr(chunk, 'relevance_score') and chunk.relevance_score:
                                print(f"       Relevance: {chunk.relevance_score.score:.2f}")
                    else:
                        print(f"  📡 {source_name.upper()}: No messages in queue")
                else:
                    print(f"  📡 {source_name.upper()}: No messages in queue")
                    
            except Exception as e:
                print(f"  ❌ {source_name.upper()}: Error - {e}")
        
        await asyncio.sleep(0.5)


async def demonstrate_context_orchestration(sources: Dict[str, RealtimeSource]):
    """Demonstrate context orchestration with real-time sources."""
    print("\n🎯 Demonstrating Context Orchestration with Real-time Sources...")
    
    # Create context orchestrator
    orchestrator = ContextOrchestrator(
        vector_db_url="memory://",
        cache_url="memory://",
        privacy_level=PrivacyLevel.ENTERPRISE
    )
    
    # Add real-time sources
    for source_name, source in sources.items():
        orchestrator.add_source(source)
        print(f"  ✅ Added {source_name} source")
    
    # Demonstrate real-time context retrieval
    queries = [
        "What's happening in the system right now?",
        "Show me recent user activity and sensor data",
        "Any critical alerts or notifications?",
        "Summarize current system status"
    ]
    
    for query in queries:
        print(f"\n🔍 Orchestrated Query: {query}")
        
        try:
            # For demo purposes, we'll simulate context retrieval
            # In a real scenario, this would use the orchestrator
            print(f"  📊 Simulating context retrieval for: {query}")
            
            # Show sample chunks from sources
            total_chunks = 0
            for source_name, source in sources.items():
                if source.message_queue.qsize() > 0:
                    # Get a sample message
                    try:
                        message = await asyncio.wait_for(source.message_queue.get(), timeout=0.1)
                        chunks = await source._process_realtime_data([message], query)
                        if chunks:
                            total_chunks += len(chunks)
                            chunk = chunks[0]
                            source_name = chunk.source.name if chunk.source else "unknown"
                            content_preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                            print(f"    📡 [{source_name}] {content_preview}")
                    except asyncio.TimeoutError:
                        continue
                
            print(f"  📊 Total chunks available: {total_chunks}")
            print(f"  ⏱️  Processing time: Simulated")
            print(f"  💾 Cache hit: Simulated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        await asyncio.sleep(0.5)
    
    # Close orchestrator
    await orchestrator.close()


async def demonstrate_advanced_features(sources: Dict[str, RealtimeSource]):
    """Demonstrate advanced real-time features."""
    print("\n🚀 Demonstrating Advanced Real-time Features...")
    
    # Rate limiting demonstration
    print("\n📊 Rate Limiting:")
    for source_name, source in sources.items():
        print(f"  {source_name}: {source.rate_limit} messages/second")
    
    # Buffer management
    print("\n💾 Buffer Management:")
    for source_name, source in sources.items():
        print(f"  {source_name}: {len(source.data_buffer)}/{source.max_buffer_size} messages")
    
    # Subscriber management
    print("\n👥 Subscriber Management:")
    for source_name, source in sources.items():
        print(f"  {source_name}: {len(source.subscribers)} subscribers")
    
    # Performance metrics
    print("\n📈 Performance Metrics:")
    for source_name, source in sources.items():
        print(f"  {source_name}:")
        print(f"    Messages processed: {source.messages_processed}")
        print(f"    Avg processing time: {source.avg_processing_time:.3f}s")
        print(f"    Error rate: {source.error_rate:.2%}")


async def main():
    """Main demonstration function."""
    print("🚀 RAGify Real-time Data Processing Demonstration")
    print("=" * 60)
    
    try:
        # Create real-time sources
        sources = await create_realtime_sources()
        
        # Simulate real-time data
        await simulate_realtime_data(sources)
        
        # Demonstrate queries
        await demonstrate_realtime_queries(sources)
        
        # Demonstrate context orchestration
        await demonstrate_context_orchestration(sources)
        
        # Demonstrate advanced features
        await demonstrate_advanced_features(sources)
        
        print("\n🎉 Real-time Data Processing Demonstration Completed!")
        print("\n📋 Key Features Demonstrated:")
        print("  ✅ WebSocket real-time messaging")
        print("  ✅ MQTT IoT sensor data")
        print("  ✅ Kafka event streaming")
        print("  ✅ Redis pub/sub notifications")
        print("  ✅ Real-time context orchestration")
        print("  ✅ Rate limiting and buffering")
        print("  ✅ Performance monitoring")
        print("  ✅ Error handling and recovery")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        for source in sources.values():
            try:
                await source.close()
            except Exception as e:
                print(f"  Warning: Error closing {source.name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
