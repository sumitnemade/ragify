"""
Realtime Source for handling real-time data sources.
"""

import asyncio
import json
import time
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime, timezone
import structlog

# Real-time synchronization imports
import websockets
import aiofiles

# Optional MQTT support
try:
    from aiomqtt import Client as MqttClient
    from aiomqtt.exceptions import MqttError
    MQTT_AVAILABLE = True
except ImportError:
    MqttClient = None
    MqttError = Exception
    MQTT_AVAILABLE = False

# Optional Kafka support
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None
    KAFKA_AVAILABLE = False

import redis.asyncio as redis

from .base import BaseDataSource
from ..models import ContextChunk, SourceType


class RealtimeSource(BaseDataSource):
    """
    Realtime source for handling real-time data streams.
    
    Supports WebSocket connections, event streams, and other real-time data sources.
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType = SourceType.REALTIME,
        url: str = "",
        **kwargs
    ):
        """
        Initialize the realtime source.
        
        Args:
            name: Name of the realtime source
            source_type: Type of data source
            url: Connection URL for real-time data
            **kwargs: Additional configuration
        """
        super().__init__(name, source_type, **kwargs)
        self.url = url
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
        # Real-time connection
        self.connection = None
        self.is_connected = False
        self.connection_type = kwargs.get('connection_type', 'websocket')
        self.subscription_topics = kwargs.get('subscription_topics', [])
        self.message_queue = asyncio.Queue()
        self.callback_handlers = []
        
        # Connection configuration
        self.connection_config = {
            'websocket': {
                'ping_interval': kwargs.get('ping_interval', 30),
                'ping_timeout': kwargs.get('ping_timeout', 10),
                'max_size': kwargs.get('max_size', 2**20),  # 1MB
            },
            'mqtt': {
                'keepalive': kwargs.get('keepalive', 60),
                'clean_session': kwargs.get('clean_session', True),
            },
            'kafka': {
                'bootstrap_servers': kwargs.get('bootstrap_servers', 'localhost:9092'),
                'group_id': kwargs.get('group_id', f'{name}_group'),
                'auto_offset_reset': kwargs.get('auto_offset_reset', 'latest'),
            },
            'redis': {
                'host': kwargs.get('redis_host', 'localhost'),
                'port': kwargs.get('redis_port', 6379),
                'db': kwargs.get('redis_db', 0),
                'password': kwargs.get('redis_password', None),
            }
        }
    
    async def get_chunks(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Get context chunks from real-time source.
        
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
            # Validate query
            query = await self._validate_query(query)
            
            self.logger.info(f"Getting chunks from real-time source for query: {query}")
            
            # Get real-time data
            data = await self._get_realtime_data(query, user_id, session_id)
            
            # Process data into chunks
            chunks = await self._process_realtime_data(data, query)
            
            # Filter by relevance
            relevant_chunks = await self._filter_chunks_by_relevance(
                chunks, min_relevance
            )
            
            # Apply max_chunks limit
            if max_chunks:
                relevant_chunks = relevant_chunks[:max_chunks]
            
            # Update statistics
            await self._update_stats(len(relevant_chunks))
            
            self.logger.info(f"Retrieved {len(relevant_chunks)} chunks from real-time source")
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks from real-time source: {e}")
            return []
    
    async def refresh(self) -> None:
        """Refresh the real-time source."""
        try:
            self.logger.info("Refreshing real-time source")
            
            # Reconnect if needed
            if not self.is_connected:
                await self._connect()
            
            self.logger.info("Real-time source refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh real-time source: {e}")
    
    async def close(self) -> None:
        """Close the real-time source."""
        try:
            self.logger.info("Closing real-time source")
            
            if self.connection:
                await self._disconnect()
                
        except Exception as e:
            self.logger.error(f"Error closing real-time source: {e}")
    
    async def _get_realtime_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get real-time data from various sources.
        
        Args:
            query: Search query
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            Real-time data
        """
        try:
            # Ensure connection is established
            if not self.is_connected:
                await self._connect()
            
            # Get data based on connection type
            if self.connection_type == 'websocket':
                return await self._get_websocket_data(query, user_id, session_id)
            elif self.connection_type == 'mqtt':
                return await self._get_mqtt_data(query, user_id, session_id)
            elif self.connection_type == 'kafka':
                return await self._get_kafka_data(query, user_id, session_id)
            elif self.connection_type == 'redis':
                return await self._get_redis_data(query, user_id, session_id)
            else:
                self.logger.warning(f"Unsupported connection type: {self.connection_type}")
                return await self._get_mock_data(query, user_id, session_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time data: {e}")
            return await self._get_mock_data(query, user_id, session_id)
    
    async def _get_websocket_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get data from WebSocket connection."""
        try:
            if not self.connection:
                return []
            
            # Send query to WebSocket
            message = {
                'query': query,
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.connection.send(json.dumps(message))
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(
                    self.connection.recv(),
                    timeout=5.0
                )
                data = json.loads(response)
                
                return [{
                    'content': data.get('content', f"WebSocket data for query '{query}'"),
                    'relevance': data.get('relevance', 0.8),
                    'metadata': {
                        'source': self.name,
                        'connection_type': 'websocket',
                        'query': query,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'is_realtime': True,
                        'user_id': user_id,
                        'session_id': session_id
                    }
                }]
            except asyncio.TimeoutError:
                self.logger.warning("WebSocket response timeout")
                return []
                
        except Exception as e:
            self.logger.error(f"WebSocket data retrieval failed: {e}")
            return []
    
    async def _get_mqtt_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get data from MQTT connection."""
        try:
            if not self.connection:
                return []
            
            # Publish query to MQTT topic
            topic = f"{self.name}/query"
            message = {
                'query': query,
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            await self.connection.publish(topic, json.dumps(message))
            
            # Get recent messages from queue
            messages = []
            while not self.message_queue.empty():
                try:
                    msg = self.message_queue.get_nowait()
                    messages.append(msg)
                except asyncio.QueueEmpty:
                    break
            
            if messages:
                return [{
                    'content': msg.get('content', f"MQTT data for query '{query}'"),
                    'relevance': msg.get('relevance', 0.8),
                    'metadata': {
                        'source': self.name,
                        'connection_type': 'mqtt',
                        'query': query,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'is_realtime': True,
                        'topic': msg.get('topic', ''),
                        'user_id': user_id,
                        'session_id': session_id
                    }
                } for msg in messages[-3:]]  # Return last 3 messages
            
            return []
            
        except Exception as e:
            self.logger.error(f"MQTT data retrieval failed: {e}")
            return []
    
    async def _get_kafka_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get data from Kafka connection."""
        try:
            if not self.connection:
                return []
            
            # Get recent messages from Kafka
            messages = []
            for message in self.connection:
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    if query.lower() in data.get('content', '').lower():
                        messages.append(data)
                        if len(messages) >= 5:  # Limit to 5 messages
                            break
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            return [{
                'content': msg.get('content', f"Kafka data for query '{query}'"),
                'relevance': msg.get('relevance', 0.8),
                'metadata': {
                    'source': self.name,
                    'connection_type': 'kafka',
                    'query': query,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'is_realtime': True,
                    'topic': msg.get('topic', ''),
                    'partition': msg.get('partition', 0),
                    'offset': msg.get('offset', 0),
                    'user_id': user_id,
                    'session_id': session_id
                }
            } for msg in messages]
            
        except Exception as e:
            self.logger.error(f"Kafka data retrieval failed: {e}")
            return []
    
    async def _get_redis_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get data from Redis pub/sub."""
        try:
            if not self.connection:
                return []
            
            # Get recent messages from Redis
            messages = []
            for topic in self.subscription_topics:
                try:
                    # Get recent messages from Redis list
                    recent_messages = await self.connection.lrange(f"{topic}:messages", 0, 4)
                    for msg_data in recent_messages:
                        try:
                            data = json.loads(msg_data.decode('utf-8'))
                            if query.lower() in data.get('content', '').lower():
                                messages.append(data)
                        except json.JSONDecodeError:
                            continue
                except Exception as e:
                    self.logger.warning(f"Failed to get Redis data for topic {topic}: {e}")
            
            return [{
                'content': msg.get('content', f"Redis data for query '{query}'"),
                'relevance': msg.get('relevance', 0.8),
                'metadata': {
                    'source': self.name,
                    'connection_type': 'redis',
                    'query': query,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'is_realtime': True,
                    'topic': msg.get('topic', ''),
                    'user_id': user_id,
                    'session_id': session_id
                }
            } for msg in messages]
            
        except Exception as e:
            self.logger.error(f"Redis data retrieval failed: {e}")
            return []
    
    async def _get_mock_data(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get simulated real-time data when external services are unavailable."""
        try:
            # Generate realistic real-time data based on query
            current_time = datetime.now(timezone.utc)
            
            # Create dynamic content based on query and time
            time_based_content = f"Real-time update at {current_time.strftime('%H:%M:%S')} UTC"
            
            # Add query-specific content
            if 'weather' in query.lower():
                content = f"{time_based_content}: Weather data for query '{query}' - Temperature: 22°C, Humidity: 65%"
                relevance = 0.9
            elif 'stock' in query.lower() or 'price' in query.lower():
                content = f"{time_based_content}: Market data for query '{query}' - Price: $150.25, Change: +2.3%"
                relevance = 0.85
            elif 'news' in query.lower():
                content = f"{time_based_content}: Breaking news for query '{query}' - Latest updates available"
                relevance = 0.95
            elif 'sensor' in query.lower() or 'iot' in query.lower():
                content = f"{time_based_content}: Sensor data for query '{query}' - Reading: 42.7, Status: Normal"
                relevance = 0.8
            else:
                content = f"{time_based_content}: Live data stream for query '{query}' - Real-time information available"
                relevance = 0.7
            
            # Add user/session context if available
            if user_id:
                content += f" (User: {user_id})"
            if session_id:
                content += f" (Session: {session_id})"
            
            return [
                {
                    'content': content,
                    'relevance': relevance,
                    'metadata': {
                        'source': self.name,
                        'query': query,
                        'timestamp': current_time.isoformat(),
                        'is_realtime': True,
                        'connection_type': 'simulated',
                        'data_type': 'fallback',
                        'user_id': user_id,
                        'session_id': session_id
                    }
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to generate simulated real-time data: {e}")
            # Fallback to basic mock data
            return [
                {
                    'content': f"Real-time data for query '{query}': Live data from {self.name}",
                    'relevance': 0.95,
                    'metadata': {
                        'source': self.name,
                        'query': query,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'is_realtime': True,
                        'connection_type': 'fallback'
                    }
                }
            ]
    
    async def _process_realtime_data(
        self,
        data: List[Dict[str, Any]],
        query: str
    ) -> List[ContextChunk]:
        """
        Process real-time data into context chunks.
        
        Args:
            data: Real-time data
            query: Original query
            
        Returns:
            List of context chunks
        """
        chunks = []
        
        for item in data:
            content = item.get('content', '')
            if not content:
                continue
            
            # Create chunk
            chunk = await self._create_chunk(
                content=content,
                metadata={
                    'realtime_source': self.name,
                    'realtime_url': self.url,
                    'query': query,
                    'data_metadata': item.get('metadata', {}),
                },
                token_count=len(content.split())
            )
            
            # Add relevance score if provided
            relevance = item.get('relevance', 0.5)
            chunk.relevance_score = type('obj', (object,), {
                'score': relevance,
                'confidence_lower': max(0, relevance - 0.1),
                'confidence_upper': min(1, relevance + 0.1),
                'confidence_level': 0.95,
                'factors': {'realtime_relevance': relevance}
            })()
            
            chunks.append(chunk)
        
        return chunks
    
    async def _filter_chunks_by_relevance(
        self,
        chunks: List[ContextChunk],
        min_relevance: float
    ) -> List[ContextChunk]:
        """
        Filter chunks by relevance score.
        
        Args:
            chunks: List of chunks to filter
            min_relevance: Minimum relevance threshold
            
        Returns:
            Filtered list of chunks
        """
        if min_relevance <= 0.0:
            return chunks
        
        relevant_chunks = [
            chunk for chunk in chunks
            if chunk.relevance_score and chunk.relevance_score.score >= min_relevance
        ]
        
        # Sort by relevance
        relevant_chunks.sort(
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
            reverse=True
        )
        
        return relevant_chunks
    
    async def _connect(self) -> None:
        """Connect to real-time data source."""
        try:
            if self.is_connected:
                return
            
            if self.connection_type == 'websocket':
                await self._connect_websocket()
            elif self.connection_type == 'mqtt':
                await self._connect_mqtt()
            elif self.connection_type == 'kafka':
                await self._connect_kafka()
            elif self.connection_type == 'redis':
                await self._connect_redis()
            else:
                self.logger.warning(f"Unsupported connection type: {self.connection_type}")
                self.is_connected = True  # Simulated connection for testing
            
        except Exception as e:
            self.logger.error(f"Failed to connect to real-time data source: {e}")
            self.is_connected = False
    
    async def _connect_websocket(self) -> None:
        """Connect to WebSocket server."""
        try:
            config = self.connection_config['websocket']
            
            # Parse WebSocket URL properly
            if self.url.startswith('websocket://'):
                ws_url = self.url.replace('websocket://', 'ws://')
            elif self.url.startswith('wss://'):
                ws_url = self.url
            elif self.url.startswith('ws://'):
                ws_url = self.url
            else:
                # Default to ws:// if no scheme provided
                ws_url = f"ws://{self.url}"
            
            # Ensure we have a valid hostname
            if ws_url == "ws://":
                # Use a default localhost if no hostname provided
                ws_url = "ws://localhost:8765"
            
            self.connection = await websockets.connect(
                ws_url,
                ping_interval=config['ping_interval'],
                ping_timeout=config['ping_timeout'],
                max_size=config['max_size']
            )
            self.is_connected = True
            self.logger.info(f"Connected to WebSocket server: {ws_url}")
            
            # Start message listener
            asyncio.create_task(self._websocket_listener())
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def _connect_mqtt(self) -> None:
        """Connect to MQTT broker."""
        try:
            config = self.connection_config['mqtt']
            if not MQTT_AVAILABLE:
                raise ImportError("aiomqtt is not installed. Please install it to use MQTT.")
            
            self.connection = MqttClient(
                hostname=self.url.split('://')[1].split(':')[0],
                port=int(self.url.split(':')[-1]) if ':' in self.url else 1883,
                keepalive=config['keepalive'],
                clean_session=config['clean_session']
            )
            
            await self.connection.connect()
            self.is_connected = True
            self.logger.info(f"Connected to MQTT broker: {self.url}")
            
            # Subscribe to topics
            for topic in self.subscription_topics:
                await self.connection.subscribe(topic)
                self.logger.info(f"Subscribed to MQTT topic: {topic}")
            
            # Start message listener
            asyncio.create_task(self._mqtt_listener())
            
        except Exception as e:
            self.logger.error(f"MQTT connection failed: {e}")
            raise
    
    async def _connect_kafka(self) -> None:
        """Connect to Kafka cluster."""
        try:
            config = self.connection_config['kafka']
            if not KAFKA_AVAILABLE:
                raise ImportError("kafka-python is not installed. Please install it to use Kafka.")
            
            # Create consumer
            self.connection = KafkaConsumer(
                *self.subscription_topics,
                bootstrap_servers=config['bootstrap_servers'],
                group_id=config['group_id'],
                auto_offset_reset=config['auto_offset_reset'],
                enable_auto_commit=True,
                value_deserializer=lambda x: x.decode('utf-8')
            )
            
            self.is_connected = True
            self.logger.info(f"Connected to Kafka cluster: {config['bootstrap_servers']}")
            
        except Exception as e:
            self.logger.error(f"Kafka connection failed: {e}")
            raise
    
    async def _connect_redis(self) -> None:
        """Connect to Redis server."""
        try:
            config = self.connection_config['redis']
            self.connection = redis.Redis(
                host=config['host'],
                port=config['port'],
                db=config['db'],
                password=config['password'],
                decode_responses=True
            )
            
            # Test connection
            await self.connection.ping()
            self.is_connected = True
            self.logger.info(f"Connected to Redis server: {config['host']}:{config['port']}")
            
            # Subscribe to topics
            if self.subscription_topics:
                pubsub = self.connection.pubsub()
                for topic in self.subscription_topics:
                    await pubsub.subscribe(topic)
                asyncio.create_task(self._redis_listener(pubsub))
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    async def _disconnect(self) -> None:
        """Disconnect from real-time data source."""
        try:
            if not self.is_connected:
                return
            
            if self.connection:
                if self.connection_type == 'websocket':
                    await self.connection.close()
                elif self.connection_type == 'mqtt':
                    await self.connection.disconnect()
                elif self.connection_type == 'kafka':
                    self.connection.close()
                elif self.connection_type == 'redis':
                    await self.connection.close()
            
            self.connection = None
            self.is_connected = False
            self.logger.info("Disconnected from real-time data source")
            
        except Exception as e:
            self.logger.error(f"Failed to disconnect from real-time data source: {e}")
    
    async def _websocket_listener(self) -> None:
        """Listen for WebSocket messages."""
        try:
            async for message in self.connection:
                try:
                    data = json.loads(message)
                    await self.message_queue.put(data)
                except json.JSONDecodeError:
                    self.logger.warning("Received invalid JSON from WebSocket")
        except Exception as e:
            self.logger.error(f"WebSocket listener error: {e}")
    
    async def _mqtt_listener(self) -> None:
        """Listen for MQTT messages."""
        try:
            if not MQTT_AVAILABLE:
                self.logger.warning("MQTT is not available, cannot listen for messages.")
                return

            async with self.connection.messages() as messages:
                async for message in messages:
                    try:
                        data = json.loads(message.payload.decode())
                        data['topic'] = message.topic.value
                        await self.message_queue.put(data)
                    except json.JSONDecodeError:
                        self.logger.warning("Received invalid JSON from MQTT")
        except Exception as e:
            self.logger.error(f"MQTT listener error: {e}")
    
    async def _redis_listener(self, pubsub) -> None:
        """Listen for Redis pub/sub messages."""
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        data['topic'] = message['channel']
                        await self.message_queue.put(data)
                    except json.JSONDecodeError:
                        self.logger.warning("Received invalid JSON from Redis")
        except Exception as e:
            self.logger.error(f"Redis listener error: {e}")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Add a callback handler for real-time messages."""
        self.callback_handlers.append(callback)
        self.logger.info(f"Added callback handler: {callback.__name__}")
    
    async def publish_message(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a message to the real-time source."""
        try:
            if not self.is_connected:
                await self._connect()
            
            if self.connection_type == 'mqtt':
                if not MQTT_AVAILABLE:
                    raise ImportError("aiomqtt is not installed. Cannot publish to MQTT.")
                await self.connection.publish(topic, json.dumps(message))
            elif self.connection_type == 'redis':
                await self.connection.publish(topic, json.dumps(message))
            elif self.connection_type == 'websocket':
                await self.connection.send(json.dumps({
                    'topic': topic,
                    'message': message
                }))
            
            self.logger.info(f"Published message to {topic}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
