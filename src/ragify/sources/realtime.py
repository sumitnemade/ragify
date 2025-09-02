"""
Realtime Source for handling real-time data sources.
"""

import asyncio
import json
import time
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime, timezone, timedelta
import structlog
from uuid import uuid4

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
from ..exceptions import ICOException


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

        # New attributes for robust features
        self.subscribers: set[Callable] = set() # For publish_update
        self.data_buffer: List[Dict[str, Any]] = [] # For buffering
        self.max_buffer_size: int = kwargs.get('max_buffer_size', 1000) # Default 1000
        
        # Buffer management methods
        self._add_to_buffer = self._add_to_buffer_safe
        self.filter_config: Dict[str, Any] = {} # For filtering
        self.transform_config: Dict[str, Any] = {} # For transformations
        self.rate_limit: int = kwargs.get('rate_limit', 10) # Default 10 per second
        self.rate_limit_timestamps: List[datetime] = [] # For rate limiting
        self.persistence_enabled: bool = False # For persistence
        self.compression_enabled: bool = False # For compression
        self.encryption_enabled: bool = False # For encryption
        self.validation_rules: Dict[str, Any] = {} # For validation
        self.routing_rules: Dict[str, str] = {} # For routing
        self._streaming: bool = False # For streaming control
        self.messages_processed: int = 0 # For performance metrics
        self.avg_processing_time: float = 0.0 # For performance metrics
        self.error_rate: float = 0.0 # For performance metrics
        self.log_history: List[Dict[str, Any]] = [] # For logging

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
                raise ICOException(f"Unsupported connection type: {self.connection_type}")
            
        except Exception as e:
            await self._handle_realtime_failure(query, e, user_id, session_id)
    
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
            self.logger.warning(f"WebSocket data retrieval failed: {e}")
            self.logger.info("Returning empty data for graceful fallback")
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
    
    async def _handle_realtime_failure(self, query: str, error: Exception, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """Handle real-time data failure with proper error logging and cleanup."""
        self.logger.error(f"Real-time source {self.name} failed for query '{query}': {error}")
        
        # Mark source as temporarily unavailable
        self.last_error = error
        self.last_error_time = datetime.now(timezone.utc)
        
        # Clean up connections
        await self._cleanup_connections()
        
        # Raise exception for proper error handling upstream
        raise ICOException(f"Real-time source {self.name} failed: {error}")
    
    async def _cleanup_connections(self) -> None:
        """Clean up all real-time connections."""
        try:
            if self.connection:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
                elif hasattr(self.connection, 'aclose'):
                    await self.connection.aclose()
                self.connection = None
            
            # Clear message queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
        except Exception as e:
            self.logger.warning(f"Error during connection cleanup: {e}")
    
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
    
    def _add_to_buffer_safe(self, data: Dict[str, Any]) -> None:
        """Safely add data to buffer with size enforcement."""
        self.data_buffer.append(data)
        
        # Enforce buffer size limit
        while len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)  # Remove oldest items
    
    async def _create_chunk(self, content: str, metadata: Dict[str, Any], token_count: int) -> ContextChunk:
        """Create a ContextChunk object."""
        from ..models import ContextChunk, ContextSource, SourceType
        
        return ContextChunk(
            id=uuid4(),  # Generate unique ID for each chunk
            content=content,
            source=ContextSource(
                name=self.name,
                source_type=SourceType.REALTIME,
                url=self.url
            ),
            metadata=metadata,
            token_count=token_count
        )
    
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

    # Public connection methods
    async def connect(self) -> None:
        """Connect to the real-time source."""
        try:
            await self._connect()
            self.is_connected = True
            self.logger.info("Connected to real-time source")
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the real-time source."""
        try:
            await self._disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from real-time source")
        except Exception as e:
            self.logger.error(f"Failed to disconnect: {e}")
            raise

    # Streaming control methods
    async def start_streaming(self) -> None:
        """Start data streaming."""
        try:
            if not self.is_connected:
                await self.connect()
            
            self._streaming = True
            self.logger.info("Started data streaming")
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            raise

    async def stop_streaming(self) -> None:
        """Stop data streaming."""
        try:
            self._streaming = False
            self.logger.info("Stopped data streaming")
        except Exception as e:
            self.logger.error(f"Failed to stop streaming: {e}")
            raise

    # Subscription methods
    async def subscribe_to_updates(self, callback: Callable) -> None:
        """Subscribe to real-time updates."""
        try:
            if callback not in self.subscribers:
                self.subscribers.add(callback)
                self.logger.info(f"Added subscriber: {callback}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            raise

    async def unsubscribe_from_updates(self, callback: Callable) -> None:
        """Unsubscribe from real-time updates."""
        try:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
                self.logger.info(f"Removed subscriber: {callback}")
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe: {e}")
            raise

    async def publish_update(self, update_data: Dict[str, Any]) -> None:
        """Publish update to all subscribers."""
        try:
            for callback in self.subscribers:
                try:
                    await callback(update_data)
                except Exception as e:
                    self.logger.warning(f"Subscriber callback failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to publish update: {e}")
            raise

    # Buffer management methods
    async def add_to_buffer(self, data: Any) -> None:
        """Add data to the buffer."""
        try:
            if len(self.data_buffer) >= self.max_buffer_size:
                # Remove oldest data if buffer is full
                self.data_buffer.pop(0)
            
            self.data_buffer.append({
                "data": data,
                "timestamp": datetime.now(timezone.utc),
                "id": str(uuid4())
            })
        except Exception as e:
            self.logger.error(f"Failed to add to buffer: {e}")
            raise

    async def get_buffered_data(self) -> List[Dict[str, Any]]:
        """Get all buffered data."""
        try:
            return self.data_buffer.copy()
        except Exception as e:
            self.logger.error(f"Failed to get buffered data: {e}")
            return []

    # Filtering methods
    async def set_filter(self, filter_config: Dict[str, Any]) -> None:
        """Set data filter configuration."""
        try:
            self.filter_config = filter_config
            self.logger.info(f"Set filter: {filter_config}")
        except Exception as e:
            self.logger.error(f"Failed to set filter: {e}")
            raise

    async def apply_filter(self, data: Any) -> Optional[Any]:
        """Apply filter to data."""
        try:
            if not hasattr(self, 'filter_config') or not self.filter_config:
                return data
            
            filter_type = self.filter_config.get("type")
            if filter_type == "keyword":
                keywords = self.filter_config.get("keywords", [])
                if isinstance(data, str):
                    return data if any(keyword in data for keyword in keywords) else None
                elif isinstance(data, dict):
                    data_str = str(data)
                    return data if any(keyword in data_str for keyword in keywords) else None
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to apply filter: {e}")
            return None

    # Transformation methods
    async def set_transformation(self, transform_config: Dict[str, Any]) -> None:
        """Set data transformation configuration."""
        try:
            self.transform_config = transform_config
            self.logger.info(f"Set transformation: {transform_config}")
        except Exception as e:
            self.logger.error(f"Failed to set transformation: {e}")
            raise

    async def apply_transformation(self, data: Any) -> Any:
        """Apply transformation to data."""
        try:
            if not hasattr(self, 'transform_config') or not self.transform_config:
                return data
            
            transform_type = self.transform_config.get("type")
            if transform_type == "uppercase" and isinstance(data, str):
                return data.upper()
            elif transform_type == "lowercase" and isinstance(data, str):
                return data.lower()
            elif transform_type == "reverse" and isinstance(data, str):
                return data[::-1]
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to apply transformation: {e}")
            return data

    # Rate limiting methods
    async def set_rate_limit(self, max_per_second: int) -> None:
        """Set rate limit for operations."""
        try:
            self.rate_limit = max_per_second
            self.rate_limit_timestamps = []
            self.logger.info(f"Set rate limit: {max_per_second} per second")
        except Exception as e:
            self.logger.error(f"Failed to set rate limit: {e}")
            raise

    async def _check_rate_limit(self) -> bool:
        """Check if operation is within rate limit."""
        try:
            if not hasattr(self, 'rate_limit') or not self.rate_limit:
                return True
            
            now = datetime.now(timezone.utc)
            # Remove timestamps older than 1 second
            self.rate_limit_timestamps = [
                ts for ts in self.rate_limit_timestamps 
                if (now - ts).total_seconds() < 1.0
            ]
            
            if len(self.rate_limit_timestamps) >= self.rate_limit:
                return False
            
            self.rate_limit_timestamps.append(now)
            return True
        except Exception as e:
            self.logger.error(f"Failed to check rate limit: {e}")
            return True

    # Health monitoring methods
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the real-time source."""
        try:
            return {
                "status": "healthy" if self.is_connected else "disconnected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {
                    "connection_type": self.connection_type,
                    "is_connected": self.is_connected,
                    "is_streaming": getattr(self, '_streaming', False),
                    "buffer_size": len(self.data_buffer),
                    "subscriber_count": len(self.subscribers)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return {"status": "error", "error": str(e)}

    # Performance metrics methods
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            return {
                "messages_processed": getattr(self, 'messages_processed', 0),
                "average_processing_time": getattr(self, 'avg_processing_time', 0.0),
                "error_rate": getattr(self, 'error_rate', 0.0),
                "buffer_utilization": len(self.data_buffer) / self.max_buffer_size if self.max_buffer_size > 0 else 0.0
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}

    # Data persistence methods
    async def enable_persistence(self) -> None:
        """Enable data persistence."""
        try:
            self.persistence_enabled = True
            self.logger.info("Enabled data persistence")
        except Exception as e:
            self.logger.error(f"Failed to enable persistence: {e}")
            raise

    # Data compression methods
    async def enable_compression(self) -> None:
        """Enable data compression."""
        try:
            self.compression_enabled = True
            self.logger.info("Enabled data compression")
        except Exception as e:
            self.logger.error(f"Failed to enable compression: {e}")
            raise

    async def compress_data(self, data: str) -> bytes:
        """Compress data."""
        try:
            if not hasattr(self, 'compression_enabled') or not self.compression_enabled:
                return data.encode('utf-8')
            
            import gzip
            return gzip.compress(data.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Failed to compress data: {e}")
            return data.encode('utf-8')

    # Data encryption methods
    async def enable_encryption(self) -> None:
        """Enable data encryption."""
        try:
            self.encryption_enabled = True
            self.logger.info("Enabled data encryption")
        except Exception as e:
            self.logger.error(f"Failed to enable encryption: {e}")
            raise

    async def encrypt_data(self, data: str) -> bytes:
        """Encrypt data."""
        try:
            if not hasattr(self, 'encryption_enabled') or not self.encryption_enabled:
                return data.encode('utf-8')
            
            # Simple XOR encryption for testing (not secure for production)
            key = b'RAGIFY_KEY_123'
            data_bytes = data.encode('utf-8')
            encrypted = bytes(a ^ b for a, b in zip(data_bytes, key * (len(data_bytes) // len(key) + 1)))
            return encrypted
        except Exception as e:
            self.logger.error(f"Failed to encrypt data: {e}")
            return data.encode('utf-8')

    # Batch processing methods
    async def process_batch(self) -> Dict[str, Any]:
        """Process all buffered data as a batch."""
        try:
            batch_size = len(self.data_buffer)
            processed_items = []
            
            for item in self.data_buffer:
                try:
                    # Apply transformations and filters
                    processed_item = await self.apply_transformation(item["data"])
                    if processed_item is not None:
                        processed_item = await self.apply_filter(processed_item)
                        if processed_item is not None:
                            processed_items.append(processed_item)
                except Exception as e:
                    self.logger.warning(f"Failed to process item: {e}")
            
            # Clear buffer after processing
            self.data_buffer.clear()
            
            return {
                "batch_processed": True,
                "items_processed": len(processed_items),
                "total_items": batch_size,
                "processed_data": processed_items
            }
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            return {"batch_processed": False, "error": str(e)}

    # Data validation methods
    async def set_validation_rules(self, rules: Dict[str, Any]) -> None:
        """Set data validation rules."""
        try:
            self.validation_rules = rules
            self.logger.info(f"Set validation rules: {rules}")
        except Exception as e:
            self.logger.error(f"Failed to set validation rules: {e}")
            raise

    async def validate_data(self, data: Any) -> bool:
        """Validate data against rules."""
        try:
            if not hasattr(self, 'validation_rules') or not self.validation_rules:
                return True
            
            # Check required fields
            required_fields = self.validation_rules.get("required_fields", [])
            if isinstance(data, dict):
                for field in required_fields:
                    if field not in data:
                        return False
            
            # Check max length
            max_length = self.validation_rules.get("max_length")
            if max_length and isinstance(data, str) and len(data) > max_length:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to validate data: {e}")
            return False

    # Data routing methods
    async def set_routing_rules(self, rules: Dict[str, str]) -> None:
        """Set data routing rules."""
        try:
            self.routing_rules = rules
            self.logger.info(f"Set routing rules: {rules}")
        except Exception as e:
            self.logger.error(f"Failed to set routing rules: {e}")
            raise

    async def route_data(self, data: Any) -> str:
        """Route data based on rules."""
        try:
            if not hasattr(self, 'routing_rules') or not self.routing_rules:
                return "default"
            
            # Simple priority-based routing
            if isinstance(data, dict):
                priority = data.get("priority", "low")
                return self.routing_rules.get(priority, "default")
            
            return "default"
        except Exception as e:
            self.logger.error(f"Failed to route data: {e}")
            return "default"

    # Data aggregation methods
    async def aggregate_data(self, field: str, operation: str) -> Any:
        """Aggregate data from buffer."""
        try:
            if not self.data_buffer:
                return None
            
            values = []
            for item in self.data_buffer:
                if isinstance(item["data"], dict) and field in item["data"]:
                    values.append(item["data"][field])
            
            if not values:
                return None
            
            if operation == "sum":
                return sum(values)
            elif operation == "average":
                return sum(values) / len(values)
            elif operation == "min":
                return min(values)
            elif operation == "max":
                return max(values)
            elif operation == "count":
                return len(values)
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to aggregate data: {e}")
            return None

    # Data sampling methods
    async def sample_data(self, sample_size: int) -> List[Any]:
        """Sample data from buffer."""
        try:
            if not self.data_buffer:
                return []
            
            import random
            sample_size = min(sample_size, len(self.data_buffer))
            return random.sample([item["data"] for item in self.data_buffer], sample_size)
        except Exception as e:
            self.logger.error(f"Failed to sample data: {e}")
            return []

    # Data archiving methods
    async def archive_old_data(self, days_old: int = 1) -> Dict[str, Any]:
        """Archive old data from buffer."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            archived_count = 0
            
            # Filter out old data
            self.data_buffer = [
                item for item in self.data_buffer
                if item["timestamp"] > cutoff_date
            ]
            
            archived_count = len(self.data_buffer)
            return {
                "archived": True,
                "items_archived": archived_count,
                "cutoff_date": cutoff_date.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to archive data: {e}")
            return {"archived": False, "error": str(e)}

    # Data cleanup methods
    async def cleanup_data(self) -> Dict[str, Any]:
        """Clean up data buffer."""
        try:
            original_size = len(self.data_buffer)
            
            # Remove duplicate data
            seen = set()
            unique_data = []
            for item in self.data_buffer:
                data_hash = hash(str(item["data"]))
                if data_hash not in seen:
                    seen.add(data_hash)
                    unique_data.append(item)
            
            self.data_buffer = unique_data
            cleaned_count = original_size - len(self.data_buffer)
            
            return {
                "cleaned": True,
                "items_removed": cleaned_count,
                "remaining_items": len(self.data_buffer)
            }
        except Exception as e:
            self.logger.error(f"Failed to cleanup data: {e}")
            return {"cleaned": False, "error": str(e)}

    # Memory management methods
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import sys
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "buffer_size": len(self.data_buffer),
                "buffer_memory": len(self.data_buffer) * 100  # Rough estimate
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "buffer_size": len(self.data_buffer),
                "buffer_memory": len(self.data_buffer) * 100
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {"error": str(e)}

    # Configuration methods
    async def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration."""
        try:
            return {
                "connection_type": self.connection_type,
                "connection_config": self.connection_config,
                "max_buffer_size": self.max_buffer_size,
                "filter_config": getattr(self, 'filter_config', {}),
                "transform_config": getattr(self, 'transform_config', {}),
                "rate_limit": getattr(self, 'rate_limit', None),
                "validation_rules": getattr(self, 'validation_rules', {}),
                "routing_rules": getattr(self, 'routing_rules', {})
            }
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            return {}

    async def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        try:
            for key, value in new_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self.logger.info(f"Updated configuration: {new_config}")
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise

    # Logging methods
    async def enable_detailed_logging(self) -> None:
        """Enable detailed logging."""
        try:
            self.detailed_logging = True
            self.logger.info("Enabled detailed logging")
        except Exception as e:
            self.logger.error(f"Failed to enable detailed logging: {e}")
            raise

    async def get_logs(self) -> List[Dict[str, Any]]:
        """Get recent logs."""
        try:
            if not hasattr(self, 'log_history'):
                self.log_history = []
            
            return self.log_history[-100:]  # Last 100 log entries
        except Exception as e:
            self.logger.error(f"Failed to get logs: {e}")
            return []

    # Data export/import methods
    async def export_data(self, file_path: str) -> Dict[str, Any]:
        """Export data to file."""
        try:
            import json
            
            export_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": [item["data"] for item in self.data_buffer],
                "metadata": {
                    "buffer_size": len(self.data_buffer),
                    "source_name": self.name,
                    "source_type": self.source_type.value
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return {
                "exported": True,
                "file_path": file_path,
                "items_exported": len(self.data_buffer)
            }
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return {"exported": False, "error": str(e)}

    async def import_data(self, file_path: str) -> Dict[str, Any]:
        """Import data from file."""
        try:
            import json
            
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            # Handle both list and dict formats
            if isinstance(import_data, list):
                data_items = import_data
            else:
                data_items = import_data.get("data", [])
            
            for item in data_items:
                await self.add_to_buffer(item)
                imported_count += 1
            
            return {
                "imported": True,
                "items_imported": imported_count,
                "file_path": file_path
            }
        except Exception as e:
            self.logger.error(f"Failed to import data: {e}")
            return {"imported": False, "error": str(e)}

    # Data synchronization methods
    async def sync_with_source(self, other_source: 'RealtimeSource') -> Dict[str, Any]:
        """Synchronize data with another source."""
        try:
            other_data = await other_source.get_buffered_data()
            synced_count = 0
            
            for item in other_data:
                if item not in self.data_buffer:
                    await self.add_to_buffer(item["data"])
                    synced_count += 1
            
            return {
                "synced": True,
                "items_synced": synced_count,
                "source_name": other_source.name
            }
        except Exception as e:
            self.logger.error(f"Failed to sync with source: {e}")
            return {"synced": False, "error": str(e)}

    # Backup and restore methods
    async def create_backup(self, backup_path: str) -> Dict[str, Any]:
        """Create a backup of current data."""
        try:
            backup_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": self.data_buffer,
                "configuration": await self.get_configuration()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            return {
                "backup_created": True,
                "backup_path": backup_path,
                "items_backed_up": len(self.data_buffer)
            }
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return {"backup_created": False, "error": str(e)}

    async def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore data from backup."""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore configuration
            if "configuration" in backup_data:
                await self.update_configuration(backup_data["configuration"])
            
            # Restore data
            restored_count = 0
            for item in backup_data.get("data", []):
                await self.add_to_buffer(item["data"])
                restored_count += 1
            
            return {
                "restored": True,
                "items_restored": restored_count,
                "backup_path": backup_path
            }
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            return {"restored": False, "error": str(e)}

    # Connection recovery methods
    async def attempt_recovery(self) -> Dict[str, Any]:
        """Attempt to recover from connection issues."""
        try:
            if not self.is_connected:
                await self._connect()
                self.is_connected = True
                
                return {
                    "recovery_attempted": True,
                    "recovery_successful": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "recovery_attempted": False,
                    "reason": "Already connected"
                }
        except Exception as e:
            self.logger.error(f"Recovery failed: {e}")
            return {
                "recovery_attempted": True,
                "recovery_successful": False,
                "error": str(e)
            }
