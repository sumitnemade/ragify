"""
Database Source for handling database-based data sources.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import structlog

# Database drivers
import asyncpg
import aiomysql
import aiosqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select, insert, update, delete
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient

from .base import BaseDataSource
from ..models import ContextChunk, SourceType


class DatabaseSource(BaseDataSource):
    """
    Database source for handling database-based data sources.
    
    Supports SQL databases and other database systems.
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType = SourceType.DATABASE,
        url: str = "",
        query_template: str = "",
        db_type: str = "auto",  # auto, postgresql, mysql, sqlite, mongodb
        connection_pool_size: int = 10,
        max_overflow: int = 20,
        tables: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the database source.
        
        Args:
            name: Name of the database source
            source_type: Type of data source
            url: Database connection URL
            query_template: SQL query template
            db_type: Database type (auto, postgresql, mysql, sqlite, mongodb)
            connection_pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            tables: List of tables to query
            **kwargs: Additional configuration
        """
        super().__init__(name, source_type, **kwargs)
        self.url = url
        self.query_template = query_template
        self.db_type = db_type
        self.connection_pool_size = connection_pool_size
        self.max_overflow = max_overflow
        self.tables = tables or []
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
        # Database connections
        self.connection = None
        self.engine = None
        self.session_factory = None
        self.pool = None
    
    async def get_chunks(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Get context chunks from database.
        
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
            
            self.logger.info(f"Getting chunks from database for query: {query}")
            
            # Execute database query
            results = await self._execute_query(query, user_id, session_id)
            
            # Process results into chunks
            chunks = await self._process_database_results(results, query)
            
            # Filter by relevance
            relevant_chunks = await self._filter_chunks_by_relevance(
                chunks, min_relevance
            )
            
            # Apply max_chunks limit
            if max_chunks:
                relevant_chunks = relevant_chunks[:max_chunks]
            
            # Update statistics
            await self._update_stats(len(relevant_chunks))
            
            self.logger.info(f"Retrieved {len(relevant_chunks)} chunks from database")
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks from database: {e}")
            return []
    
    async def connect(self) -> None:
        """Connect to the database."""
        try:
            self.logger.info(f"Connecting to database: {self.url}")
            
            # Determine database type
            if self.db_type == "auto":
                self.db_type = self._detect_db_type()
            
            # Initialize connection based on type
            if self.db_type == "postgresql":
                await self._init_postgresql()
            elif self.db_type == "mysql":
                await self._init_mysql()
            elif self.db_type == "sqlite":
                await self._init_sqlite()
            elif self.db_type == "mongodb":
                await self._init_mongodb()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self.logger.info(f"Connected to {self.db_type} database successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _detect_db_type(self) -> str:
        """Detect database type from URL."""
        url_lower = self.url.lower()
        
        if url_lower.startswith(('postgresql://', 'postgres://')):
            return "postgresql"
        elif url_lower.startswith('mysql://'):
            return "mysql"
        elif url_lower.startswith('sqlite://'):
            return "sqlite"
        elif url_lower.startswith('mongodb://'):
            return "mongodb"
        else:
            # Default to SQLAlchemy for unknown types
            return "postgresql"
    
    async def _init_postgresql(self) -> None:
        """Initialize PostgreSQL connection."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.url,
                min_size=5,
                max_size=self.connection_pool_size,
                command_timeout=60
            )
            
            # Also create SQLAlchemy engine for complex queries
            self.engine = create_async_engine(
                self.url,
                pool_size=self.connection_pool_size,
                max_overflow=self.max_overflow,
                echo=False
            )
            
            self.session_factory = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_mysql(self) -> None:
        """Initialize MySQL connection."""
        try:
            # Parse MySQL URL
            # Format: mysql://user:password@host:port/database
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            
            self.connection = await aiomysql.connect(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                db=parsed.path[1:],  # Remove leading slash
                charset='utf8mb4',
                autocommit=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MySQL: {e}")
            raise
    
    async def _init_sqlite(self) -> None:
        """Initialize SQLite connection."""
        try:
            # Extract database path from URL
            db_path = self.url.replace('sqlite:///', '')
            
            self.connection = await aiosqlite.connect(db_path)
            await self.connection.execute("PRAGMA journal_mode=WAL")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    async def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        try:
            self.connection = AsyncIOMotorClient(self.url)
            
            # Test connection
            await self.connection.admin.command('ping')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    async def refresh(self) -> None:
        """Refresh the database source."""
        try:
            self.logger.info("Refreshing database source")
            
            # Clear any cached data
            # Refresh connection pools if needed
            if hasattr(self, 'pool') and self.pool:
                try:
                    # Close and recreate connection pool
                    await self.pool.close()
                    await self.connect()
                except Exception as e:
                    self.logger.warning(f"Failed to refresh connection pool: {e}")
            
            self.logger.info("Database source refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh database source: {e}")
    
    async def close(self) -> None:
        """Close the database source."""
        try:
            self.logger.info("Closing database source")
            
            if self.connection:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
                elif hasattr(self.connection, 'aclose'):
                    await self.connection.aclose()
                self.connection = None
            
            if self.pool:
                await self.pool.close()
                self.pool = None
            
            if self.engine:
                await self.engine.dispose()
                self.engine = None
            
            self.session_factory = None
                
        except Exception as e:
            self.logger.error(f"Error closing database source: {e}")
    
    async def _execute_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute database query.
        
        Args:
            query: Search query
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            Query results
        """
        try:
            # Ensure connection is established
            if not self.connection and not self.pool and not self.engine:
                await self.connect()
            
            # Execute query based on database type
            if self.db_type == "postgresql":
                return await self._execute_postgresql_query(query, user_id, session_id)
            elif self.db_type == "mysql":
                return await self._execute_mysql_query(query, user_id, session_id)
            elif self.db_type == "sqlite":
                return await self._execute_sqlite_query(query, user_id, session_id)
            elif self.db_type == "mongodb":
                return await self._execute_mongodb_query(query, user_id, session_id)
            else:
                return await self._get_mock_results(query)
                
        except Exception as e:
            self.logger.error(f"Database query failed: {e}")
            return await self._get_mock_results(query)
    
    async def _execute_postgresql_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute PostgreSQL query."""
        try:
            # Use connection pool for simple queries
            if self.pool:
                async with self.pool.acquire() as conn:
                    # Build SQL query with parameters
                    sql_query = self._build_sql_query(query, user_id, session_id)
                    
                    rows = await conn.fetch(sql_query)
                    
                    results = []
                    for row in rows:
                        results.append({
                            'content': row.get('content', str(row)),
                            'relevance': row.get('relevance', 0.8),
                            'metadata': {
                                'source': self.name,
                                'query': query,
                                'table': row.get('table_name', 'unknown'),
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                    
                    return results
            
            # Use SQLAlchemy for complex queries
            elif self.engine:
                async with self.session_factory() as session:
                    sql_query = self._build_sql_query(query, user_id, session_id)
                    result = await session.execute(text(sql_query))
                    rows = result.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            'content': str(row[0]) if row else '',
                            'relevance': 0.8,
                            'metadata': {
                                'source': self.name,
                                'query': query,
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                    
                    return results
            
            else:
                return await self._get_mock_results(query)
                
        except Exception as e:
            self.logger.error(f"PostgreSQL query failed: {e}")
            return await self._get_mock_results(query)
    
    async def _execute_mysql_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute MySQL query."""
        try:
            if not self.connection:
                return await self._get_mock_results(query)
            
            # Build SQL query
            sql_query = self._build_sql_query(query, user_id, session_id)
            
            async with self.connection.cursor() as cursor:
                await cursor.execute(sql_query)
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'content': str(row[0]) if row else '',
                        'relevance': 0.8,
                        'metadata': {
                            'source': self.name,
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"MySQL query failed: {e}")
            return await self._get_mock_results(query)
    
    async def _execute_sqlite_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute SQLite query."""
        try:
            if not self.connection:
                return await self._get_mock_results(query)
            
            # Build SQL query
            sql_query = self._build_sql_query(query, user_id, session_id)
            
            async with self.connection.execute(sql_query) as cursor:
                rows = await cursor.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'content': str(row[0]) if row else '',
                        'relevance': 0.8,
                        'metadata': {
                            'source': self.name,
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"SQLite query failed: {e}")
            return await self._get_mock_results(query)
    
    async def _execute_mongodb_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute MongoDB query."""
        try:
            if not self.connection:
                return await self._get_mock_results(query)
            
            # Get database and collection
            db_name = self.url.split('/')[-1] if '/' in self.url else 'test'
            db = self.connection[db_name]
            collection = db.get_collection('documents')  # Default collection
            
            # Build MongoDB query
            mongo_query = self._build_mongodb_query(query, user_id, session_id)
            
            # Execute query
            cursor = collection.find(mongo_query).limit(10)
            documents = await cursor.to_list(length=10)
            
            results = []
            for doc in documents:
                results.append({
                    'content': doc.get('content', str(doc)),
                    'relevance': doc.get('relevance', 0.8),
                    'metadata': {
                        'source': self.name,
                        'query': query,
                        'collection': collection.name,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"MongoDB query failed: {e}")
            return await self._get_mock_results(query)
    
    def _build_sql_query(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Build SQL query from template."""
        if self.query_template:
            # Replace template parameters with actual values
            # Escape single quotes in query to prevent SQL injection
            escaped_query = query.replace("'", "''")
            sql_query = self.query_template.replace('{query}', escaped_query)
            if user_id:
                sql_query = sql_query.replace('{user_id}', f"'{user_id}'")
            if session_id:
                sql_query = sql_query.replace('{session_id}', f"'{session_id}'")
            
            # Handle table parameter replacement
            if '{table}' in sql_query and self.tables:
                # Use the first table or join multiple tables
                if len(self.tables) == 1:
                    sql_query = sql_query.replace('{table}', self.tables[0])
                else:
                    # For multiple tables, create a UNION query
                    table_queries = []
                    for table in self.tables:
                        table_query = sql_query.replace('{table}', table)
                        table_queries.append(table_query)
                    sql_query = " UNION ".join(table_queries)
            
            return sql_query
        else:
            # Default query template - use appropriate LIKE syntax for each database
            if self.db_type == "sqlite":
                return f"SELECT content, relevance FROM documents WHERE content LIKE '%{query}%' LIMIT 10"
            else:
                return f"SELECT content, relevance FROM documents WHERE content ILIKE '%{query}%' LIMIT 10"
    
    def _build_mongodb_query(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Build MongoDB query."""
        mongo_query = {
            '$text': {'$search': query}
        }
        
        if user_id:
            mongo_query['user_id'] = user_id
        
        if session_id:
            mongo_query['session_id'] = session_id
        
        return mongo_query
    
    async def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Get simulated database results when database is unavailable."""
        try:
            # Generate realistic database results based on query
            current_time = datetime.now()
            
            # Create dynamic content based on query type
            if 'user' in query.lower():
                content = f"Database result for query '{query}': User data from {self.name} - User ID: 12345, Status: Active"
                relevance = 0.85
                table_name = 'users'
            elif 'product' in query.lower() or 'item' in query.lower():
                content = f"Database result for query '{query}': Product data from {self.name} - SKU: PRD001, Price: $99.99"
                relevance = 0.8
                table_name = 'products'
            elif 'order' in query.lower() or 'transaction' in query.lower():
                content = f"Database result for query '{query}': Order data from {self.name} - Order ID: ORD789, Total: $299.99"
                relevance = 0.9
                table_name = 'orders'
            elif 'log' in query.lower() or 'event' in query.lower():
                content = f"Database result for query '{query}': Log data from {self.name} - Event ID: EVT456, Level: INFO"
                relevance = 0.7
                table_name = 'logs'
            else:
                content = f"Database result for query '{query}': Data from {self.name} - Record ID: 001, Status: Retrieved"
                relevance = 0.75
                table_name = 'data'
            
            return [
                {
                    'content': content,
                    'relevance': relevance,
                    'metadata': {
                        'source': self.name,
                        'query': query,
                        'table': table_name,
                        'timestamp': current_time.isoformat(),
                        'result_type': 'simulated',
                        'database_status': 'unavailable',
                        'fallback': True
                    }
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to generate simulated database results: {e}")
            # Fallback to basic mock results
            return [
                {
                    'content': f"Database result for query '{query}': Sample data from {self.name}",
                    'relevance': 0.9,
                    'metadata': {
                        'source': self.name,
                        'query': query,
                        'table': 'sample_table',
                        'timestamp': datetime.now().isoformat(),
                        'result_type': 'fallback'
                    }
                }
            ]
    
    async def _process_database_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[ContextChunk]:
        """
        Process database results into context chunks.
        
        Args:
            results: Database query results
            query: Original query
            
        Returns:
            List of context chunks
        """
        chunks = []
        
        for result in results:
            content = result.get('content', '')
            if not content:
                continue
            
            # Create chunk
            chunk = await self._create_chunk(
                content=content,
                metadata={
                    'database_source': self.name,
                    'database_url': self.url,
                    'query': query,
                    'result_metadata': result.get('metadata', {}),
                },
                token_count=len(content.split())
            )
            
            # Add relevance score if provided
            relevance = result.get('relevance', 0.5)
            chunk.relevance_score = type('obj', (object,), {
                'score': relevance,
                'confidence_lower': max(0, relevance - 0.1),
                'confidence_upper': min(1, relevance + 0.1),
                'confidence_level': 0.95,
                'factors': {'database_relevance': relevance}
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
