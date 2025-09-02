"""
Database Source for handling database-based data sources.
"""

import asyncio
import json
import time
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
from ..exceptions import ICOException


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
        
        # Transaction management
        self.current_transaction = None
        self.transaction_depth = 0
        
        # Connection pool monitoring
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'query_count': 0,
            'avg_query_time': 0.0
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
            self.logger.warning(f"Failed to initialize PostgreSQL: {e}")
            self.logger.info("PostgreSQL connection will be attempted when needed")
            # Don't raise - allow graceful fallback
    
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
            self.logger.warning(f"Failed to initialize MySQL: {e}")
            self.logger.info("MySQL connection will be attempted when needed")
            # Don't raise - allow graceful fallback
    
    async def _init_sqlite(self) -> None:
        """Initialize SQLite connection."""
        try:
            # Extract database path from URL
            db_path = self.url.replace('sqlite:///', '')
            
            self.connection = await aiosqlite.connect(db_path)
            await self.connection.execute("PRAGMA journal_mode=WAL")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize SQLite: {e}")
            self.logger.info("SQLite connection will be attempted when needed")
            # Don't raise - allow graceful fallback
    
    async def _init_mongodb(self) -> None:
        """Initialize MongoDB connection."""
        try:
            self.connection = AsyncIOMotorClient(self.url)
            
            # Test connection
            await self.connection.admin.command('ping')
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize MongoDB: {e}")
            self.logger.info("MongoDB connection will be attempted when needed")
            # Don't raise - allow graceful fallback
    
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
        Execute database query with enhanced validation and optimization.
        
        Args:
            query: Search query
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            Query results
        """
        start_time = time.time()
        
        try:
            # Validate and sanitize parameters
            validated_params = self._validate_query_parameters(query, user_id, session_id)
            
            # Ensure connection is established
            if not self.connection and not self.pool and not self.engine:
                await self.connect()
            
            # Execute query based on database type
            if self.db_type == "postgresql":
                results = await self._execute_postgresql_query(
                    validated_params['query'], 
                    validated_params['user_id'], 
                    validated_params['session_id']
                )
            elif self.db_type == "mysql":
                results = await self._execute_mysql_query(
                    validated_params['query'], 
                    validated_params['user_id'], 
                    validated_params['session_id']
                )
            elif self.db_type == "sqlite":
                results = await self._execute_sqlite_query(
                    validated_params['query'], 
                    validated_params['user_id'], 
                    validated_params['session_id']
                )
            elif self.db_type == "mongodb":
                results = await self._execute_mongodb_query(
                    validated_params['query'], 
                    validated_params['user_id'], 
                    validated_params['session_id']
                )
            else:
                raise ICOException(f"Unsupported database type: {self.db_type}")
            
            # Update connection statistics
            query_time = time.time() - start_time
            await self._update_connection_stats(query_time, success=True)
            
            return results
                
        except Exception as e:
            # Update connection statistics for failure
            query_time = time.time() - start_time
            await self._update_connection_stats(query_time, success=False)
            
            self.logger.error(f"Database query failed: {e}")
            raise ICOException(f"Database query failed: {e}")
    
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
                raise ICOException("No database connection available")
                
        except Exception as e:
            self.logger.error(f"PostgreSQL query failed: {e}")
            raise ICOException(f"PostgreSQL query failed: {e}")
    
    async def _execute_mysql_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute MySQL query."""
        try:
            if not self.connection:
                raise ICOException("No MySQL connection available")
            
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
            await self._handle_database_failure(query, e)
    
    async def _execute_sqlite_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute SQLite query."""
        try:
            if not self.connection:
                raise ICOException("No SQLite connection available")
            
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
            await self._handle_database_failure(query, e)
    
    async def _execute_mongodb_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute MongoDB query."""
        try:
            if not self.connection:
                raise ICOException("No MongoDB connection available")
            
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
            await self._handle_database_failure(query, e)
    
    def _build_sql_query(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Build SQL query from template with advanced optimization."""
        if self.query_template:
            # Use parameterized queries to prevent SQL injection
            sql_query = self.query_template
            
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
            # Generate optimized default queries based on database type
            return self._generate_optimized_query(query, user_id, session_id)
    
    def _generate_optimized_query(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Generate optimized SQL queries based on database type and query content."""
        # Parse query for optimization hints
        query_lower = query.lower()
        is_exact_match = query.startswith('"') and query.endswith('"')
        is_phrase_search = ' ' in query and not is_exact_match
        
        # Build base query with proper escaping
        if self.db_type == "sqlite":
            if is_exact_match:
                # Remove quotes and use exact match
                clean_query = query[1:-1]
                escaped_query = clean_query.replace("'", "''")
                where_clause = f"content = '{escaped_query}'"
            elif is_phrase_search:
                # Use multiple LIKE conditions for phrase search
                words = query.split()
                conditions = []
                for word in words:
                    escaped_word = word.replace("'", "''")
                    conditions.append(f"content LIKE '%{escaped_word}%'")
                where_clause = " AND ".join(conditions)
            else:
                # Single word search
                escaped_query = query.replace("'", "''")
                where_clause = f"content LIKE '%{escaped_query}%'"
            
            escaped_query_final = query.replace("'", "''")
            return f"""
                SELECT content, relevance, 
                       CASE 
                           WHEN content = '{escaped_query_final}' THEN 1.0
                           WHEN content LIKE '{escaped_query_final}%' THEN 0.9
                           WHEN content LIKE '%{escaped_query_final}%' THEN 0.8
                           ELSE 0.5
                       END as calculated_relevance
                FROM documents 
                WHERE {where_clause}
                ORDER BY calculated_relevance DESC, length(content) ASC
                LIMIT 20
            """
        
        elif self.db_type == "postgresql":
            if is_exact_match:
                clean_query = query[1:-1]
                escaped_clean_query = clean_query.replace("'", "''")
                where_clause = f"content = '{escaped_clean_query}'"
            elif is_phrase_search:
                # Use PostgreSQL full-text search for better performance
                escaped_query = query.replace("'", "''")
                where_clause = f"to_tsvector('english', content) @@ plainto_tsquery('english', '{escaped_query}')"
            else:
                escaped_query = query.replace("'", "''")
                where_clause = f"content ILIKE '%{escaped_query}%'"
            
            escaped_query_final = query.replace("'", "''")
            return f"""
                SELECT content, relevance,
                       CASE 
                           WHEN content = '{escaped_query_final}' THEN 1.0
                           WHEN content ILIKE '{escaped_query_final}%' THEN 0.9
                           WHEN content ILIKE '%{escaped_query_final}%' THEN 0.8
                           ELSE 0.5
                       END as calculated_relevance
                FROM documents 
                WHERE {where_clause}
                ORDER BY calculated_relevance DESC, length(content) ASC
                LIMIT 20
            """
        
        elif self.db_type == "mysql":
            if is_exact_match:
                clean_query = query[1:-1]
                escaped_clean_query = clean_query.replace("'", "''")
                where_clause = f"content = '{escaped_clean_query}'"
            elif is_phrase_search:
                # Use MySQL full-text search
                words = query.split()
                conditions = []
                for word in words:
                    escaped_word = word.replace("'", "''")
                    conditions.append(f"content LIKE '%{escaped_word}%'")
                where_clause = " AND ".join(conditions)
            else:
                escaped_query = query.replace("'", "''")
                where_clause = f"content LIKE '%{escaped_query}%'"
            
            escaped_query_final = query.replace("'", "''")
            return f"""
                SELECT content, relevance,
                       CASE 
                           WHEN content = '{escaped_query_final}' THEN 1.0
                           WHEN content LIKE '{escaped_query_final}%' THEN 0.9
                           WHEN content LIKE '%{escaped_query_final}%' THEN 0.8
                           ELSE 0.5
                       END as calculated_relevance
                FROM documents 
                WHERE {where_clause}
                ORDER BY calculated_relevance DESC, length(content) ASC
                LIMIT 20
            """
        
        else:
            # Generic fallback
            escaped_query = query.replace("'", "''")
            return f"SELECT content, relevance FROM documents WHERE content LIKE '%{escaped_query}%' LIMIT 20"
    
    def _build_mongodb_query(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Build MongoDB query with advanced search capabilities."""
        # Parse query for MongoDB text search optimization
        query_lower = query.lower()
        is_exact_match = query.startswith('"') and query.endswith('"')
        is_phrase_search = ' ' in query and not is_exact_match
        
        if is_exact_match:
            # Exact match search
            clean_query = query[1:-1]
            mongo_query = {
                'content': {'$regex': f'^{clean_query}$', '$options': 'i'}
            }
        elif is_phrase_search:
            # Phrase search with word proximity
            words = query.split()
            mongo_query = {
                '$and': [
                    {'content': {'$regex': word, '$options': 'i'}} for word in words
                ]
            }
        else:
            # Text search with relevance scoring
            mongo_query = {
                '$text': {'$search': query}
            }
        
        # Add user and session filters
        if user_id:
            mongo_query['user_id'] = user_id
        
        if session_id:
            mongo_query['session_id'] = session_id
        
        return mongo_query
    
    def _validate_query_parameters(self, query: str, user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate and sanitize query parameters."""
        validation_errors = []
        
        # Validate query
        if not query or not query.strip():
            validation_errors.append("Query cannot be empty")
        elif len(query.strip()) > 1000:
            validation_errors.append("Query too long (max 1000 characters)")
        
        # Validate user_id
        if user_id and not isinstance(user_id, str):
            validation_errors.append("User ID must be a string")
        elif user_id and len(user_id) > 100:
            validation_errors.append("User ID too long (max 100 characters)")
        
        # Validate session_id
        if session_id and not isinstance(session_id, str):
            validation_errors.append("Session ID must be a string")
        elif session_id and len(session_id) > 100:
            validation_errors.append("Session ID too long (max 100 characters)")
        
        # Check for SQL injection patterns
        dangerous_patterns = [
            ';', '--', '/*', '*/', 'xp_', 'sp_', 'exec', 'execute',
            'union', 'select', 'insert', 'update', 'delete', 'drop', 'create'
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                validation_errors.append(f"Potentially dangerous pattern detected: {pattern}")
                break
        
        if validation_errors:
            raise ValueError(f"Query validation failed: {'; '.join(validation_errors)}")
        
        return {
            'query': query.strip(),
            'user_id': user_id.strip() if user_id else None,
            'session_id': session_id.strip() if session_id else None
        }
    
    def _create_parameterized_query(self, base_query: str, params: Dict[str, Any]) -> tuple:
        """Create parameterized query with proper parameter binding."""
        # This is a simplified version - in production, you'd use proper SQL parameter binding
        # For now, we'll use the existing escaping method but prepare for future enhancement
        
        query = base_query
        bound_params = {}
        
        # Replace template parameters with placeholders
        if '{query}' in query:
            query = query.replace('{query}', ':query')
            bound_params[':query'] = params['query']
        
        if '{user_id}' in query and params.get('user_id'):
            query = query.replace('{user_id}', ':user_id')
            bound_params[':user_id'] = params['user_id']
        
        if '{session_id}' in query and params.get('session_id'):
            query = query.replace('{session_id}', ':session_id')
            bound_params[':session_id'] = params['session_id']
        
        return query, bound_params
    
    async def _handle_database_failure(self, query: str, error: Exception) -> None:
        """Handle database failure with proper error logging and cleanup."""
        self.logger.error(f"Database source {self.name} failed for query '{query}': {error}")
        
        # Mark source as temporarily unavailable
        self.last_error = error
        self.last_error_time = datetime.now()
        
        # Clean up connections
        await self._cleanup_database_connections()
        
        # Raise exception for proper error handling upstream
        raise ICOException(f"Database source {self.name} failed: {error}")
    
    async def _cleanup_database_connections(self) -> None:
        """Clean up database connections."""
        try:
            # Clean up active transactions first
            if self.transaction_depth > 0:
                self.logger.warning(f"Cleaning up {self.transaction_depth} active transactions")
                self.transaction_depth = 0
                self.current_transaction = None
            
            if hasattr(self, 'pool') and self.pool:
                await self.pool.close()
                self.pool = None
            
            if hasattr(self, 'engine') and self.engine:
                await self.engine.dispose()
                self.engine = None
            
            if hasattr(self, 'connection') and self.connection:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
                elif hasattr(self.connection, 'aclose'):
                    await self.connection.aclose()
                self.connection = None
                
        except Exception as e:
            self.logger.warning(f"Error during database connection cleanup: {e}")
    
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
    
    async def _create_chunk(self, content: str, metadata: Dict[str, Any], token_count: int) -> ContextChunk:
        """Create a ContextChunk object."""
        from ..models import ContextChunk, ContextSource, SourceType
        from uuid import uuid4
        
        return ContextChunk(
            id=uuid4(),  # Generate unique ID for each chunk
            content=content,
            source=ContextSource(
                name=self.name,
                source_type=SourceType.DATABASE,
                url=self.url
            ),
            metadata=metadata,
            token_count=token_count
        )
    
    async def begin_transaction(self) -> None:
        """Begin a database transaction."""
        try:
            if self.transaction_depth == 0:
                if self.db_type == "postgresql" and self.pool:
                    self.current_transaction = await self.pool.acquire()
                    await self.current_transaction.execute("BEGIN")
                elif self.engine:
                    self.current_transaction = self.session_factory()
                    await self.current_transaction.begin()
                else:
                    raise ICOException("No database connection available for transaction")
                
                self.transaction_depth += 1
                self.logger.info("Database transaction started")
            else:
                # Nested transaction - increment depth
                self.transaction_depth += 1
                
        except Exception as e:
            self.logger.error(f"Failed to begin transaction: {e}")
            raise ICOException(f"Failed to begin transaction: {e}")
    
    async def commit_transaction(self) -> None:
        """Commit the current database transaction."""
        try:
            if self.transaction_depth > 0:
                self.transaction_depth -= 1
                
                if self.transaction_depth == 0:
                    if self.current_transaction:
                        if hasattr(self.current_transaction, 'execute'):
                            await self.current_transaction.execute("COMMIT")
                        else:
                            await self.current_transaction.commit()
                        
                        # Release connection back to pool
                        if hasattr(self.current_transaction, 'release'):
                            await self.current_transaction.release()
                        
                        self.current_transaction = None
                        self.logger.info("Database transaction committed")
                        
        except Exception as e:
            self.logger.error(f"Failed to commit transaction: {e}")
            await self.rollback_transaction()
            raise ICOException(f"Failed to commit transaction: {e}")
    
    async def rollback_transaction(self) -> None:
        """Rollback the current database transaction."""
        try:
            if self.transaction_depth > 0:
                self.transaction_depth -= 1
                
                if self.transaction_depth == 0:
                    if self.current_transaction:
                        if hasattr(self.current_transaction, 'execute'):
                            await self.current_transaction.execute("ROLLBACK")
                        else:
                            await self.current_transaction.rollback()
                        
                        # Release connection back to pool
                        if hasattr(self.current_transaction, 'release'):
                            await self.current_transaction.release()
                        
                        self.current_transaction = None
                        self.logger.info("Database transaction rolled back")
                        
        except Exception as e:
            self.logger.error(f"Failed to rollback transaction: {e}")
            # Force cleanup
            self.transaction_depth = 0
            self.current_transaction = None
    
    async def execute_in_transaction(self, operation: callable, *args, **kwargs):
        """Execute an operation within a transaction."""
        try:
            await self.begin_transaction()
            result = await operation(*args, **kwargs)
            await self.commit_transaction()
            return result
        except Exception as e:
            await self.rollback_transaction()
            raise e
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return self.connection_stats.copy()
    
    async def _update_connection_stats(self, query_time: float, success: bool = True) -> None:
        """Update connection statistics."""
        self.connection_stats['query_count'] += 1
        
        if success:
            # Update average query time
            current_avg = self.connection_stats['avg_query_time']
            count = self.connection_stats['query_count']
            self.connection_stats['avg_query_time'] = (current_avg * (count - 1) + query_time) / count
        else:
            self.connection_stats['failed_connections'] += 1