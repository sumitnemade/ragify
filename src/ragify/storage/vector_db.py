"""
Vector Database for context embedding storage and similarity search.
"""

import asyncio
from pathlib import Path
import json
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import structlog
from urllib.parse import urlparse

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    from weaviate.classes.init import Auth
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..models import ContextChunk
from ..exceptions import VectorDBError


class VectorDatabase:
    """
    Vector database for storing and searching context embeddings.
    
    Provides efficient similarity search, indexing, and management
    of vector embeddings for context chunks.
    
    Supports:
    - ChromaDB (local/remote)
    - Pinecone (cloud)
    - Weaviate (local/cloud)
    - FAISS (local)
    """
    
    def __init__(self, vector_db_url: str):
        """
        Initialize the vector database.
        
        Args:
            vector_db_url: Vector database connection URL
                Format: {db_type}://{connection_string}
                Examples:
                - chroma:///path/to/chroma/db
                - chroma://localhost:8000
                - pinecone://api_key:index_name
                - weaviate://localhost:8080
                - faiss:///path/to/faiss/index
        """
        self.vector_db_url = vector_db_url
        self.logger = structlog.get_logger(__name__)
        
        # Parse URL to determine database type
        parsed_url = urlparse(vector_db_url)
        self.db_type = parsed_url.scheme.lower()
        self.connection_string = parsed_url.netloc + parsed_url.path
        
        # Vector database client
        self.vector_client = None
        self.collection = None
        self.index = None
        
        # Database configuration
        self.config = {
            'dimension': 384,  # Default embedding dimension
            'metric': 'cosine',  # Distance metric
            'index_type': 'ivf',  # Index type
            'nlist': 100,  # Number of clusters for IVF
        }
        
        # Performance optimization
        self.connection_pool = {}
        self.max_connections = 10
        self.connection_timeout = 30.0
        self.search_cache = {}
        self.search_cache_size = 1000
        self.search_cache_ttl = 1800  # 30 minutes
        
        # Statistics
        self.stats = {
            'total_vectors': 0,
            'indexed_vectors': 0,
            'searches_performed': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'connection_pool_hits': 0,
            'connection_pool_misses': 0,
        }
        
        # Validate database type
        if self.db_type not in ['chroma', 'pinecone', 'weaviate', 'faiss', 'memory']:
            raise VectorDBError("initialization", f"Unsupported database type: {self.db_type}")
        
        # Check availability
        self._check_availability()
    
    def _check_availability(self) -> None:
        """Check if the required database client is available."""
        if self.db_type == "chroma":
            try:
                import chromadb
            except ImportError:
                raise VectorDBError("initialization", "ChromaDB not available. Install with: pip install chromadb")
        elif self.db_type == "faiss":
            try:
                import faiss
            except ImportError:
                raise VectorDBError("initialization", "FAISS not available. Install with: pip install faiss-cpu")
        elif self.db_type == "pinecone":
            try:
                import pinecone
            except ImportError:
                raise VectorDBError("initialization", "Pinecone not available. Install with: pip install pinecone-client")
        elif self.db_type == "weaviate":
            try:
                import weaviate
            except ImportError:
                raise VectorDBError("initialization", "Weaviate not available. Install with: pip install weaviate-client")
        elif self.db_type == "memory":
            # Memory database is always available
            pass
        else:
            raise VectorDBError("initialization", f"Unsupported database type: {self.db_type}")
    
    async def initialize(self) -> None:
        """Initialize the vector database (alias for connect)."""
        await self.connect()
    
    async def connect(self) -> None:
        """Connect to the vector database."""
        try:
            self.logger.info(f"Connecting to {self.db_type} database")
            
            if self.db_type == 'chroma':
                await self._init_chroma_client()
            elif self.db_type == 'pinecone':
                await self._init_pinecone_client()
            elif self.db_type == 'weaviate':
                await self._init_weaviate_client()
            elif self.db_type == 'faiss':
                await self._init_faiss_client()
            elif self.db_type == 'memory':
                await self._init_memory_client()
            
            self.logger.info(f"Successfully connected to {self.db_type} database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.db_type} database: {e}")
            raise VectorDBError("connection", str(e))
    
    async def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> List[str]:
        """Add embeddings to the vector database (alias for store_embeddings)."""
        # Create dummy chunks for compatibility
        from ..models import ContextChunk, ContextSource
        from uuid import uuid4
        
        chunks = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            chunk = ContextChunk(
                id=uuid4(),
                content=meta.get('content', f'Embedding {i}'),
                source=ContextSource(
                    id=uuid4(),
                    name=meta.get('source', 'vector_db'),
                    source_type='vector'
                ),
                metadata=meta
            )
            chunks.append(chunk)
        
        return await self.store_embeddings(chunks, embeddings)
    
    async def store_embeddings(
        self,
        chunks: List[ContextChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Store embeddings for context chunks.
        
        Args:
            chunks: List of context chunks
            embeddings: List of embedding vectors
            
        Returns:
            List of stored vector IDs
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            self.logger.info(f"Storing {len(chunks)} embeddings in {self.db_type}")
            
            # Prepare data for storage
            vector_data = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_data.append({
                    'id': str(chunk.id),
                    'embedding': embedding,
                    'metadata': {
                        'chunk_id': str(chunk.id),
                        'source_name': chunk.source.name,
                        'source_type': chunk.source.source_type.value,
                        'content_length': len(chunk.content),
                        'created_at': chunk.created_at.isoformat(),
                        'content_preview': chunk.content[:200],  # Store preview for debugging
                    }
                })
            
            # Store in vector database
            vector_ids = await self._store_vectors(vector_data)
            
            # Update statistics
            self.stats['total_vectors'] += len(vector_ids)
            
            self.logger.info(f"Successfully stored {len(vector_ids)} embeddings")
            return vector_ids
            
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {e}")
            raise VectorDBError("store", str(e))
    
    async def search(
        self,
        query_embedding: List[float],
        k: int = 10,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors (alias for search_similar)."""
        return await self.search_similar(query_embedding, top_k=k, min_score=0.0)
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            min_score: Minimum similarity score
            filters: Optional filters for search
            
        Returns:
            List of (vector_id, score, metadata) tuples
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            self.logger.info(f"Searching for similar vectors in {self.db_type} (top_k={top_k})")
            
            # Perform similarity search
            results = await self._search_vectors(
                query_embedding, top_k, min_score, filters
            )
            
            # Filter by minimum score
            filtered_results = [
                (vector_id, score, metadata)
                for vector_id, score, metadata in results
                if score >= min_score
            ]
            
            # Update statistics
            search_time = asyncio.get_event_loop().time() - start_time
            self.stats['searches_performed'] += 1
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['searches_performed'] - 1) + search_time) /
                self.stats['searches_performed']
            )
            
            self.logger.info(f"Found {len(filtered_results)} similar vectors")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar vectors: {e}")
            raise VectorDBError("search", str(e))
    
    async def delete_embeddings(self, vector_ids: List[str]) -> None:
        """
        Delete embeddings by vector IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        try:
            self.logger.info(f"Deleting {len(vector_ids)} embeddings from {self.db_type}")
            
            # Delete from vector database
            await self._delete_vectors(vector_ids)
            
            # Update statistics
            self.stats['total_vectors'] -= len(vector_ids)
            
            self.logger.info(f"Successfully deleted {len(vector_ids)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to delete embeddings: {e}")
            raise VectorDBError("delete", str(e))
    
    async def update_embeddings(
        self,
        vector_ids: List[str],
        new_embeddings: List[List[float]],
    ) -> None:
        """
        Update existing embeddings.
        
        Args:
            vector_ids: List of vector IDs to update
            new_embeddings: List of new embedding vectors
        """
        try:
            if len(vector_ids) != len(new_embeddings):
                raise ValueError("Number of vector IDs must match number of embeddings")
            
            self.logger.info(f"Updating {len(vector_ids)} embeddings in {self.db_type}")
            
            # Update in vector database
            await self._update_vectors(vector_ids, new_embeddings)
            
            self.logger.info(f"Successfully updated {len(vector_ids)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to update embeddings: {e}")
            raise VectorDBError("update", str(e))
    
    async def get_embedding(self, vector_id: str) -> Optional[List[float]]:
        """
        Get embedding by vector ID.
        
        Args:
            vector_id: Vector ID to retrieve
            
        Returns:
            Embedding vector or None if not found
        """
        try:
            return await self._get_vector(vector_id)
        except Exception as e:
            self.logger.error(f"Failed to get embedding {vector_id}: {e}")
            return None
    
    async def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector ID.
        
        Args:
            vector_id: Vector ID to get metadata for
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            return await self._get_vector_metadata(vector_id)
        except Exception as e:
            self.logger.error(f"Failed to get metadata for {vector_id}: {e}")
            return None
    
    async def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Create or rebuild the vector index.
        
        Args:
            index_params: Optional index parameters
        """
        try:
            params = {**self.config, **(index_params or {})}
            self.logger.info(f"Creating vector index in {self.db_type} with params: {params}")
            
            await self._create_vector_index(params)
            
            self.logger.info("Successfully created vector index")
            
        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            raise VectorDBError("index_creation", str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector database statistics.
        
        Returns:
            Dictionary with vector database statistics
        """
        try:
            db_stats = await self._get_database_stats()
            
            return {
                **self.stats,
                'database_stats': db_stats,
                'config': self.config,
                'db_type': self.db_type,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get vector database stats: {e}")
            return self.stats
    
    # ChromaDB Implementation
    async def _init_chroma_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Suppress ChromaDB telemetry warnings
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            
            if self.connection_string.startswith('/'):
                # Local path
                db_path = self.connection_string
                self.vector_client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                # Remote server
                host, port = self.connection_string.split(':')
                self.vector_client = chromadb.HttpClient(
                    host=host,
                    port=int(port),
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )
            
            # Create or get collection
            collection_name = "ragify_contexts"
            try:
                self.collection = self.vector_client.get_collection(collection_name)
            except:
                self.collection = self.vector_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Ragify context embeddings"}
                )
            
        except Exception as e:
            raise VectorDBError("chroma_init", f"Failed to initialize ChromaDB: {e}")
    
    async def _store_vectors_chroma(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in ChromaDB."""
        try:
            ids = [data['id'] for data in vector_data]
            embeddings = [data['embedding'] for data in vector_data]
            metadatas = [data['metadata'] for data in vector_data]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return ids
        except Exception as e:
            raise VectorDBError("chroma_store", f"Failed to store vectors in ChromaDB: {e}")
    
    async def _search_vectors_chroma(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in ChromaDB."""
        try:
            # Convert filters to ChromaDB format
            where_filter = None
            if filters:
                if len(filters) == 1:
                    # Single filter
                    key, value = list(filters.items())[0]
                    if isinstance(value, (list, tuple)):
                        where_filter = {key: {"$in": value}}
                    else:
                        where_filter = {key: value}
                else:
                    # Multiple filters - use AND operator
                    where_filter = {"$and": []}
                    for key, value in filters.items():
                        if isinstance(value, (list, tuple)):
                            where_filter["$and"].append({key: {"$in": value}})
                        else:
                            where_filter["$and"].append({key: value})
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"]
            )
            
            # Convert results to standard format
            vector_results = []
            if results['ids'] and results['ids'][0]:
                for i, vector_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    vector_results.append((vector_id, similarity_score, metadata))
            
            return vector_results
        except Exception as e:
            raise VectorDBError("chroma_search", f"Failed to search vectors in ChromaDB: {e}")
    
    # Pinecone Implementation
    async def _init_pinecone_client(self) -> None:
        """Initialize Pinecone client."""
        try:
            # Parse connection string: api_key:index_name
            if ':' not in self.connection_string:
                raise VectorDBError("pinecone_init", "Pinecone connection string must be 'api_key:index_name'")
            
            api_key, index_name = self.connection_string.split(':', 1)
            
            # Try new Pinecone API first (v2+)
            try:
                self.vector_client = pinecone.Pinecone(api_key=api_key)
                self.index = self.vector_client.Index(index_name)
            except AttributeError:
                # Fallback to old Pinecone API (v1)
                try:
                    pinecone.init(api_key=api_key, environment=self.config.get('environment', 'us-west1-gcp'))
                    self.index = pinecone.Index(index_name)
                except Exception as e:
                    raise VectorDBError("pinecone_init", f"Failed to initialize Pinecone with old API: {e}")
            
        except Exception as e:
            raise VectorDBError("pinecone_init", f"Failed to initialize Pinecone: {e}")
    
    async def _store_vectors_pinecone(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in Pinecone."""
        try:
            vectors = []
            for data in vector_data:
                vectors.append({
                    'id': data['id'],
                    'values': data['embedding'],
                    'metadata': data['metadata']
                })
            
            # Pinecone upsert with new API
            self.index.upsert(vectors=vectors)
            
            return [data['id'] for data in vector_data]
        except Exception as e:
            raise VectorDBError("pinecone_store", f"Failed to store vectors in Pinecone: {e}")
    
    async def _search_vectors_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in Pinecone."""
        try:
            # Convert filters to Pinecone format
            filter_dict = None
            if filters:
                filter_dict = {}
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        filter_dict[key] = {"$in": value}
                    else:
                        filter_dict[key] = value
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert results to standard format
            vector_results = []
            for match in results.matches:
                vector_results.append((
                    match.id,
                    match.score,
                    match.metadata or {}
                ))
            
            return vector_results
        except Exception as e:
            raise VectorDBError("pinecone_search", f"Failed to search vectors in Pinecone: {e}")
    
    # Weaviate Implementation
    async def _init_weaviate_client(self) -> None:
        """Initialize Weaviate client."""
        try:
            # Parse connection string: host:port
            if ':' in self.connection_string:
                host, port = self.connection_string.split(':')
                url = f"http://{host}:{port}"
            else:
                url = f"http://{self.connection_string}:8080"
            
            self.vector_client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY", "")),
                headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")}
            )
            
            # Create schema if it doesn't exist
            await self._create_weaviate_schema()
            
        except Exception as e:
            raise VectorDBError("weaviate_init", f"Failed to initialize Weaviate: {e}")
    
    async def _create_weaviate_schema(self) -> None:
        """Create Weaviate schema for context chunks."""
        try:
            schema = {
                "class": "ContextChunk",
                "description": "A context chunk with embedding",
                "properties": [
                    {
                        "name": "chunk_id",
                        "dataType": ["string"],
                        "description": "Unique chunk identifier"
                    },
                    {
                        "name": "source_name",
                        "dataType": ["string"],
                        "description": "Source name"
                    },
                    {
                        "name": "source_type",
                        "dataType": ["string"],
                        "description": "Source type"
                    },
                    {
                        "name": "content_preview",
                        "dataType": ["text"],
                        "description": "Content preview"
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Creation timestamp"
                    }
                ],
                "vectorizer": "none"  # We'll provide vectors manually
            }
            
            # Try to create schema (ignore if already exists)
            try:
                self.vector_client.schema.create_class(schema)
            except:
                pass  # Schema might already exist
                
        except Exception as e:
            raise VectorDBError("weaviate_schema", f"Failed to create Weaviate schema: {e}")
    
    async def _store_vectors_weaviate(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in Weaviate."""
        try:
            with self.vector_client.batch as batch:
                for data in vector_data:
                    batch.add_object(
                        class_name="ContextChunk",
                        data_object={
                            "chunk_id": data['id'],
                            "source_name": data['metadata']['source_name'],
                            "source_type": data['metadata']['source_type'],
                            "content_preview": data['metadata']['content_preview'],
                            "created_at": data['metadata']['created_at']
                        },
                        vector=data['embedding']
                    )
            
            return [data['id'] for data in vector_data]
        except Exception as e:
            raise VectorDBError("weaviate_store", f"Failed to store vectors in Weaviate: {e}")
    
    async def _search_vectors_weaviate(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in Weaviate."""
        try:
            # Convert filters to Weaviate format
            where_filter = None
            if filters:
                where_filter = {
                    "operator": "And",
                    "operands": []
                }
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        where_filter["operands"].append({
                            "path": [key],
                            "operator": "ContainsAny",
                            "valueString": value
                        })
                    else:
                        where_filter["operands"].append({
                            "path": [key],
                            "operator": "Equal",
                            "valueString": str(value)
                        })
            
            results = self.vector_client.query.get(
                "ContextChunk", ["chunk_id", "source_name", "source_type", "content_preview"]
            ).with_near_vector({
                "vector": query_embedding
            }).with_limit(top_k).with_where(where_filter).do()
            
            # Convert results to standard format
            vector_results = []
            for result in results['data']['Get']['ContextChunk']:
                vector_results.append((
                    result['chunk_id'],
                    result.get('_additional', {}).get('certainty', 0.0),
                    {
                        'source_name': result['source_name'],
                        'source_type': result['source_type'],
                        'content_preview': result['content_preview']
                    }
                ))
            
            return vector_results
        except Exception as e:
            raise VectorDBError("weaviate_search", f"Failed to search vectors in Weaviate: {e}")
    
    # FAISS Implementation
    async def _init_faiss_client(self) -> None:
        """Initialize FAISS client."""
        try:
            # Parse connection string as file path
            index_path = Path(self.connection_string)
            metadata_path = index_path.parent / f"{index_path.stem}_metadata.pkl"
            
            self.index_path = index_path
            self.metadata_path = metadata_path
            
            # Load existing index or create new one
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        self.metadata_store = pickle.load(f)
                else:
                    self.metadata_store = {}
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.config['dimension'])  # Inner product for cosine similarity
                self.metadata_store = {}
            
            # Update config dimension based on actual index
            self.config['dimension'] = self.index.d
            
            self.stats['total_vectors'] = self.index.ntotal
            
        except Exception as e:
            raise VectorDBError("faiss_init", f"Failed to initialize FAISS: {e}")
    
    # Memory Implementation
    async def _init_memory_client(self) -> None:
        """Initialize in-memory vector database."""
        try:
            # Don't create index until we know the dimension
            self.index = None
            self.metadata_store = {}
            
            self.stats['total_vectors'] = 0
            
        except Exception as e:
            raise VectorDBError("memory_init", f"Failed to initialize memory database: {e}")
    
    async def _store_vectors_memory(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in memory database."""
        try:
            if not vector_data:
                return []
                
            vectors = np.array([data['embedding'] for data in vector_data], dtype=np.float32)
            
            # Initialize index if not exists or check dimension mismatch
            if self.index is None:
                # First time adding vectors, create index with correct dimension
                self.index = faiss.IndexFlatIP(vectors.shape[1])
                self.config['dimension'] = vectors.shape[1]
            elif vectors.shape[1] != self.index.d:
                # Dimension mismatch, create new index
                self.index = faiss.IndexFlatIP(vectors.shape[1])
                self.config['dimension'] = vectors.shape[1]
                self.metadata_store = {}  # Clear metadata since we're starting fresh
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Store metadata with proper indexing
            vector_ids = []
            for i, data in enumerate(vector_data):
                vector_id = data['id']
                self.metadata_store[vector_id] = data['metadata']
                vector_ids.append(vector_id)
            
            # Update stats
            self.stats['total_vectors'] = self.index.ntotal
            
            return vector_ids
        except Exception as e:
            raise VectorDBError("memory_store", f"Failed to store vectors in memory database: {e}")
    
    async def _search_vectors_memory(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in memory database."""
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            scores, indices = self.index.search(query_vector, top_k)
            
            # Convert results to standard format
            vector_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    # Get vector ID from metadata (assuming sequential storage)
                    vector_id = list(self.metadata_store.keys())[idx]
                    metadata = self.metadata_store[vector_id]
                    
                    vector_results.append((vector_id, float(score), metadata))
            
            return vector_results
        except Exception as e:
            raise VectorDBError("memory_search", f"Failed to search vectors in memory database: {e}")
    
    async def _store_vectors_faiss(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in FAISS."""
        try:
            vectors = np.array([data['embedding'] for data in vector_data], dtype=np.float32)
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Store metadata
            for data in vector_data:
                self.metadata_store[data['id']] = data['metadata']
            
            # Save index and metadata
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            return [data['id'] for data in vector_data]
        except Exception as e:
            raise VectorDBError("faiss_store", f"Failed to store vectors in FAISS: {e}")
    
    async def _search_vectors_faiss(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in FAISS."""
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            scores, indices = self.index.search(query_vector, top_k)
            
            # Convert results to standard format
            vector_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    # Get vector ID from metadata (assuming sequential storage)
                    vector_id = list(self.metadata_store.keys())[idx]
                    metadata = self.metadata_store[vector_id]
                    
                    vector_results.append((vector_id, float(score), metadata))
            
            return vector_results
        except Exception as e:
            raise VectorDBError("faiss_search", f"Failed to search vectors in FAISS: {e}")
    
    # Generic wrapper methods
    async def _store_vectors(self, vector_data: List[Dict[str, Any]]) -> List[str]:
        """Store vectors in the database."""
        if self.db_type == 'chroma':
            return await self._store_vectors_chroma(vector_data)
        elif self.db_type == 'pinecone':
            return await self._store_vectors_pinecone(vector_data)
        elif self.db_type == 'weaviate':
            return await self._store_vectors_weaviate(vector_data)
        elif self.db_type == 'faiss':
            return await self._store_vectors_faiss(vector_data)
        elif self.db_type == 'memory':
            return await self._store_vectors_memory(vector_data)
        else:
            raise VectorDBError("store", f"Unsupported database type: {self.db_type}")
    
    async def _search_vectors(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if self.db_type == 'chroma':
            return await self._search_vectors_chroma(query_embedding, top_k, min_score, filters)
        elif self.db_type == 'pinecone':
            return await self._search_vectors_pinecone(query_embedding, top_k, min_score, filters)
        elif self.db_type == 'weaviate':
            return await self._search_vectors_weaviate(query_embedding, top_k, min_score, filters)
        elif self.db_type == 'faiss':
            return await self._search_vectors_faiss(query_embedding, top_k, min_score, filters)
        elif self.db_type == 'memory':
            return await self._search_vectors_memory(query_embedding, top_k, min_score, filters)
        else:
            raise VectorDBError("search", f"Unsupported database type: {self.db_type}")
    
    async def _delete_vectors(self, vector_ids: List[str]) -> None:
        """Delete vectors from the database."""
        if self.db_type == 'chroma':
            self.collection.delete(ids=vector_ids)
        elif self.db_type == 'pinecone':
            self.index.delete(ids=vector_ids)
        elif self.db_type == 'weaviate':
            for vector_id in vector_ids:
                self.vector_client.data_object.delete_by_id(vector_id, "ContextChunk")
        elif self.db_type == 'faiss':
            # FAISS doesn't support deletion, we'll need to rebuild
            self.logger.warning("FAISS doesn't support deletion, consider rebuilding index")
        elif self.db_type == 'memory':
            # Memory database doesn't support deletion
            self.logger.warning("Memory database doesn't support deletion")
        else:
            raise VectorDBError("delete", f"Unsupported database type: {self.db_type}")
    
    async def _update_vectors(
        self,
        vector_ids: List[str],
        new_embeddings: List[List[float]],
    ) -> None:
        """Update vectors in the database."""
        if self.db_type == 'chroma':
            self.collection.update(
                ids=vector_ids,
                embeddings=new_embeddings
            )
        elif self.db_type == 'pinecone':
            vectors = []
            for vector_id, embedding in zip(vector_ids, new_embeddings):
                vectors.append({
                    'id': vector_id,
                    'values': embedding
                })
            self.index.upsert(vectors=vectors)
        elif self.db_type == 'weaviate':
            # Weaviate doesn't support direct updates, delete and recreate
            await self._delete_vectors(vector_ids)
            # Re-add with new embeddings
            vector_data = [
                {'id': vid, 'embedding': emb, 'metadata': {}} 
                for vid, emb in zip(vector_ids, new_embeddings)
            ]
            await self._store_vectors(vector_data)
        elif self.db_type == 'faiss':
            # FAISS doesn't support updates, we'll need to rebuild
            self.logger.warning("FAISS doesn't support updates, consider rebuilding index")
        elif self.db_type == 'memory':
            # Memory database doesn't support updates
            self.logger.warning("Memory database doesn't support updates")
        else:
            raise VectorDBError("update", f"Unsupported database type: {self.db_type}")
    
    async def _get_vector(self, vector_id: str) -> Optional[List[float]]:
        """Get vector by ID."""
        if self.db_type == 'chroma':
            result = self.collection.get(ids=[vector_id], include=['embeddings'])
            if result['embeddings']:
                return result['embeddings'][0]
        elif self.db_type == 'pinecone':
            result = self.index.fetch(ids=[vector_id])
            if vector_id in result.vectors:
                return result.vectors[vector_id].values
        elif self.db_type == 'weaviate':
            # Weaviate doesn't support direct vector retrieval
            return None
        elif self.db_type == 'faiss':
            # FAISS doesn't support direct vector retrieval by ID
            return None
        elif self.db_type == 'memory':
            # Memory database doesn't support direct vector retrieval by ID
            return None
        
        return None
    
    async def _get_vector_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get vector metadata by ID."""
        if self.db_type == 'chroma':
            result = self.collection.get(ids=[vector_id], include=['metadatas'])
            if result['metadatas']:
                return result['metadatas'][0]
        elif self.db_type == 'pinecone':
            result = self.index.fetch(ids=[vector_id])
            if vector_id in result.vectors:
                return result.vectors[vector_id].metadata
        elif self.db_type == 'weaviate':
            result = self.vector_client.data_object.get_by_id(vector_id, "ContextChunk")
            if result:
                return result.properties
        elif self.db_type == 'faiss':
            return self.metadata_store.get(vector_id)
        elif self.db_type == 'memory':
            return self.metadata_store.get(vector_id)
        
        return None
    
    async def _create_vector_index(self, params: Dict[str, Any]) -> None:
        """Create vector index."""
        if self.db_type == 'chroma':
            # ChromaDB creates indexes automatically
            pass
        elif self.db_type == 'pinecone':
            # Pinecone indexes are created during initialization
            pass
        elif self.db_type == 'weaviate':
            # Weaviate schemas are created during initialization
            pass
        elif self.db_type == 'faiss':
            # Rebuild FAISS index with new parameters
            if self.index.ntotal > 0:
                # Get all vectors
                vectors = self.index.reconstruct_n(0, self.index.ntotal)
                
                # Create new index with new parameters
                if params.get('index_type') == 'ivf':
                    nlist = params.get('nlist', 100)
                    # For small datasets, use flat index instead of IVF
                    if self.index.ntotal < 100:  # Small dataset threshold
                        self.index = faiss.IndexFlatIP(self.config['dimension'])
                        self.index.add(vectors)
                    else:
                        # Ensure nlist doesn't exceed number of vectors
                        nlist = min(nlist, self.index.ntotal)
                        if nlist > 1:  # Need at least 2 clusters for IVF
                            self.index = faiss.IndexIVFFlat(
                                faiss.IndexFlatIP(self.config['dimension']),
                                self.config['dimension'],
                                nlist
                            )
                            self.index.train(vectors)
                            self.index.add(vectors)
                        else:
                            # Fall back to flat index for small datasets
                            self.index = faiss.IndexFlatIP(self.config['dimension'])
                            self.index.add(vectors)
                else:
                    self.index = faiss.IndexFlatIP(self.config['dimension'])
                    self.index.add(vectors)
                
                # Save new index
                faiss.write_index(self.index, str(self.index_path))
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'db_type': self.db_type,
            'total_vectors': self.stats['total_vectors'],
        }
        
        if self.db_type == 'chroma':
            stats['collection_count'] = len(self.vector_client.list_collections())
        elif self.db_type == 'pinecone':
            stats['index_stats'] = self.index.describe_index_stats()
        elif self.db_type == 'weaviate':
            stats['schema_classes'] = len(self.vector_client.schema.get())
        elif self.db_type == 'faiss':
            stats['index_size'] = self.index.ntotal
            stats['index_dimension'] = self.index.d
        elif self.db_type == 'memory':
            stats['index_size'] = self.index.ntotal
            stats['index_dimension'] = self.index.d
        
        return stats
    
    async def close(self) -> None:
        """Close the vector database connection."""
        try:
            if self.vector_client:
                if self.db_type == 'chroma':
                    # ChromaDB doesn't need explicit closing
                    pass
                elif self.db_type == 'pinecone':
                    # Pinecone doesn't need explicit closing
                    pass
                elif self.db_type == 'weaviate':
                    self.vector_client.close()
                elif self.db_type == 'faiss':
                    # Save index before closing
                    if hasattr(self, 'index_path') and self.index:
                        faiss.write_index(self.index, str(self.index_path))
                        with open(self.metadata_path, 'wb') as f:
                            pickle.dump(self.metadata_store, f)
                elif self.db_type == 'memory':
                    # Memory database doesn't need explicit closing
                    pass
            
            self.logger.info(f"Vector database connection closed: {self.db_type}")
            
        except Exception as e:
            self.logger.error(f"Error closing vector database connection: {e}")
    
    # Performance optimization methods
    async def _get_connection(self, connection_key: str = "default"):
        """
        Get a connection from the pool or create a new one.
        
        Args:
            connection_key: Key to identify the connection
            
        Returns:
            Database connection object
        """
        if connection_key in self.connection_pool:
            connection = self.connection_pool[connection_key]
            if self._is_connection_valid(connection):
                self.stats['connection_pool_hits'] += 1
                return connection
            else:
                # Remove invalid connection
                del self.connection_pool[connection_key]
        
        # Create new connection
        try:
            connection = await self._create_connection()
            if len(self.connection_pool) < self.max_connections:
                self.connection_pool[connection_key] = connection
                self.stats['connection_pool_misses'] += 1
            return connection
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            raise
    
    def _is_connection_valid(self, connection) -> bool:
        """Check if a connection is still valid."""
        try:
            # Basic connection validation - can be overridden by specific implementations
            if hasattr(connection, 'ping'):
                return connection.ping()
            elif hasattr(connection, 'health'):
                return connection.health()
            else:
                # Assume connection is valid if no validation method available
                return True
        except Exception:
            return False
    
    async def _create_connection(self):
        """Create a new database connection."""
        # This method should be implemented by specific database classes
        # For now, return None to indicate it needs implementation
        return None
    
    def _generate_search_cache_key(self, query_embedding: List[float], top_k: int, min_score: float, filters: Optional[Dict] = None) -> str:
        """Generate cache key for search results."""
        # Create a deterministic hash of search parameters
        import hashlib
        
        search_params = {
            'query_hash': hashlib.md5(str(query_embedding).encode()).hexdigest()[:16],
            'top_k': top_k,
            'min_score': min_score,
            'filters': filters or {}
        }
        
        search_str = json.dumps(search_params, sort_keys=True)
        return hashlib.md5(search_str.encode()).hexdigest()
    
    def _get_cached_search(self, cache_key: str) -> Optional[Tuple[List[ContextChunk], float]]:
        """Get cached search results if available."""
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.search_cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_result['results'], cached_result['search_time']
            else:
                # Expired cache entry
                del self.search_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def _cache_search_results(self, cache_key: str, results: List[ContextChunk], search_time: float):
        """Cache search results."""
        # Clean up cache if it's too large
        if len(self.search_cache) >= self.search_cache_size:
            self._cleanup_search_cache()
        
        self.search_cache[cache_key] = {
            'results': results,
            'search_time': search_time,
            'timestamp': time.time()
        }
    
    def _cleanup_search_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.search_cache.items()
            if current_time - value['timestamp'] > self.search_cache_ttl
        ]
        
        for key in expired_keys:
            del self.search_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.search_cache) >= self.search_cache_size:
            sorted_items = sorted(
                self.search_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            items_to_remove = len(sorted_items) - self.search_cache_size + 100
            for key, _ in sorted_items[:items_to_remove]:
                del self.search_cache[key]
    
    async def search_optimized(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict] = None,
        use_cache: bool = True
    ) -> List[ContextChunk]:
        """
        Search for similar vectors with caching and connection pooling.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Maximum number of results
            min_score: Minimum similarity score
            filters: Optional filters for search
            use_cache: Whether to use search result caching
            
        Returns:
            List of context chunks with similarity scores
        """
        start_time = time.time()
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._generate_search_cache_key(query_embedding, top_k, min_score, filters)
            cached_result = self._get_cached_search(cache_key)
            if cached_result:
                results, _ = cached_result
                self.logger.debug(f"Search cache hit: {len(results)} results")
                return results
        
        try:
            # Get connection from pool
            connection = await self._get_connection()
            
            # Perform search
            results = await self._perform_search(
                connection, query_embedding, top_k, min_score, filters
            )
            
            # Update statistics
            search_time = time.time() - start_time
            self.stats['searches_performed'] += 1
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['searches_performed'] - 1) + search_time) /
                self.stats['searches_performed']
            )
            
            # Cache results if enabled
            if use_cache:
                self._cache_search_results(cache_key, results, search_time)
            
            self.logger.debug(f"Search completed in {search_time:.3f}s: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise VectorDBError(f"Search operation failed: {e}")
    
    async def _perform_search(
        self,
        connection,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filters: Optional[Dict]
    ) -> List[ContextChunk]:
        """
        Perform the actual search operation.
        This method should be implemented by specific database implementations.
        
        Args:
            connection: Database connection
            query_embedding: Query vector
            top_k: Maximum results
            min_score: Minimum score
            filters: Search filters
            
        Returns:
            List of context chunks
        """
        # Default implementation - should be overridden
        raise NotImplementedError("_perform_search must be implemented by subclasses")
