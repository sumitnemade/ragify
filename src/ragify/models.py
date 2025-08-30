"""
Core data models for the Intelligent Context Orchestration plugin.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class PrivacyLevel(str, Enum):
    """Privacy levels for context storage."""
    PUBLIC = "public"
    PRIVATE = "private"
    ENTERPRISE = "enterprise"
    RESTRICTED = "restricted"


class SourceType(str, Enum):
    """Types of data sources."""
    DOCUMENT = "document"
    API = "api"
    DATABASE = "database"
    REALTIME = "realtime"
    VECTOR = "vector"
    CACHE = "cache"


class ConflictType(str, Enum):
    """Types of conflicts that can occur during fusion."""
    CONTENT_CONTRADICTION = "content_contradiction"
    FACTUAL_DISAGREEMENT = "factual_disagreement"
    TEMPORAL_CONFLICT = "temporal_conflict"
    SOURCE_AUTHORITY = "source_authority"
    DATA_FRESHNESS = "data_freshness"
    SEMANTIC_CONFLICT = "semantic_conflict"


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    HIGHEST_RELEVANCE = "highest_relevance"
    NEWEST_DATA = "newest_data"
    HIGHEST_AUTHORITY = "highest_authority"
    CONSENSUS = "consensus"
    WEIGHTED_AVERAGE = "weighted_average"
    HIERARCHICAL = "hierarchical"
    MANUAL_REVIEW = "manual_review"


class ConflictInfo(BaseModel):
    """Information about a detected conflict."""
    conflict_type: ConflictType
    conflicting_chunks: List[UUID]  # Chunk IDs involved in conflict
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    resolution_strategy: ConflictResolutionStrategy
    resolved_chunk_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FusionMetadata(BaseModel):
    """Metadata about the fusion process."""
    fusion_strategy: str
    conflict_count: int = 0
    resolved_conflicts: List[ConflictInfo] = Field(default_factory=list)
    source_weights: Dict[str, float] = Field(default_factory=dict)
    fusion_confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextSource(BaseModel):
    """Represents a source of context data."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    source_type: SourceType
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_score: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class RelevanceScore(BaseModel):
    """Represents a relevance score with confidence interval."""
    score: float = Field(ge=0.0, le=1.0)
    confidence_lower: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_upper: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict)
    
    @field_validator('confidence_upper')
    @classmethod
    def validate_confidence_bounds(cls, v, info):
        if v is not None and info.data and 'confidence_lower' in info.data and info.data['confidence_lower'] is not None:
            if v < info.data['confidence_lower']:
                raise ValueError('confidence_upper must be >= confidence_lower')
        return v


class ContextChunk(BaseModel):
    """Represents a chunk of context data."""
    id: UUID = Field(default_factory=uuid4)
    content: str
    source: ContextSource
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    relevance_score: Optional[RelevanceScore] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: Optional[int] = None
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    conflict_flags: Set[ConflictType] = Field(default_factory=set)
    fusion_metadata: Optional[FusionMetadata] = None

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class Context(BaseModel):
    """Represents a complete context for an LLM query."""
    id: UUID = Field(default_factory=uuid4)
    query: str
    chunks: List[ContextChunk] = Field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    relevance_score: Optional[RelevanceScore] = None
    total_tokens: int = 0
    max_tokens: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    fusion_metadata: Optional[FusionMetadata] = None

    @field_validator('chunks', mode='before')
    @classmethod
    def validate_chunks(cls, v):
        """Ensure chunks is always a list."""
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return list(v)
        # If it's a single chunk, wrap it in a list
        return [v]

    @property
    def sources(self) -> List[ContextSource]:
        """Get unique sources from chunks."""
        return list({chunk.source.id: chunk.source for chunk in self.chunks}.values())

    @property
    def content(self) -> str:
        """Get combined content from all chunks."""
        return "\n\n".join(chunk.content for chunk in self.chunks)

    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add a chunk to the context."""
        self.chunks.append(chunk)
        self.total_tokens += chunk.token_count or 0

    def remove_chunk(self, chunk_id: UUID) -> None:
        """Remove a chunk from the context."""
        for i, chunk in enumerate(self.chunks):
            if chunk.id == chunk_id:
                self.total_tokens -= chunk.token_count or 0
                del self.chunks[i]
                break

    def optimize_for_tokens(self, max_tokens: int) -> None:
        """Optimize context to fit within token limit."""
        if self.total_tokens <= max_tokens:
            return

        # Sort chunks by relevance score
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
            reverse=True
        )

        # Keep chunks until we hit the limit
        self.chunks = []
        self.total_tokens = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.token_count or 0
            if self.total_tokens + chunk_tokens <= max_tokens:
                self.chunks.append(chunk)
                self.total_tokens += chunk_tokens
            else:
                break

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }


class ContextRequest(BaseModel):
    """Request model for context retrieval."""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    max_tokens: Optional[int] = None
    max_chunks: Optional[int] = None
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    include_metadata: bool = True
    sources: Optional[List[str]] = None  # Source names to include
    exclude_sources: Optional[List[str]] = None  # Source names to exclude
    fusion_strategy: str = "intelligent"
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE


class ContextResponse(BaseModel):
    """Response model for context retrieval."""
    context: Context
    processing_time: float
    cache_hit: bool = False
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    """Configuration for a data source."""
    name: str
    source_type: SourceType
    url: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    refresh_interval: Optional[int] = None  # seconds
    max_retries: int = 3
    timeout: int = 30
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    """Configuration for the context orchestrator."""
    vector_db_url: Optional[str] = None
    cache_url: Optional[str] = None
    privacy_level: PrivacyLevel = PrivacyLevel.PRIVATE
    max_context_size: int = 10000  # tokens
    default_relevance_threshold: float = 0.5
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    enable_analytics: bool = True
    log_level: str = "INFO"
    fusion_config: Dict[str, Any] = Field(default_factory=dict)
    conflict_detection_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    source_timeout: float = 30.0  # seconds for source processing timeout
    max_concurrent_sources: int = 10  # maximum concurrent source processing
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PrivacyRule(BaseModel):
    """Privacy rule for access control."""
    resource_pattern: str
    allowed_roles: List[str]
    required_clearance: PrivacyLevel
    time_restrictions: Optional[Dict[str, Any]] = None
    location_restrictions: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AccessControl(BaseModel):
    """Access control configuration."""
    user_id: str
    role: str
    permissions: List[str]
    clearance_level: PrivacyLevel
    restrictions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateType(str, Enum):
    """Types of updates that can be performed."""
    INCREMENTAL = "incremental"
    FULL = "full"
    PARTIAL = "partial"
    MERGE = "merge"
    REPLACE = "replace"
