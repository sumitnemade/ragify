"""
Tests for intelligent multi-source context fusion with conflict resolution.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from ragify.engines.fusion import IntelligentContextFusionEngine
from ragify.models import (
    Context, ContextChunk, ContextSource, SourceType, RelevanceScore,
    OrchestratorConfig, ConflictType, ConflictResolutionStrategy,
    PrivacyLevel
)


class TestIntelligentFusion:
    """Test intelligent context fusion with conflict resolution."""
    
    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine for testing."""
        config = OrchestratorConfig(
            conflict_detection_threshold=0.7,
            fusion_config={
                'enable_conflict_detection': True,
                'enable_semantic_analysis': True,
            }
        )
        return IntelligentContextFusionEngine(config)
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample data sources."""
        return {
            'database': ContextSource(
                name="Database",
                source_type=SourceType.DATABASE,
                authority_score=1.0,
                freshness_score=0.9
            ),
            'api': ContextSource(
                name="API",
                source_type=SourceType.API,
                authority_score=0.9,
                freshness_score=1.0
            ),
            'document': ContextSource(
                name="Document",
                source_type=SourceType.DOCUMENT,
                authority_score=0.8,
                freshness_score=0.7
            ),
            'cache': ContextSource(
                name="Cache",
                source_type=SourceType.CACHE,
                authority_score=0.5,
                freshness_score=0.6
            )
        }
    
    @pytest.fixture
    def conflicting_chunks(self, sample_sources):
        """Create chunks with conflicts for testing."""
        now = datetime.utcnow()
        old_time = now - timedelta(days=60)
        
        return [
            # Database source - high authority, recent
            ContextChunk(
                content="Sales revenue for Q1 2024 was $1.2 million",
                source=sample_sources['database'],
                relevance_score=RelevanceScore(score=0.9),
                created_at=now,
                updated_at=now,
                token_count=15
            ),
            # API source - conflicting data, high authority
            ContextChunk(
                content="Sales revenue for Q1 2024 was $1.5 million",
                source=sample_sources['api'],
                relevance_score=RelevanceScore(score=0.8),
                created_at=now,
                updated_at=now,
                token_count=15
            ),
            # Document source - old data, lower authority
            ContextChunk(
                content="Sales revenue for Q1 2024 was $1.0 million",
                source=sample_sources['document'],
                relevance_score=RelevanceScore(score=0.7),
                created_at=old_time,
                updated_at=old_time,
                token_count=15
            ),
            # Cache source - conflicting data, low authority
            ContextChunk(
                content="Sales revenue for Q1 2024 was $1.3 million",
                source=sample_sources['cache'],
                relevance_score=RelevanceScore(score=0.6),
                created_at=now,
                updated_at=now,
                token_count=15
            )
        ]
    
    @pytest.fixture
    def sample_contexts(self, conflicting_chunks):
        """Create sample contexts for fusion testing."""
        return [
            Context(
                query="What was the Q1 2024 sales revenue?",
                chunks=conflicting_chunks[:2],  # Database and API
                user_id="user123",
                session_id="session456",
                privacy_level=PrivacyLevel.PRIVATE
            ),
            Context(
                query="What was the Q1 2024 sales revenue?",
                chunks=conflicting_chunks[2:],  # Document and Cache
                user_id="user123",
                session_id="session456",
                privacy_level=PrivacyLevel.PRIVATE
            )
        ]
    
    @pytest.mark.asyncio
    async def test_intelligent_fusion_basic(self, fusion_engine, sample_contexts):
        """Test basic intelligent fusion without conflicts."""
        # Create contexts with non-conflicting chunks
        context1 = Context(
            query="What is the company structure?",
            chunks=[
                ContextChunk(
                    content="The company has 3 departments: Sales, Marketing, and Engineering.",
                    source=ContextSource(name="HR", source_type=SourceType.DATABASE),
                    relevance_score=RelevanceScore(score=0.8)
                )
            ],
            user_id="user123"
        )
        
        context2 = Context(
            query="What is the company structure?",
            chunks=[
                ContextChunk(
                    content="Engineering team has 25 developers.",
                    source=ContextSource(name="Engineering", source_type=SourceType.API),
                    relevance_score=RelevanceScore(score=0.7)
                )
            ],
            user_id="user123"
        )
        
        fused_context = await fusion_engine.fuse_contexts([context1, context2])
        
        assert fused_context is not None
        assert len(fused_context.chunks) == 2  # Both chunks should be preserved
        assert fused_context.fusion_metadata is not None
        assert fused_context.fusion_metadata.conflict_count == 0
        assert fused_context.fusion_metadata.fusion_strategy == "intelligent"
    
    @pytest.mark.asyncio
    async def test_conflict_detection_factual_disagreement(self, fusion_engine, sample_contexts):
        """Test detection of factual disagreements."""
        fused_context = await fusion_engine.fuse_contexts(
            sample_contexts,
            conflict_resolution=ConflictResolutionStrategy.HIGHEST_AUTHORITY
        )
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        assert fused_context.fusion_metadata.conflict_count > 0
        
        # Check that conflicts were detected
        conflicts = fused_context.fusion_metadata.resolved_conflicts
        assert len(conflicts) > 0
        
        # Should have factual disagreement conflicts
        factual_conflicts = [
            c for c in conflicts 
            if c.conflict_type == ConflictType.FACTUAL_DISAGREEMENT
        ]
        assert len(factual_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_highest_authority(self, fusion_engine, sample_contexts):
        """Test conflict resolution using highest authority strategy."""
        fused_context = await fusion_engine.fuse_contexts(
            sample_contexts,
            conflict_resolution=ConflictResolutionStrategy.HIGHEST_AUTHORITY
        )
        
        assert fused_context is not None
        assert len(fused_context.chunks) > 0
        
        # Should resolve to database source (highest authority)
        resolved_chunk = fused_context.chunks[0]
        assert resolved_chunk.source.source_type == SourceType.DATABASE
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_newest_data(self, fusion_engine, sample_contexts):
        """Test conflict resolution using newest data strategy."""
        fused_context = await fusion_engine.fuse_contexts(
            sample_contexts,
            conflict_resolution=ConflictResolutionStrategy.NEWEST_DATA
        )
        
        assert fused_context is not None
        assert len(fused_context.chunks) > 0
        
        # Should prefer newer data over older data
        resolved_chunk = fused_context.chunks[0]
        # Should not be from document source (oldest)
        assert resolved_chunk.source.source_type != SourceType.DOCUMENT
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_consensus(self, fusion_engine, sample_contexts):
        """Test conflict resolution using consensus strategy."""
        fused_context = await fusion_engine.fuse_contexts(
            sample_contexts,
            conflict_resolution=ConflictResolutionStrategy.CONSENSUS
        )
        
        assert fused_context is not None
        assert len(fused_context.chunks) > 0
        
        # Consensus should consider both relevance and authority
        resolved_chunk = fused_context.chunks[0]
        assert resolved_chunk.relevance_score is not None
    
    @pytest.mark.asyncio
    async def test_temporal_conflict_detection(self, fusion_engine):
        """Test detection of temporal conflicts."""
        now = datetime.utcnow()
        old_time = now - timedelta(days=60)
        
        context1 = Context(
            query="What is the current market share?",
            chunks=[
                ContextChunk(
                    content="Market share is 15% as of Q1 2024",
                    source=ContextSource(name="Recent", source_type=SourceType.API),
                    created_at=now,
                    updated_at=now,
                    relevance_score=RelevanceScore(score=0.8)
                )
            ],
            user_id="user123"
        )
        
        context2 = Context(
            query="What is the current market share?",
            chunks=[
                ContextChunk(
                    content="Market share is 12% as of Q4 2023",
                    source=ContextSource(name="Old", source_type=SourceType.DOCUMENT),
                    created_at=old_time,
                    updated_at=old_time,
                    relevance_score=RelevanceScore(score=0.7)
                )
            ],
            user_id="user123"
        )
        
        fused_context = await fusion_engine.fuse_contexts([context1, context2])
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        
        # Should detect temporal conflict
        temporal_conflicts = [
            c for c in fused_context.fusion_metadata.resolved_conflicts
            if c.conflict_type == ConflictType.TEMPORAL_CONFLICT
        ]
        assert len(temporal_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_source_authority_conflict_detection(self, fusion_engine):
        """Test detection of source authority conflicts."""
        context1 = Context(
            query="What is the employee count?",
            chunks=[
                ContextChunk(
                    content="Company has 150 employees",
                    source=ContextSource(name="HR Database", source_type=SourceType.DATABASE),
                    relevance_score=RelevanceScore(score=0.9)
                )
            ],
            user_id="user123"
        )
        
        context2 = Context(
            query="What is the employee count?",
            chunks=[
                ContextChunk(
                    content="Company has 140 employees",
                    source=ContextSource(name="Cache", source_type=SourceType.CACHE),
                    relevance_score=RelevanceScore(score=0.6)
                )
            ],
            user_id="user123"
        )
        
        fused_context = await fusion_engine.fuse_contexts([context1, context2])
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        
        # Should detect source authority conflict
        authority_conflicts = [
            c for c in fused_context.fusion_metadata.resolved_conflicts
            if c.conflict_type == ConflictType.SOURCE_AUTHORITY
        ]
        assert len(authority_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_conflict_detection(self, fusion_engine):
        """Test detection of semantic conflicts."""
        context1 = Context(
            query="What is the product strategy?",
            chunks=[
                ContextChunk(
                    content="The product strategy focuses on enterprise customers with premium pricing",
                    source=ContextSource(name="Strategy Doc", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.8)
                )
            ],
            user_id="user123"
        )
        
        context2 = Context(
            query="What is the product strategy?",
            chunks=[
                ContextChunk(
                    content="The product strategy focuses on enterprise customers with competitive pricing",
                    source=ContextSource(name="API", source_type=SourceType.API),
                    relevance_score=RelevanceScore(score=0.7)
                )
            ],
            user_id="user123"
        )
        
        fused_context = await fusion_engine.fuse_contexts([context1, context2])
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        
        # Should detect semantic conflict due to similar content but different sources
        semantic_conflicts = [
            c for c in fused_context.fusion_metadata.resolved_conflicts
            if c.conflict_type == ConflictType.SEMANTIC_CONFLICT
        ]
        assert len(semantic_conflicts) > 0
    
    @pytest.mark.asyncio
    async def test_fusion_confidence_calculation(self, fusion_engine, sample_contexts):
        """Test fusion confidence calculation."""
        fused_context = await fusion_engine.fuse_contexts(sample_contexts)
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        assert fused_context.fusion_metadata.fusion_confidence >= 0.0
        assert fused_context.fusion_metadata.fusion_confidence <= 1.0
        
        # Confidence should be affected by conflicts
        if fused_context.fusion_metadata.conflict_count > 0:
            assert fused_context.fusion_metadata.fusion_confidence < 1.0
    
    @pytest.mark.asyncio
    async def test_weighted_average_resolution(self, fusion_engine, sample_contexts):
        """Test weighted average conflict resolution."""
        fused_context = await fusion_engine.fuse_contexts(
            sample_contexts,
            conflict_resolution=ConflictResolutionStrategy.WEIGHTED_AVERAGE
        )
        
        assert fused_context is not None
        assert len(fused_context.chunks) > 0
        
        # Should create a weighted average chunk
        resolved_chunk = fused_context.chunks[0]
        assert resolved_chunk.metadata.get('resolution_method') == 'weighted_average'
        assert 'original_chunks' in resolved_chunk.metadata
    
    @pytest.mark.asyncio
    async def test_fusion_metadata_completeness(self, fusion_engine, sample_contexts):
        """Test that fusion metadata is complete and accurate."""
        fused_context = await fusion_engine.fuse_contexts(sample_contexts)
        
        assert fused_context is not None
        assert fused_context.fusion_metadata is not None
        
        metadata = fused_context.fusion_metadata
        
        # Check required fields
        assert metadata.fusion_strategy == "intelligent"
        assert metadata.conflict_count >= 0
        assert metadata.fusion_confidence >= 0.0
        assert metadata.fusion_confidence <= 1.0
        assert metadata.processing_time > 0.0
        
        # Check conflicts list
        assert len(metadata.resolved_conflicts) == metadata.conflict_count
        
        # Check conflict details
        for conflict in metadata.resolved_conflicts:
            assert conflict.conflict_type in ConflictType
            assert len(conflict.conflicting_chunks) > 0
            assert conflict.confidence >= 0.0
            assert conflict.confidence <= 1.0
            assert conflict.description
            assert conflict.resolution_strategy in ConflictResolutionStrategy
    
    @pytest.mark.asyncio
    async def test_single_context_fusion(self, fusion_engine):
        """Test fusion with single context (should return unchanged)."""
        context = Context(
            query="Test query",
            chunks=[
                ContextChunk(
                    content="Test content",
                    source=ContextSource(name="Test", source_type=SourceType.DOCUMENT),
                    relevance_score=RelevanceScore(score=0.8)
                )
            ],
            user_id="user123"
        )
        
        fused_context = await fusion_engine.fuse_contexts([context])
        
        assert fused_context is not None
        assert fused_context.id == context.id
        assert len(fused_context.chunks) == 1
        assert fused_context.chunks[0].content == "Test content"
    
    @pytest.mark.asyncio
    async def test_empty_contexts_fusion(self, fusion_engine):
        """Test fusion with empty contexts list (should raise exception)."""
        with pytest.raises(Exception):
            await fusion_engine.fuse_contexts([])
    
    @pytest.mark.asyncio
    async def test_unknown_fusion_strategy(self, fusion_engine, sample_contexts):
        """Test fusion with unknown strategy (should raise exception)."""
        with pytest.raises(Exception):
            await fusion_engine.fuse_contexts(sample_contexts, strategy="unknown_strategy")
    
    @pytest.mark.asyncio
    async def test_unknown_resolution_strategy(self, fusion_engine, sample_contexts):
        """Test fusion with unknown resolution strategy (should raise exception)."""
        with pytest.raises(Exception):
            await fusion_engine.fuse_contexts(
                sample_contexts,
                conflict_resolution="unknown_resolution"
            )
