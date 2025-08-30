#!/usr/bin/env python3
"""
Intelligent Multi-Source Context Fusion Demo

This example demonstrates the advanced conflict detection and resolution
capabilities of the Ragify intelligent fusion engine.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List

from ragify import ContextOrchestrator
from src.ragify.models import (
    Context, ContextChunk, ContextSource, SourceType, RelevanceScore,
    ConflictType, ConflictResolutionStrategy, PrivacyLevel
)
from src.ragify.sources import DocumentSource, APISource, DatabaseSource


async def create_conflicting_data_sources():
    """Create data sources with conflicting information."""
    
    # Create sources with different authority levels
    sources = {
        'database': ContextSource(
            name="Sales Database",
            source_type=SourceType.DATABASE,
            authority_score=1.0,
            freshness_score=0.9
        ),
        'api': ContextSource(
            name="CRM API",
            source_type=SourceType.API,
            authority_score=0.9,
            freshness_score=1.0
        ),
        'document': ContextSource(
            name="Q4 Report",
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
    
    return sources


async def create_conflicting_chunks(sources):
    """Create chunks with conflicting information."""
    
    now = datetime.utcnow()
    old_time = now - timedelta(days=60)
    
    chunks = [
        # Database source - high authority, recent
        ContextChunk(
            content="Sales revenue for Q1 2024 was $1.2 million with 15% growth",
            source=sources['database'],
            relevance_score=RelevanceScore(score=0.9),
            created_at=now,
            updated_at=now,
            token_count=20
        ),
        # API source - conflicting data, high authority
        ContextChunk(
            content="Sales revenue for Q1 2024 was $1.5 million with 20% growth",
            source=sources['api'],
            relevance_score=RelevanceScore(score=0.8),
            created_at=now,
            updated_at=now,
            token_count=20
        ),
        # Document source - old data, lower authority
        ContextChunk(
            content="Sales revenue for Q4 2023 was $1.0 million with 10% growth",
            source=sources['document'],
            relevance_score=RelevanceScore(score=0.7),
            created_at=old_time,
            updated_at=old_time,
            token_count=20
        ),
        # Cache source - conflicting data, low authority
        ContextChunk(
            content="Sales revenue for Q1 2024 was $1.3 million with 18% growth",
            source=sources['cache'],
            relevance_score=RelevanceScore(score=0.6),
            created_at=now,
            updated_at=now,
            token_count=20
        )
    ]
    
    return chunks


async def create_contexts_with_conflicts(chunks):
    """Create contexts with conflicting chunks."""
    
    contexts = [
        Context(
            query="What was the Q1 2024 sales revenue and growth?",
            chunks=chunks[:2],  # Database and API
            user_id="user123",
            session_id="session456",
            privacy_level=PrivacyLevel.PRIVATE
        ),
        Context(
            query="What was the Q1 2024 sales revenue and growth?",
            chunks=chunks[2:],  # Document and Cache
            user_id="user123",
            session_id="session456",
            privacy_level=PrivacyLevel.PRIVATE
        )
    ]
    
    return contexts


async def demonstrate_intelligent_fusion():
    """Demonstrate intelligent fusion with conflict resolution."""
    
    print("🚀 Intelligent Multi-Source Context Fusion Demo")
    print("=" * 60)
    
    # Create conflicting data sources and chunks
    sources = await create_conflicting_data_sources()
    chunks = await create_conflicting_chunks(sources)
    contexts = await create_contexts_with_conflicts(chunks)
    
    print(f"\n📊 Created {len(contexts)} contexts with {len(chunks)} chunks")
    print(f"🔍 Sources: {[source.name for source in sources.values()]}")
    
    # Initialize orchestrator
    orchestrator = ContextOrchestrator(
        privacy_level=PrivacyLevel.PRIVATE
    )
    
    print(f"\n🔧 Using fusion engine: {type(orchestrator.fusion_engine).__name__}")
    
    # Test different conflict resolution strategies
    strategies = [
        ConflictResolutionStrategy.HIGHEST_AUTHORITY,
        ConflictResolutionStrategy.NEWEST_DATA,
        ConflictResolutionStrategy.CONSENSUS,
        ConflictResolutionStrategy.WEIGHTED_AVERAGE,
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.value} resolution strategy:")
        print("-" * 40)
        
        # Perform intelligent fusion
        fused_context = await orchestrator.fusion_engine.fuse_contexts(
            contexts,
            strategy='intelligent',
            conflict_resolution=strategy
        )
        
        # Display results
        print(f"✅ Fused context created with {len(fused_context.chunks)} chunks")
        
        if fused_context.fusion_metadata:
            metadata = fused_context.fusion_metadata
            print(f"📈 Fusion confidence: {metadata.fusion_confidence:.2f}")
            print(f"⚡ Processing time: {metadata.processing_time:.3f}s")
            print(f"🔍 Conflicts detected: {metadata.conflict_count}")
            
            if metadata.resolved_conflicts:
                print("\n🔧 Resolved conflicts:")
                for i, conflict in enumerate(metadata.resolved_conflicts, 1):
                    print(f"  {i}. {conflict.conflict_type.value}")
                    print(f"     Description: {conflict.description}")
                    print(f"     Confidence: {conflict.confidence:.2f}")
                    print(f"     Resolution: {conflict.resolution_strategy.value}")
        
        # Show final chunks
        print(f"\n📝 Final chunks after {strategy.value} resolution:")
        for i, chunk in enumerate(fused_context.chunks, 1):
            print(f"  {i}. Source: {chunk.source.name} ({chunk.source.source_type.value})")
            print(f"     Content: {chunk.content}")
            print(f"     Relevance: {chunk.relevance_score.score:.2f}")
            if chunk.metadata.get('resolution_method'):
                print(f"     Resolution: {chunk.metadata['resolution_method']}")


async def demonstrate_conflict_types():
    """Demonstrate different types of conflicts."""
    
    print("\n🔍 Conflict Type Detection Demo")
    print("=" * 40)
    
    orchestrator = ContextOrchestrator()
    
    # 1. Content Contradiction
    print("\n1️⃣ Content Contradiction:")
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
    
    fused = await orchestrator.fusion_engine.fuse_contexts([context1, context2])
    if fused.fusion_metadata and fused.fusion_metadata.resolved_conflicts:
        for conflict in fused.fusion_metadata.resolved_conflicts:
            if conflict.conflict_type == ConflictType.CONTENT_CONTRADICTION:
                print(f"   ✅ Detected: {conflict.description}")
    
    # 2. Temporal Conflict
    print("\n2️⃣ Temporal Conflict:")
    now = datetime.utcnow()
    old_time = now - timedelta(days=60)
    
    context1 = Context(
        query="What is the current market share?",
        chunks=[
            ContextChunk(
                content="Market share is 15% as of Q1 2024",
                source=ContextSource(name="Recent API", source_type=SourceType.API),
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
                source=ContextSource(name="Old Document", source_type=SourceType.DOCUMENT),
                created_at=old_time,
                updated_at=old_time,
                relevance_score=RelevanceScore(score=0.7)
            )
        ],
        user_id="user123"
    )
    
    fused = await orchestrator.fusion_engine.fuse_contexts([context1, context2])
    if fused.fusion_metadata and fused.fusion_metadata.resolved_conflicts:
        for conflict in fused.fusion_metadata.resolved_conflicts:
            if conflict.conflict_type == ConflictType.TEMPORAL_CONFLICT:
                print(f"   ✅ Detected: {conflict.description}")
    
    # 3. Semantic Conflict
    print("\n3️⃣ Semantic Conflict:")
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
    
    fused = await orchestrator.fusion_engine.fuse_contexts([context1, context2])
    if fused.fusion_metadata and fused.fusion_metadata.resolved_conflicts:
        for conflict in fused.fusion_metadata.resolved_conflicts:
            if conflict.conflict_type == ConflictType.SEMANTIC_CONFLICT:
                print(f"   ✅ Detected: {conflict.description}")


async def demonstrate_fusion_confidence():
    """Demonstrate fusion confidence calculation."""
    
    print("\n📊 Fusion Confidence Analysis")
    print("=" * 35)
    
    orchestrator = ContextOrchestrator()
    
    # Create contexts with varying levels of conflict
    sources = await create_conflicting_data_sources()
    chunks = await create_conflicting_chunks(sources)
    contexts = await create_contexts_with_conflicts(chunks)
    
    # Test fusion with different strategies
    strategies = [
        ConflictResolutionStrategy.HIGHEST_AUTHORITY,
        ConflictResolutionStrategy.CONSENSUS,
        ConflictResolutionStrategy.WEIGHTED_AVERAGE,
    ]
    
    for strategy in strategies:
        fused = await orchestrator.fusion_engine.fuse_contexts(
            contexts,
            conflict_resolution=strategy
        )
        
        if fused.fusion_metadata:
            confidence = fused.fusion_metadata.fusion_confidence
            conflict_count = fused.fusion_metadata.conflict_count
            
            print(f"\n🎯 {strategy.value}:")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Conflicts: {conflict_count}")
            print(f"   Final chunks: {len(fused.chunks)}")
            
            # Analyze confidence factors
            if confidence < 0.8:
                print(f"   ⚠️  Lower confidence due to conflicts")
            else:
                print(f"   ✅ High confidence fusion")


async def main():
    """Main demonstration function."""
    
    print("🎯 Ragify Intelligent Context Fusion Demo")
    print("=" * 50)
    print("This demo showcases advanced conflict detection and resolution")
    print("capabilities in multi-source context fusion.\n")
    
    try:
        # Demonstrate intelligent fusion with different strategies
        await demonstrate_intelligent_fusion()
        
        # Demonstrate conflict type detection
        await demonstrate_conflict_types()
        
        # Demonstrate fusion confidence analysis
        await demonstrate_fusion_confidence()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Multi-source context fusion")
        print("   ✅ Advanced conflict detection")
        print("   ✅ Multiple resolution strategies")
        print("   ✅ Fusion confidence calculation")
        print("   ✅ Comprehensive metadata tracking")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
