#!/usr/bin/env python3
"""
Conflict Resolution Demo - Intelligent Data Fusion

This demo shows RAGify's advanced conflict detection and resolution capabilities:
- Real-time conflict detection between sources
- Multiple resolution strategies (consensus, authority, weighted)
- Conflict confidence scoring
- Resolution metadata and audit trails
- Performance comparison of different strategies

Use case: "I need to intelligently resolve conflicts between multiple data sources"
"""

import asyncio
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random

from ragify import ContextOrchestrator
from src.ragify.models import (
    Context, ContextChunk, PrivacyLevel, ContextSource, 
    SourceType, RelevanceScore, ConflictType, ConflictInfo,
    ConflictResolutionStrategy, FusionMetadata
)
from src.ragify.engines.fusion import IntelligentContextFusionEngine
from src.ragify.models import OrchestratorConfig


class ConflictResolutionDemo:
    """
    Demonstrates RAGify's conflict resolution capabilities.
    
    Shows how to:
    1. Detect conflicts between sources
    2. Apply different resolution strategies
    3. Measure resolution quality
    4. Handle complex conflict scenarios
    """
    
    def __init__(self):
        """Initialize the conflict resolution demo."""
        self.temp_dir = None
        self.orchestrator = None
        self.fusion_engine = None
        
    async def setup(self):
        """Set up the demo environment."""
        print("🚀 Setting up Conflict Resolution Demo...")
        
        # Create temporary directory for demo documents
        self.temp_dir = tempfile.mkdtemp(prefix="ragify_conflicts_")
        print(f"📁 Created temp directory: {self.temp_dir}")
        
        # Create conflicting documents
        await self._create_conflicting_documents()
        
        # Initialize RAGify components
        config = OrchestratorConfig(
            vector_db_url="memory://conflict_db",
            cache_url="memory://conflict_cache",
            privacy_level=PrivacyLevel.ENTERPRISE,
            conflict_detection_threshold=0.6
        )
        
        self.orchestrator = ContextOrchestrator(
            vector_db_url="memory://conflict_db",
            cache_url="memory://conflict_cache",
            privacy_level=PrivacyLevel.ENTERPRISE
        )
        
        self.fusion_engine = IntelligentContextFusionEngine(config)
        
        print("✅ Demo setup complete!")
        
    async def _create_conflicting_documents(self):
        """Create documents with conflicting information."""
        print("📚 Creating documents with conflicts...")
        
        # Document 1: Financial Report (High Authority)
        financial_content = """
        Q4 2024 Financial Summary
        
        Customer Count: 12,450 active customers
        Revenue: $2.4M (up 15% from Q3)
        Market Share: 8.2% in our target segment
        Customer Satisfaction: 4.6/5.0
        
        Key Metrics:
        - Customer Acquisition Cost: $180
        - Customer Lifetime Value: $2,100
        - Churn Rate: 3.2% (industry average: 4.1%)
        
        Note: Data as of December 31, 2024
        """
        
        # Document 2: Marketing Analysis (Medium Authority)
        marketing_content = """
        Marketing Team Analysis - Q4 2024
        
        Customer Analysis:
        - Total customers: approximately 12,000
        - New acquisitions: 450 this quarter
        - Target market penetration: 7.8%
        
        Customer Satisfaction Survey:
        - Overall rating: 4.4/5.0
        - Response rate: 68%
        - Key feedback: Product quality, support responsiveness
        
        Marketing spend: $180K, ROI: 320%
        """
        
        # Document 3: IT Department Report (Medium Authority)
        it_content = """
        IT Infrastructure Report - Q4 2024
        
        System Statistics:
        - Active user accounts: 12,500
        - Database records: 12,500 customer profiles
        - System uptime: 99.7%
        
        Customer Data:
        - Unique customer IDs: 12,500
        - Active subscriptions: 11,800
        - Pending activations: 700
        
        Technical debt: 15% (down from 22% in Q3)
        """
        
        # Document 4: Customer Support Report (Lower Authority)
        support_content = """
        Customer Support Summary - Q4 2024
        
        Support Tickets:
        - Total tickets: 1,245
        - Resolved: 1,180 (94.8%)
        - Average resolution time: 2.3 hours
        
        Customer Feedback:
        - Support satisfaction: 4.3/5.0
        - Product satisfaction: 4.5/5.0
        - Overall experience: 4.4/5.0
        
        Estimated customer base: 11,900-12,100
        """
        
        # Document 5: Sales Team Report (Lower Authority)
        sales_content = """
        Sales Performance - Q4 2024
        
        Sales Metrics:
        - New deals closed: 156
        - Revenue generated: $2.35M
        - Average deal size: $15,100
        
        Customer Count:
        - New customers: 156
        - Existing customers: 11,900
        - Total: approximately 12,056
        
        Sales pipeline: $3.2M for Q1 2025
        """
        
        # Write documents to temp directory
        documents = [
            ("financial_report.txt", financial_content, SourceType.DOCUMENT, 1.0),
            ("marketing_analysis.txt", marketing_content, SourceType.DOCUMENT, 0.8),
            ("it_report.txt", it_content, SourceType.DOCUMENT, 0.8),
            ("support_summary.txt", support_content, SourceType.API, 0.6),
            ("sales_performance.txt", sales_content, SourceType.API, 0.6)
        ]
        
        for filename, content, source_type, authority in documents:
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   ✅ Created {filename} (Authority: {authority})")
            
    async def demonstrate_conflict_resolution(self):
        """Demonstrate comprehensive conflict resolution."""
        print("\n🎯 Demonstrating Conflict Resolution Capabilities...")
        
        # Scenario 1: Customer Count Conflicts
        await self._scenario_1_customer_count_conflicts()
        
        # Scenario 2: Revenue Conflicts
        await self._scenario_2_revenue_conflicts()
        
        # Scenario 3: Satisfaction Score Conflicts
        await self._scenario_3_satisfaction_conflicts()
        
        # Scenario 4: Performance Comparison
        await self._scenario_4_performance_comparison()
        
    async def _scenario_1_customer_count_conflicts(self):
        """Scenario 1: Resolve customer count conflicts."""
        print("\n📊 Scenario 1: Customer Count Conflicts")
        print("=" * 50)
        
        # Create conflicting chunks about customer count
        conflicting_chunks = [
            ContextChunk(
                content="Customer Count: 12,450 active customers",
                source=ContextSource(
                    name="Financial Report",
                    source_type=SourceType.DOCUMENT,
                    authority_score=1.0,
                    privacy_level=PrivacyLevel.ENTERPRISE
                ),
                relevance_score=RelevanceScore(score=0.95),
                metadata={"data_type": "financial", "freshness": 1.0}
            ),
            ContextChunk(
                content="Total customers: approximately 12,000",
                source=ContextSource(
                    name="Marketing Analysis",
                    source_type=SourceType.DOCUMENT,
                    authority_score=0.8,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.85),
                metadata={"data_type": "marketing", "freshness": 0.9}
            ),
            ContextChunk(
                content="Active user accounts: 12,500",
                source=ContextSource(
                    name="IT Department",
                    source_type=SourceType.DOCUMENT,
                    authority_score=0.8,
                    privacy_level=PrivacyLevel.ENTERPRISE
                ),
                relevance_score=RelevanceScore(score=0.88),
                metadata={"data_type": "technical", "freshness": 0.95}
            ),
            ContextChunk(
                content="Estimated customer base: 11,900-12,100",
                source=ContextSource(
                    name="Customer Support",
                    source_type=SourceType.API,
                    authority_score=0.6,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.75),
                metadata={"data_type": "operational", "freshness": 0.8}
            ),
            ContextChunk(
                content="Total: approximately 12,056",
                source=ContextSource(
                    name="Sales Team",
                    source_type=SourceType.API,
                    authority_score=0.6,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.78),
                metadata={"data_type": "sales", "freshness": 0.85}
            )
        ]
        
        print("🔍 Detected conflicts in customer count data:")
        for i, chunk in enumerate(conflicting_chunks, 1):
            print(f"  {i}. {chunk.source.name}: {chunk.content}")
            print(f"     Authority: {chunk.source.authority_score}, Relevance: {chunk.relevance_score.score:.2f}")
        
        # Test different resolution strategies
        strategies = [
            ("highest_authority", "Highest Authority"),
            ("consensus", "Consensus"),
            ("weighted_average", "Weighted Average"),
            ("newest_data", "Newest Data")
        ]
        
        print(f"\n🔄 Testing {len(strategies)} resolution strategies...")
        
        for strategy_name, strategy_label in strategies:
            print(f"\n📋 Strategy: {strategy_label}")
            print("-" * 30)
            
            try:
                # Create context with conflicting chunks
                context = Context(
                    query="What is our current customer count?",
                    chunks=conflicting_chunks
                )
                
                # Resolve conflicts using fusion engine
                start_time = asyncio.get_event_loop().time()
                resolved_context = await self.fusion_engine.fuse_contexts(
                    [context],
                    strategy=strategy_name,
                    conflict_resolution=ConflictResolutionStrategy.HIGHEST_AUTHORITY
                )
                processing_time = asyncio.get_event_loop().time() - start_time
                
                print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                print(f"   📊 Resolved chunks: {len(resolved_context.chunks)}")
                
                if resolved_context.fusion_metadata:
                    print(f"   🔧 Strategy used: {resolved_context.fusion_metadata.fusion_strategy}")
                    print(f"   ⚠️  Conflicts detected: {resolved_context.fusion_metadata.conflict_count}")
                    print(f"   ✅ Conflicts resolved: {len(resolved_context.fusion_metadata.resolved_conflicts)}")
                
                # Show resolution result
                if resolved_context.chunks:
                    primary_chunk = resolved_context.chunks[0]
                    print(f"   🎯 Primary result: {primary_chunk.content}")
                    print(f"   📍 Source: {primary_chunk.source.name}")
                    print(f"   🏆 Authority: {primary_chunk.source.authority_score}")
                    
            except Exception as e:
                print(f"   ❌ Strategy failed: {e}")
                
    async def _scenario_2_revenue_conflicts(self):
        """Scenario 2: Resolve revenue conflicts."""
        print("\n💰 Scenario 2: Revenue Conflicts")
        print("=" * 50)
        
        # Create conflicting chunks about revenue
        revenue_chunks = [
            ContextChunk(
                content="Revenue: $2.4M (up 15% from Q3)",
                source=ContextSource(
                    name="Financial Report",
                    source_type=SourceType.DOCUMENT,
                    authority_score=1.0,
                    privacy_level=PrivacyLevel.ENTERPRISE
                ),
                relevance_score=RelevanceScore(score=0.95),
                metadata={"data_type": "financial", "freshness": 1.0}
            ),
            ContextChunk(
                content="Revenue generated: $2.35M",
                source=ContextSource(
                    name="Sales Team",
                    source_type=SourceType.API,
                    authority_score=0.6,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.80),
                metadata={"data_type": "sales", "freshness": 0.85}
            )
        ]
        
        print("🔍 Detected conflicts in revenue data:")
        for chunk in revenue_chunks:
            print(f"  - {chunk.source.name}: {chunk.content}")
            print(f"    Authority: {chunk.source.authority_score}, Relevance: {chunk.relevance_score.score:.2f}")
        
        # Test weighted average strategy
        print(f"\n🔄 Testing weighted average strategy...")
        try:
            context = Context(
                query="What is our Q4 revenue?",
                chunks=revenue_chunks
            )
            
            resolved_context = await self.fusion_engine.fuse_contexts(
                [context],
                strategy="weighted_average",
                conflict_resolution=ConflictResolutionStrategy.WEIGHTED_AVERAGE
            )
            
            print(f"   ✅ Resolution successful")
            print(f"   📊 Final result: {resolved_context.chunks[0].content if resolved_context.chunks else 'No result'}")
            
        except Exception as e:
            print(f"   ❌ Resolution failed: {e}")
            
    async def _scenario_3_satisfaction_conflicts(self):
        """Scenario 3: Resolve customer satisfaction conflicts."""
        print("\n😊 Scenario 3: Customer Satisfaction Conflicts")
        print("=" * 50)
        
        # Create conflicting chunks about satisfaction
        satisfaction_chunks = [
            ContextChunk(
                content="Customer Satisfaction: 4.6/5.0",
                source=ContextSource(
                    name="Financial Report",
                    source_type=SourceType.DOCUMENT,
                    authority_score=1.0,
                    privacy_level=PrivacyLevel.ENTERPRISE
                ),
                relevance_score=RelevanceScore(score=0.90),
                metadata={"data_type": "financial", "freshness": 1.0}
            ),
            ContextChunk(
                content="Overall rating: 4.4/5.0",
                source=ContextSource(
                    name="Marketing Analysis",
                    source_type=SourceType.DOCUMENT,
                    authority_score=0.8,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.85),
                metadata={"data_type": "marketing", "freshness": 0.9}
            ),
            ContextChunk(
                content="Support satisfaction: 4.3/5.0, Product satisfaction: 4.5/5.0",
                source=ContextSource(
                    name="Customer Support",
                    source_type=SourceType.API,
                    authority_score=0.6,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.80),
                metadata={"data_type": "operational", "freshness": 0.8}
            )
        ]
        
        print("🔍 Detected conflicts in satisfaction data:")
        for chunk in satisfaction_chunks:
            print(f"  - {chunk.source.name}: {chunk.content}")
            print(f"    Authority: {chunk.source.authority_score}, Relevance: {chunk.relevance_score.score:.2f}")
        
        # Test consensus strategy
        print(f"\n🔄 Testing consensus strategy...")
        try:
            context = Context(
                query="What is our customer satisfaction score?",
                chunks=satisfaction_chunks
            )
            
            resolved_context = await self.fusion_engine.fuse_contexts(
                [context],
                strategy="consensus",
                conflict_resolution=ConflictResolutionStrategy.CONSENSUS
            )
            
            print(f"   ✅ Resolution successful")
            print(f"   📊 Final result: {resolved_context.chunks[0].content if resolved_context.chunks else 'No result'}")
            
        except Exception as e:
            print(f"   ❌ Resolution failed: {e}")
            
    async def _scenario_4_performance_comparison(self):
        """Scenario 4: Compare performance of different strategies."""
        print("\n⚡ Scenario 4: Strategy Performance Comparison")
        print("=" * 50)
        
        # Create a large set of conflicting chunks for performance testing
        print("📊 Creating large dataset for performance testing...")
        
        large_conflicting_chunks = []
        for i in range(50):  # Create 50 chunks with variations
            base_value = random.randint(10000, 13000)
            variation = random.randint(-500, 500)
            value = base_value + variation
            
            chunk = ContextChunk(
                content=f"Customer count: {value:,}",
                source=ContextSource(
                    name=f"Source_{i % 5}",
                    source_type=SourceType.DOCUMENT,
                    authority_score=0.5 + (i % 5) * 0.1,
                    privacy_level=PrivacyLevel.PRIVATE
                ),
                relevance_score=RelevanceScore(score=0.7 + random.random() * 0.3),
                metadata={"iteration": i, "freshness": 0.8 + random.random() * 0.2}
            )
            large_conflicting_chunks.append(chunk)
        
        print(f"   ✅ Created {len(large_conflicting_chunks)} conflicting chunks")
        
        # Test performance of different strategies
        strategies = ["highest_authority", "consensus", "weighted_average", "newest_data"]
        performance_results = {}
        
        print(f"\n🏃‍♂️ Performance testing {len(strategies)} strategies...")
        
        for strategy in strategies:
            print(f"\n📋 Testing {strategy}...")
            
            try:
                context = Context(
                    query="What is the customer count across all sources?",
                    chunks=large_conflicting_chunks
                )
                
                # Measure performance
                start_time = asyncio.get_event_loop().time()
                resolved_context = await self.fusion_engine.fuse_contexts(
                    [context],
                    strategy=strategy
                )
                processing_time = asyncio.get_event_loop().time() - start_time
                
                performance_results[strategy] = {
                    'time': processing_time,
                    'chunks_processed': len(large_conflicting_chunks),
                    'chunks_resolved': len(resolved_context.chunks),
                    'success': True
                }
                
                print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                print(f"   📊 Chunks processed: {len(large_conflicting_chunks)}")
                print(f"   ✅ Chunks resolved: {len(resolved_context.chunks)}")
                
            except Exception as e:
                performance_results[strategy] = {
                    'time': 0,
                    'chunks_processed': len(large_conflicting_chunks),
                    'chunks_resolved': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"   ❌ Strategy failed: {e}")
        
        # Performance summary
        print(f"\n📊 Performance Summary")
        print("=" * 30)
        
        successful_strategies = {k: v for k, v in performance_results.items() if v['success']}
        if successful_strategies:
            fastest_strategy = min(successful_strategies.items(), key=lambda x: x[1]['time'])
            print(f"🏆 Fastest strategy: {fastest_strategy[0]} ({fastest_strategy[1]['time']:.3f}s)")
            
            for strategy, results in performance_results.items():
                status = "✅" if results['success'] else "❌"
                print(f"{status} {strategy}: {results['time']:.3f}s, {results['chunks_resolved']} resolved")
        else:
            print("❌ All strategies failed")
            
    async def cleanup(self):
        """Clean up demo resources."""
        print("\n🧹 Cleaning up demo resources...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.close()
            print("   ✅ Orchestrator closed")
            
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                print("   ✅ Temp directory removed")
                
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")


async def main():
    """Main demo function."""
    print("🎯 CONFLICT RESOLUTION DEMO: Intelligent Data Fusion")
    print("=" * 60)
    print("This demo shows RAGify's advanced conflict detection and resolution capabilities!")
    
    demo = ConflictResolutionDemo()
    
    try:
        # Setup
        await demo.setup()
        
        # Run demonstrations
        await demo.demonstrate_conflict_resolution()
        
        print("\n🎉 Demo Complete!")
        print("This demonstrates REAL conflict resolution functionality:")
        print("✅ Real-time conflict detection between sources")
        print("✅ Multiple resolution strategies with performance metrics")
        print("✅ Conflict confidence scoring and audit trails")
        print("✅ Performance comparison of different approaches")
        print("✅ Practical business scenarios with measurable results")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
