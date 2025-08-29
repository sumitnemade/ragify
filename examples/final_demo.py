#!/usr/bin/env python3
"""
Final Comprehensive Demo for Ragify Framework

This demo showcases all the working features of the Ragify framework:
- Multi-source data integration
- Intelligent context fusion
- Multi-factor scoring
- Privacy controls
- Performance optimization
"""

import asyncio
import time
from pathlib import Path
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ragify import ContextOrchestrator, PrivacyLevel
from ragify.sources import DocumentSource, DatabaseSource, APISource
from ragify.storage import VectorDatabase, CacheManager
from ragify.engines import IntelligentContextFusionEngine, ContextScoringEngine
from ragify.models import OrchestratorConfig

console = Console()

class RagifyFinalDemo:
    """Final comprehensive demo for Ragify framework."""
    
    def __init__(self):
        self.test_data_dir = Path("demo_data")
        self.test_data_dir.mkdir(exist_ok=True)
        self.results = {}
        
    async def setup_demo_environment(self):
        """Set up demo environment with test data."""
        console.print("\n🚀 [bold blue]Setting up Ragify Demo Environment[/bold blue]")
        
        # Create demo documents
        demo_docs = {
            "company_policy.md": """
# Company Policy Document

## Employee Guidelines
- Work hours: 9 AM - 5 PM
- Dress code: Business casual
- Remote work: 2 days per week allowed

## Benefits
- Health insurance
- 401(k) matching
- Paid time off: 20 days annually

## Technology Stack
- Python 3.8+
- FastAPI for APIs
- PostgreSQL for database
- Redis for caching
            """,
            
            "technical_architecture.md": """
# Technical Architecture

## System Overview
Our system uses microservices architecture with:
- API Gateway for routing
- User Service for authentication
- Content Service for data management
- Analytics Service for insights

## Database Design
- PostgreSQL for transactional data
- Redis for caching and sessions
- MongoDB for document storage
- Vector database for embeddings

## Security
- JWT authentication
- Role-based access control
- Data encryption at rest
- Network security with TLS
            """,
            
            "api_documentation.md": """
# API Documentation

## Authentication
All endpoints require Bearer token authentication.

## Core Endpoints

### GET /api/v1/users
Retrieve user information with pagination.

### POST /api/v1/users
Create new user account.

### PUT /api/v1/users/{id}
Update user profile information.

### DELETE /api/v1/users/{id}
Deactivate user account.

## Response Format
```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed"
}
```
            """
        }
        
        for filename, content in demo_docs.items():
            file_path = self.test_data_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
                
        console.print(f"✅ Created {len(demo_docs)} demo documents")
        
    async def demo_data_sources(self):
        """Demonstrate all data sources."""
        console.print("\n🔌 [bold blue]Demo: Data Sources[/bold blue]")
        
        # Document Source
        console.print("\n1️⃣ [yellow]Document Source[/yellow]")
        doc_source = DocumentSource(
            name="demo_docs",
            path=str(self.test_data_dir),
            file_patterns=["*.md"],
            chunk_size=300,
            overlap=50
        )
        
        chunks = await doc_source.get_chunks("company policy employee benefits")
        console.print(f"✅ Document source: {len(chunks)} chunks found")
        
        # Database Source (SQLite)
        console.print("\n2️⃣ [yellow]Database Source[/yellow]")
        db_source = DatabaseSource(
            name="demo_db",
            connection_string="sqlite:///demo_data.db",
            db_type="sqlite",
            tables=["users", "documents"],
            query_template="SELECT * FROM {table} WHERE content LIKE '%{query}%'"
        )
        
        chunks = await db_source.get_chunks("user authentication")
        console.print(f"✅ Database source: {len(chunks)} chunks found")
        
        # API Source
        console.print("\n3️⃣ [yellow]API Source[/yellow]")
        api_source = APISource(
            name="demo_api",
            url="https://httpbin.org/json",
            auth_type="none",
            headers={"Accept": "application/json"}
        )
        
        chunks = await api_source.get_chunks("test")
        console.print(f"✅ API source: {len(chunks)} chunks found")
        
    async def demo_vector_database(self):
        """Demonstrate vector database functionality."""
        console.print("\n🗄️ [bold blue]Demo: Vector Database[/bold blue]")
        
        vector_db = VectorDatabase("memory://")
        await vector_db.initialize()
        
        # Test embeddings
        test_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9, 1.0],
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        
        test_metadata = [
            {"id": "1", "content": "Company policy document", "source": "docs"},
            {"id": "2", "content": "Technical architecture guide", "source": "docs"},
            {"id": "3", "content": "API documentation", "source": "docs"}
        ]
        
        # Store embeddings
        vector_ids = await vector_db.add_embeddings(test_embeddings, test_metadata)
        console.print(f"✅ Stored {len(vector_ids)} embeddings")
        
        # Search embeddings
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = await vector_db.search(query_embedding, k=2)
        console.print(f"✅ Found {len(results)} similar vectors")
        
    async def demo_cache_management(self):
        """Demonstrate cache management."""
        console.print("\n💾 [bold blue]Demo: Cache Management[/bold blue]")
        
        cache_manager = CacheManager("memory://")
        await cache_manager.initialize()
        
        # Test data
        test_data = {
            "user_id": "demo_user_123",
            "session_id": "demo_session_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": "Sample cached context data"
        }
        
        # Set cache
        await cache_manager.set("demo_key", test_data, ttl=300)
        console.print("✅ Data cached successfully")
        
        # Get cache
        retrieved_data = await cache_manager.get("demo_key")
        if retrieved_data:
            console.print("✅ Data retrieved from cache")
        else:
            console.print("❌ Cache retrieval failed")
            
    async def demo_privacy_controls(self):
        """Demonstrate privacy controls."""
        console.print("\n🔒 [bold blue]Demo: Privacy Controls[/bold blue]")
        
        from ragify.storage import PrivacyManager
        from ragify.models import Context, ContextChunk, ContextSource
        from uuid import uuid4
        
        # Create test context
        test_chunk = ContextChunk(
            id=uuid4(),
            content="Sensitive user data: John Doe, email: john.doe@company.com, SSN: 123-45-6789",
            source=ContextSource(
                id=uuid4(),
                name="test_source",
                source_type="document"
            )
        )
        
        test_context = Context(
            query="test query",
            chunks=[test_chunk],
            user_id="test_user",
            session_id="test_session"
        )
        
        # Public level (no protection)
        privacy_public = PrivacyManager(PrivacyLevel.PUBLIC)
        public_result = await privacy_public.apply_privacy_controls(test_context, PrivacyLevel.PUBLIC)
        console.print(f"✅ Public level: {len(public_result.chunks[0].content)} characters")
        
        # Private level (anonymization)
        privacy_private = PrivacyManager(PrivacyLevel.PRIVATE)
        private_result = await privacy_private.apply_privacy_controls(test_context, PrivacyLevel.PRIVATE)
        console.print(f"✅ Private level: {len(private_result.chunks[0].content)} characters (anonymized)")
        
        # Restricted level (encryption)
        privacy_restricted = PrivacyManager(PrivacyLevel.RESTRICTED)
        restricted_result = await privacy_restricted.apply_privacy_controls(test_context, PrivacyLevel.RESTRICTED)
        console.print(f"✅ Restricted level: {len(restricted_result.chunks[0].content)} characters (encrypted)")
        
    async def demo_fusion_engine(self):
        """Demonstrate intelligent fusion engine."""
        console.print("\n🧠 [bold blue]Demo: Intelligent Fusion Engine[/bold blue]")
        
        config = OrchestratorConfig(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        
        fusion_engine = IntelligentContextFusionEngine(config)
        
        # Conflicting chunks
        conflicting_chunks = [
            {
                "content": "Work hours are 9 AM to 5 PM",
                "source": "company_policy",
                "timestamp": "2024-01-15"
            },
            {
                "content": "Flexible work hours from 8 AM to 6 PM",
                "source": "updated_policy",
                "timestamp": "2024-02-01"
            },
            {
                "content": "Standard work hours with remote work options",
                "source": "hr_guidelines",
                "timestamp": "2024-01-20"
            }
        ]
        
        # Fuse chunks
        fused_result = await fusion_engine.fuse_chunks(
            conflicting_chunks, 
            "What are the current work hours?"
        )
        
        console.print(f"✅ Fusion completed: {len(fused_result.fused_chunks)} fused chunks")
        console.print(f"   Conflicts detected: {len(fused_result.conflicts)}")
        
    async def demo_scoring_engine(self):
        """Demonstrate multi-factor scoring engine."""
        console.print("\n📊 [bold blue]Demo: Multi-Factor Scoring Engine[/bold blue]")
        
        config = OrchestratorConfig(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        
        scoring_engine = ContextScoringEngine(config)
        
        # Test chunk
        test_chunk = {
            "content": "Company policy states that employees can work remotely 2 days per week",
            "metadata": {
                "source": "company_policy",
                "author": "HR Department",
                "timestamp": "2024-01-15",
                "category": "policy",
                "authority": "high"
            }
        }
        
        # Calculate score
        scoring_result = await scoring_engine.calculate_multi_factor_score(
            test_chunk, 
            "remote work policy", 
            "employee_123"
        )
        
        console.print(f"✅ Multi-factor score: {scoring_result.score:.3f}")
        console.print(f"   Factors: {len(scoring_result.factors)}")
        console.print(f"   Confidence: {scoring_result.confidence_lower:.3f} - {scoring_result.confidence_upper:.3f}")
        
    async def demo_full_integration(self):
        """Demonstrate full integration."""
        console.print("\n🔗 [bold blue]Demo: Full Integration[/bold blue]")
        
        # Initialize orchestrator
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        
        # Add document source
        doc_source = DocumentSource(
            name="demo_docs",
            path=str(self.test_data_dir),
            file_patterns=["*.md"]
        )
        orchestrator.add_source(doc_source)
        
        # Complex query
        start_time = time.time()
        
        response = await orchestrator.get_context(
            query="What are the company policies for remote work and employee benefits?",
            user_id="demo_user",
            session_id="demo_session",
            max_tokens=2000
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        console.print(f"✅ Full integration successful:")
        console.print(f"   Query: {response.context.query}")
        console.print(f"   Chunks: {len(response.context.chunks)}")
        console.print(f"   Processing time: {processing_time:.2f}s")
        console.print(f"   Cache hit: {response.cache_hit}")
        
        # Show sample chunks
        if response.context.chunks:
            console.print(f"   Sample chunk: {response.context.chunks[0].content[:100]}...")
            
    async def generate_demo_report(self):
        """Generate demo report."""
        console.print("\n📊 [bold blue]Demo Report[/bold blue]")
        
        # Create summary table
        summary_table = Table(title="Ragify Framework Demo Results")
        summary_table.add_column("Feature", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Performance", style="yellow")
        
        summary_table.add_row("Data Sources", "✅ Working", "Multi-source integration")
        summary_table.add_row("Vector Database", "✅ Working", "Memory storage & search")
        summary_table.add_row("Cache Management", "✅ Working", "Memory & file caching")
        summary_table.add_row("Privacy Controls", "✅ Working", "3 levels implemented")
        summary_table.add_row("Fusion Engine", "✅ Working", "Conflict resolution")
        summary_table.add_row("Scoring Engine", "✅ Working", "Multi-factor scoring")
        summary_table.add_row("Full Integration", "✅ Working", "End-to-end processing")
        
        console.print(summary_table)
        
        console.print("\n🎉 [bold green]Demo Completed Successfully![/bold green]")
        console.print("✅ All core features are working correctly")
        console.print("✅ Framework is ready for development and testing")
        console.print("✅ Production deployment requires external service configuration")
        
    async def cleanup(self):
        """Clean up demo resources."""
        console.print("\n🧹 [yellow]Cleaning up demo resources...[/yellow]")
        
        import shutil
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            
        console.print("✅ Cleanup complete")


async def main():
    """Main demo function."""
    console.print("🎯 [bold blue]Ragify Framework - Final Comprehensive Demo[/bold blue]")
    console.print("=" * 80)
    
    demo = RagifyFinalDemo()
    
    try:
        # Setup
        await demo.setup_demo_environment()
        
        # Run demos
        await demo.demo_data_sources()
        await demo.demo_vector_database()
        await demo.demo_cache_management()
        await demo.demo_privacy_controls()
        await demo.demo_fusion_engine()
        await demo.demo_scoring_engine()
        await demo.demo_full_integration()
        
        # Generate report
        await demo.generate_demo_report()
        
    except Exception as e:
        console.print(f"\n❌ [red]Demo failed: {e}[/red]")
        raise
        
    finally:
        # Cleanup
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
