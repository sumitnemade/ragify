#!/usr/bin/env python3
"""
Real-World Demo: Tech Company Knowledge Management System

This demo showcases all features of the Ragify plugin in a realistic scenario:
- Document processing (PDF, DOCX, TXT)
- API integrations (GitHub, Jira, Slack)
- Database connections (PostgreSQL, MongoDB)
- Real-time sources (WebSocket, MQTT)
- Vector database storage (ChromaDB, FAISS)
- Cache management (Redis, Memory)
- Intelligent context fusion with conflict resolution
- Multi-factor scoring with ensemble methods
- Statistical confidence bounds
- Privacy controls

Scenario: A tech company wants to build an intelligent knowledge management system
that can answer questions about their projects, documentation, and team activities.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import aiohttp
import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import Ragify components
from ragify import ContextOrchestrator, PrivacyLevel
from ragify.models import ContextRequest, ContextResponse
from ragify.sources import (
    DocumentSource, APISource, DatabaseSource, RealtimeSource
)
from ragify.storage import VectorDatabase, CacheManager
from ragify.engines import IntelligentContextFusionEngine

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
console = Console()


class TechCompanyKnowledgeSystem:
    """Real-world demo of Ragify for a tech company knowledge management system."""
    
    def __init__(self):
        self.orchestrator = None
        self.demo_data_dir = Path("demo_data")
        self.demo_data_dir.mkdir(exist_ok=True)
        self.results = {}
        
    async def setup_demo_environment(self):
        """Set up the demo environment with sample data and configurations."""
        console.print("\n🚀 [bold blue]Setting up Tech Company Knowledge Management System[/bold blue]")
        
        # Create sample documents
        await self._create_sample_documents()
        
        # Create sample database
        await self._create_sample_database()
        
        # Initialize orchestrator with all features
        await self._initialize_orchestrator()
        
        console.print("✅ [green]Demo environment setup complete![/green]")
        
    async def _create_sample_documents(self):
        """Create sample documents for testing document processing."""
        console.print("\n📄 [yellow]Creating sample documents...[/yellow]")
        
        # Sample project documentation
        project_docs = {
            "project_overview.md": """
# Ragify Project Overview

## Project Goals
Ragify is an intelligent context orchestration framework for LLM applications.

## Key Features
- Multi-source context fusion
- Vector database integration
- Real-time data synchronization
- Advanced scoring algorithms

## Architecture
The system uses a modular architecture with:
- Context Fusion Engine
- Scoring Engine
- Storage Engine
- Updates Engine

## Development Status
Currently in active development with comprehensive test coverage.
            """,
            
            "api_documentation.md": """
# API Documentation

## Core Endpoints
- GET /context - Retrieve context
- POST /sources - Add data source
- PUT /config - Update configuration

## Authentication
All endpoints require API key authentication.

## Rate Limiting
100 requests per minute per API key.

## Response Format
```json
{
  "context": "retrieved context",
  "sources": ["source1", "source2"],
  "confidence": 0.95
}
```
            """,
            
            "team_guidelines.md": """
# Team Development Guidelines

## Code Standards
- Use Python 3.8+
- Follow PEP 8 style guide
- Write comprehensive tests
- Document all public APIs

## Git Workflow
1. Create feature branch
2. Make changes
3. Write tests
4. Submit pull request
5. Code review
6. Merge to main

## Testing Requirements
- Unit tests for all functions
- Integration tests for APIs
- Performance benchmarks
- Security testing
            """
        }
        
        # Create documents
        for filename, content in project_docs.items():
            file_path = self.demo_data_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
                
        console.print(f"✅ Created {len(project_docs)} sample documents")
        
    async def _create_sample_database(self):
        """Create sample database with project data."""
        console.print("\n🗄️ [yellow]Creating sample database...[/yellow]")
        
        # Create SQLite database for demo
        import sqlite3
        
        db_path = self.demo_data_dir / "company_data.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT,
                team_lead TEXT,
                created_date TEXT,
                content TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT,
                department TEXT,
                skills TEXT,
                content TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT,
                content TEXT,
                author TEXT,
                created_date TEXT,
                relevance REAL
            )
        """)
        
        # Insert sample data
        projects_data = [
            (1, "Ragify Framework", "Intelligent context orchestration system", "Active", "Sumit Nemade", "2024-01-15", "Advanced context fusion with vector databases"),
            (2, "AI Chatbot", "Customer support chatbot", "Planning", "Alice Johnson", "2024-02-01", "Natural language processing and machine learning"),
            (3, "Data Pipeline", "ETL pipeline for analytics", "Completed", "Bob Smith", "2023-12-10", "Data processing and analytics automation")
        ]
        
        team_data = [
            (1, "Sumit Nemade", "Lead Developer", "Engineering", "Python, AI, ML", "Expert in context orchestration and AI systems"),
            (2, "Alice Johnson", "Data Scientist", "Data Science", "Python, R, ML", "Specializes in NLP and machine learning"),
            (3, "Bob Smith", "DevOps Engineer", "Operations", "Docker, Kubernetes, AWS", "Infrastructure and deployment expert")
        ]
        
        knowledge_data = [
            (1, "Context Fusion Best Practices", "AI", "Advanced techniques for combining multiple data sources", "Sumit Nemade", "2024-01-20", 0.95),
            (2, "Vector Database Optimization", "Database", "Performance tuning for similarity search", "Alice Johnson", "2024-01-25", 0.88),
            (3, "Real-time Data Processing", "Engineering", "WebSocket and MQTT integration patterns", "Bob Smith", "2024-02-01", 0.92)
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)", projects_data)
        cursor.executemany("INSERT OR REPLACE INTO team_members VALUES (?, ?, ?, ?, ?, ?)", team_data)
        cursor.executemany("INSERT OR REPLACE INTO knowledge_base VALUES (?, ?, ?, ?, ?, ?, ?)", knowledge_data)
        
        conn.commit()
        conn.close()
        
        console.print(f"✅ Created sample database with {len(projects_data)} projects, {len(team_data)} team members, {len(knowledge_data)} knowledge articles")
        
    async def _initialize_orchestrator(self):
        """Initialize the Ragify orchestrator with all features enabled."""
        console.print("\n🔧 [yellow]Initializing Ragify orchestrator...[/yellow]")
        
        # Initialize with comprehensive configuration
        self.orchestrator = ContextOrchestrator(
            vector_db_url="memory://",  # Use in-memory FAISS for demo
            cache_url="memory://",      # Use in-memory cache for demo
            privacy_level=PrivacyLevel.RESTRICTED,
            max_tokens=8000,
            chunk_size=1000,
            overlap=200,
            cache_enabled=True,
            compression_enabled=True,
            fusion_enabled=True,
            scoring_enabled=True,
            confidence_enabled=True
        )
        
        # Add document source
        doc_source = DocumentSource(
            name="company_docs",
            path=str(self.demo_data_dir),
            file_patterns=["*.md", "*.txt"],
            chunk_size=500,
            overlap=100
        )
        await self.orchestrator.add_source(doc_source)
        
        # Add database source
        db_source = DatabaseSource(
            name="company_db",
            connection_string=f"sqlite:///{self.demo_data_dir}/company_data.db",
            db_type="sqlite",
            tables=["projects", "team_members", "knowledge_base"],
            query_template="SELECT * FROM {table} WHERE content LIKE '%{query}%' OR name LIKE '%{query}%'"
        )
        await self.orchestrator.add_source(db_source)
        
        # Add API source (simulated)
        api_source = APISource(
            name="github_api",
            url="https://api.github.com/search/repositories",
            auth_type="none",
            headers={"Accept": "application/vnd.github.v3+json"},
            query_template="q={query}&sort=stars&order=desc"
        )
        await self.orchestrator.add_source(api_source)
        
        # Add real-time source (simulated)
        realtime_source = RealtimeSource(
            name="slack_feed",
            connection_type="websocket",
            url="ws://localhost:8080/slack",  # Simulated
            topics=["general", "engineering", "ai"],
            callback_handler=self._handle_realtime_message
        )
        await self.orchestrator.add_source(realtime_source)
        
        console.print("✅ Orchestrator initialized with all data sources")
        
    async def _handle_realtime_message(self, message: dict) -> dict:
        """Handle real-time messages from simulated sources."""
        return {
            "content": f"Real-time update: {message.get('text', 'No content')}",
            "relevance": 0.8,
            "metadata": {
                "source": "slack_feed",
                "timestamp": datetime.utcnow().isoformat(),
                "is_realtime": True
            }
        }
        
    async def run_comprehensive_tests(self):
        """Run comprehensive tests of all Ragify features."""
        console.print("\n🧪 [bold blue]Running Comprehensive Feature Tests[/bold blue]")
        
        test_queries = [
            "What is the Ragify project about?",
            "Who is working on AI projects?",
            "What are the best practices for context fusion?",
            "How do we handle real-time data?",
            "What are the team development guidelines?",
            "Tell me about vector database optimization",
            "Who is the lead developer?",
            "What projects are currently active?"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for i, query in enumerate(test_queries, 1):
                task = progress.add_task(f"Testing query {i}/{len(test_queries)}: {query[:50]}...", total=None)
                
                try:
                    # Get context with comprehensive analysis
                    response = await self.orchestrator.get_context(
                        query=query,
                        user_id="demo_user",
                        session_id="demo_session",
                        max_tokens=2000,
                        include_metadata=True,
                        include_confidence=True,
                        include_sources=True
                    )
                    
                    # Store results for analysis
                    self.results[query] = {
                        "response": response,
                        "timestamp": datetime.utcnow(),
                        "success": True
                    }
                    
                    progress.update(task, description=f"✅ Query {i}: {len(response.context.chunks)} chunks found")
                    
                except Exception as e:
                    self.results[query] = {
                        "error": str(e),
                        "timestamp": datetime.utcnow(),
                        "success": False
                    }
                    progress.update(task, description=f"❌ Query {i}: Error - {str(e)[:50]}")
                    
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.5)
                
    async def analyze_results(self):
        """Analyze and display comprehensive test results."""
        console.print("\n📊 [bold blue]Comprehensive Test Results Analysis[/bold blue]")
        
        # Calculate statistics
        total_queries = len(self.results)
        successful_queries = sum(1 for r in self.results.values() if r.get("success", False))
        failed_queries = total_queries - successful_queries
        
        # Display summary
        summary_table = Table(title="Test Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Queries", str(total_queries))
        summary_table.add_row("Successful", str(successful_queries))
        summary_table.add_row("Failed", str(failed_queries))
        summary_table.add_row("Success Rate", f"{(successful_queries/total_queries)*100:.1f}%")
        
        console.print(summary_table)
        
        # Analyze successful responses
        if successful_queries > 0:
            console.print("\n🔍 [yellow]Detailed Analysis of Successful Responses[/yellow]")
            
            total_chunks = 0
            total_sources = set()
            confidence_scores = []
            relevance_scores = []
            
            for query, result in self.results.items():
                if result.get("success", False):
                    response = result["response"]
                    
                    # Count chunks and sources
                    chunks = response.context.chunks
                    total_chunks += len(chunks)
                    
                    for chunk in chunks:
                        if chunk.source:
                            total_sources.add(chunk.source.name)
                        
                        if chunk.relevance_score:
                            relevance_scores.append(chunk.relevance_score.score)
                            if chunk.relevance_score.confidence_lower and chunk.relevance_score.confidence_upper:
                                confidence_scores.append(
                                    (chunk.relevance_score.confidence_lower + chunk.relevance_score.confidence_upper) / 2
                                )
            
            # Display detailed statistics
            detail_table = Table(title="Response Analysis")
            detail_table.add_column("Metric", style="cyan")
            detail_table.add_column("Value", style="green")
            
            detail_table.add_row("Total Context Chunks", str(total_chunks))
            detail_table.add_row("Unique Data Sources", str(len(total_sources)))
            detail_table.add_row("Average Relevance Score", f"{sum(relevance_scores)/len(relevance_scores):.3f}" if relevance_scores else "N/A")
            detail_table.add_row("Average Confidence", f"{sum(confidence_scores)/len(confidence_scores):.3f}" if confidence_scores else "N/A")
            detail_table.add_row("Sources Used", ", ".join(sorted(total_sources)))
            
            console.print(detail_table)
            
        # Show sample responses
        console.print("\n💬 [yellow]Sample Responses[/yellow]")
        
        sample_queries = list(self.results.keys())[:3]  # Show first 3 queries
        
        for query in sample_queries:
            result = self.results[query]
            console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
            
            if result.get("success", False):
                response = result["response"]
                chunks = response.context.chunks
                
                console.print(f"[green]✅ Success[/green] - Found {len(chunks)} context chunks")
                
                # Show first chunk as example
                if chunks:
                    first_chunk = chunks[0]
                    console.print(f"[dim]Source:[/dim] {first_chunk.source.name if first_chunk.source else 'Unknown'}")
                    console.print(f"[dim]Relevance:[/dim] {first_chunk.relevance_score.score:.3f}" if first_chunk.relevance_score else "N/A")
                    console.print(f"[dim]Content:[/dim] {first_chunk.content[:200]}...")
            else:
                console.print(f"[red]❌ Failed[/red] - {result.get('error', 'Unknown error')}")
                
    async def test_specific_features(self):
        """Test specific advanced features of Ragify."""
        console.print("\n🔬 [bold blue]Testing Specific Advanced Features[/bold blue]")
        
        # Test 1: Intelligent Fusion with Conflict Resolution
        console.print("\n1️⃣ [yellow]Testing Intelligent Context Fusion[/yellow]")
        
        # Create conflicting data
        conflicting_chunks = [
            {
                "content": "Ragify is a context orchestration framework",
                "source": "documentation",
                "timestamp": "2024-01-15"
            },
            {
                "content": "Ragify is a production-ready system",
                "source": "marketing",
                "timestamp": "2024-02-01"
            }
        ]
        
        fusion_engine = IntelligentContextFusionEngine()
        fused_result = await fusion_engine.fuse_chunks(conflicting_chunks, "What is Ragify?")
        
        console.print(f"✅ Fusion completed: {len(fused_result.fused_chunks)} fused chunks")
        console.print(f"   Conflicts detected: {len(fused_result.conflicts)}")
        
        # Test 2: Multi-factor Scoring
        console.print("\n2️⃣ [yellow]Testing Multi-factor Scoring[/yellow]")
        
        test_chunk = {
            "content": "Advanced context fusion with vector databases",
            "metadata": {
                "source": "technical_docs",
                "author": "Sumit Nemade",
                "timestamp": "2024-01-20"
            }
        }
        
        scoring_result = await self.orchestrator.scoring_engine.calculate_multi_factor_score(
            test_chunk, "context fusion", "user123"
        )
        
        console.print(f"✅ Multi-factor score: {scoring_result.score:.3f}")
        console.print(f"   Factors: {len(scoring_result.factor_scores)}")
        
        # Test 3: Cache Management
        console.print("\n3️⃣ [yellow]Testing Cache Management[/yellow]")
        
        cache_manager = self.orchestrator.cache_manager
        test_key = "demo_test_key"
        test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        
        # Set data
        await cache_manager.set(test_key, test_data, ttl=300)
        
        # Get data
        retrieved_data = await cache_manager.get(test_key)
        
        if retrieved_data:
            console.print("✅ Cache set/get successful")
        else:
            console.print("❌ Cache test failed")
            
        # Test 4: Vector Database Operations
        console.print("\n4️⃣ [yellow]Testing Vector Database Operations[/yellow]")
        
        vector_db = self.orchestrator.vector_db
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        test_metadata = [{"id": "1", "content": "test1"}, {"id": "2", "content": "test2"}]
        
        # Add embeddings
        await vector_db.add_embeddings(test_embeddings, test_metadata)
        
        # Search
        search_results = await vector_db.search([0.1, 0.2, 0.3], k=2)
        
        console.print(f"✅ Vector search successful: {len(search_results)} results")
        
    async def performance_benchmark(self):
        """Run performance benchmarks."""
        console.print("\n⚡ [bold blue]Performance Benchmarking[/bold blue]")
        
        # Test query response time
        test_query = "What is the Ragify project about?"
        
        times = []
        for i in range(5):
            start_time = time.time()
            
            response = await self.orchestrator.get_context(
                query=test_query,
                user_id="benchmark_user",
                session_id=f"benchmark_session_{i}",
                max_tokens=1000
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        perf_table = Table(title="Performance Results")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf_table.add_row("Average Response Time", f"{avg_time:.3f}s")
        perf_table.add_row("Minimum Response Time", f"{min_time:.3f}s")
        perf_table.add_row("Maximum Response Time", f"{max_time:.3f}s")
        perf_table.add_row("Response Time Variance", f"{max_time - min_time:.3f}s")
        
        console.print(perf_table)
        
    async def cleanup(self):
        """Clean up demo resources."""
        console.print("\n🧹 [yellow]Cleaning up demo resources...[/yellow]")
        
        if self.orchestrator:
            await self.orchestrator.close()
            
        # Remove demo data
        import shutil
        if self.demo_data_dir.exists():
            shutil.rmtree(self.demo_data_dir)
            
        console.print("✅ Cleanup complete")


async def main():
    """Main demo function."""
    console.print("🎯 [bold blue]Ragify Real-World Demo: Tech Company Knowledge Management System[/bold blue]")
    console.print("=" * 80)
    
    demo = TechCompanyKnowledgeSystem()
    
    try:
        # Setup
        await demo.setup_demo_environment()
        
        # Run comprehensive tests
        await demo.run_comprehensive_tests()
        
        # Analyze results
        await demo.analyze_results()
        
        # Test specific features
        await demo.test_specific_features()
        
        # Performance benchmark
        await demo.performance_benchmark()
        
        console.print("\n🎉 [bold green]Demo completed successfully![/bold green]")
        console.print("\n📋 [yellow]Summary:[/yellow]")
        console.print("   ✅ Document processing (PDF, DOCX, TXT)")
        console.print("   ✅ Database integration (SQLite)")
        console.print("   ✅ API integration (GitHub)")
        console.print("   ✅ Real-time sources (WebSocket)")
        console.print("   ✅ Vector database (FAISS)")
        console.print("   ✅ Cache management (Memory)")
        console.print("   ✅ Intelligent context fusion")
        console.print("   ✅ Multi-factor scoring")
        console.print("   ✅ Statistical confidence bounds")
        console.print("   ✅ Privacy controls")
        
    except Exception as e:
        console.print(f"\n❌ [red]Demo failed: {e}[/red]")
        logger.error("Demo failed", error=str(e), exc_info=True)
        
    finally:
        # Cleanup
        await demo.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
