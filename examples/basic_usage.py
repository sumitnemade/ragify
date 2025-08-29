"""
Basic usage example for the Intelligent Context Orchestration plugin.

This example demonstrates how to integrate ICO with an LLM application
for intelligent context management.
"""

import asyncio
import os
from typing import List

from ragify import ContextOrchestrator
from ragify.models import PrivacyLevel, SourceType
from ragify.sources import DocumentSource, APISource, DatabaseSource


class MockLLM:
    """Mock LLM for demonstration purposes."""
    
    async def generate(self, prompt: str) -> str:
        """Generate a response using the LLM."""
        # In a real application, this would call an actual LLM API
        return f"LLM Response to: {prompt[:100]}..."


class MockAPISource(APISource):
    """Mock API source for demonstration."""
    
    async def get_chunks(self, query: str, **kwargs):
        """Get chunks from the API source."""
        # Mock API response
        from ragify.models import ContextChunk, ContextSource
        
        # Create a proper source object
        source = ContextSource(
            name=self.name,
            source_type=self.source_type,
            url=self.url
        )
        
        # Create a chunk with proper structure
        chunk = ContextChunk(
            content=f"API data about {query}: Sales increased by 15% this quarter.",
            source=source,
            metadata={"source": "api", "confidence": 0.8}
        )
        
        return [chunk]


class MockDatabaseSource(DatabaseSource):
    """Mock database source for demonstration."""
    
    async def get_chunks(self, query: str, **kwargs):
        """Get chunks from the database source."""
        # Mock database response
        from ragify.models import ContextChunk, ContextSource
        
        # Create a proper source object
        source = ContextSource(
            name=self.name,
            source_type=self.source_type,
            url=self.url
        )
        
        # Create a chunk with proper structure
        chunk = ContextChunk(
            content=f"Database data about {query}: Customer satisfaction is 92%.",
            source=source,
            metadata={"source": "database", "confidence": 0.9}
        )
        
        return [chunk]


async def setup_orchestrator() -> ContextOrchestrator:
    """Set up the context orchestrator with data sources."""
    
    try:
        # Initialize orchestrator with memory-based storage for demo
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",  # Use in-memory vector database for demo
            cache_url="memory://",      # Use in-memory cache for demo
            privacy_level=PrivacyLevel.ENTERPRISE
        )
        print("✅ Orchestrator created successfully")
        
        # Add document source (only if docs directory exists)
        if os.path.exists("./docs"):
            try:
                doc_source = DocumentSource(
                    name="company_docs",
                    source_type=SourceType.DOCUMENT,
                    url="./docs",
                    chunk_size=1000,
                    overlap=200
                )
                orchestrator.add_source(doc_source)
                print("✅ Document source added")
            except Exception as e:
                print(f"⚠️  Could not add document source: {e}")
        else:
            print("⚠️  Docs directory not found, skipping document source")
        
        # Add API source
        try:
            api_source = MockAPISource(
                name="sales_api",
                source_type=SourceType.API,
                url="https://api.sales.com",
                headers={"Authorization": "Bearer token"},
                timeout=30
            )
            orchestrator.add_source(api_source)
            print("✅ API source added")
        except Exception as e:
            print(f"⚠️  Could not add API source: {e}")
        
        # Add database source
        try:
            db_source = MockDatabaseSource(
                name="customer_db",
                source_type=SourceType.DATABASE,
                url="postgresql://user:pass@localhost/customers",
                query_template="SELECT * FROM customer_data WHERE content ILIKE %s"
            )
            orchestrator.add_source(db_source)
            print("✅ Database source added")
        except Exception as e:
            print(f"⚠️  Could not add database source: {e}")
        
        print(f"✅ Total sources added: {len(orchestrator.list_sources())}")
        return orchestrator
        
    except Exception as e:
        print(f"❌ Error in setup_orchestrator: {e}")
        import traceback
        traceback.print_exc()
        raise


async def process_user_query(
    orchestrator: ContextOrchestrator,
    llm: MockLLM,
    query: str,
    user_id: str = "user123"
) -> str:
    """Process a user query with intelligent context orchestration."""
    
    print(f"\n🔍 Processing query: {query}")
    print(f"👤 User: {user_id}")
    
    # Get intelligent context
    context_response = await orchestrator.get_context(
        query=query,
        user_id=user_id,
        session_id="session456",
        max_tokens=4000,
        min_relevance=0.5,
        privacy_level=PrivacyLevel.ENTERPRISE
    )
    
    context = context_response.context
    
    print(f"📊 Context retrieved:")
    print(f"   - Processing time: {context_response.processing_time:.2f}s")
    print(f"   - Cache hit: {context_response.cache_hit}")
    print(f"   - Total chunks: {len(context.chunks)}")
    print(f"   - Total tokens: {context.total_tokens}")
    print(f"   - Sources: {[s.name for s in context.sources]}")
    
    if context.relevance_score:
        print(f"   - Overall relevance: {context.relevance_score.score:.2f}")
        print(f"   - Confidence: {context.relevance_score.confidence_lower:.2f}-{context.relevance_score.confidence_upper:.2f}")
    
    # Display context chunks
    print(f"\n📄 Context chunks:")
    for i, chunk in enumerate(context.chunks, 1):
        print(f"   {i}. [{chunk.source.name}] {chunk.content[:100]}...")
        if chunk.relevance_score:
            print(f"      Relevance: {chunk.relevance_score.score:.2f}")
    
    # Generate LLM response with context
    prompt = f"""
Context Information:
{context.content}

User Query: {query}

Please provide a comprehensive answer based on the context above.
"""
    
    print(f"\n🤖 Generating LLM response...")
    response = await llm.generate(prompt)
    
    return response


async def demonstrate_real_time_updates(orchestrator: ContextOrchestrator):
    """Demonstrate real-time context updates."""
    
    print(f"\n🔄 Demonstrating real-time updates...")
    
    # Subscribe to updates
    async def handle_update(update_data):
        print(f"📡 Received update: {update_data}")
    
    await orchestrator.updates_engine.subscribe_to_updates("sales_api", handle_update)
    
    # Trigger a mock update
    await orchestrator.updates_engine.trigger_update(
        "sales_api",
        {"type": "sales_update", "data": "New sales data available"}
    )
    
    # Wait for processing
    await asyncio.sleep(1)


async def demonstrate_context_history(orchestrator: ContextOrchestrator):
    """Demonstrate context history retrieval."""
    
    print(f"\n📚 Demonstrating context history...")
    
    # Get context history for a user
    history = await orchestrator.get_context_history("user123", limit=5)
    
    print(f"Found {len(history)} historical contexts:")
    for i, context in enumerate(history, 1):
        print(f"   {i}. Query: {context.query[:50]}...")
        print(f"      Created: {context.created_at}")
        print(f"      Chunks: {len(context.chunks)}")


async def demonstrate_analytics(orchestrator: ContextOrchestrator):
    """Demonstrate analytics and monitoring."""
    
    print(f"\n📈 Demonstrating analytics...")
    
    # Get orchestrator analytics
    analytics = await orchestrator.get_analytics()
    print(f"Orchestrator Analytics:")
    for key, value in analytics.items():
        print(f"   {key}: {value}")
    
    # Get update engine stats
    update_stats = await orchestrator.updates_engine.get_update_stats()
    print(f"Update Engine Stats:")
    for key, value in update_stats.items():
        print(f"   {key}: {value}")


async def demonstrate_privacy_features(orchestrator: ContextOrchestrator):
    """Demonstrate privacy features."""
    
    print(f"\n🔒 Demonstrating privacy features...")
    
    # Test with sensitive data
    sensitive_query = "What are the sales figures for customer john.doe@company.com?"
    
    context_response = await orchestrator.get_context(
        query=sensitive_query,
        user_id="user123",
        privacy_level=PrivacyLevel.RESTRICTED
    )
    
    context = context_response.context
    
    # Check if sensitive data was handled
    if context.metadata.get('anonymized'):
        print("✅ Sensitive data was anonymized")
    
    if context.metadata.get('encrypted'):
        print("✅ Context was encrypted")
    
    print(f"Privacy level applied: {context.privacy_level}")


async def main():
    """Main demonstration function."""
    
    print("🚀 Ragify - Intelligent Context Orchestration - Basic Usage Example")
    print("=" * 60)
    
    try:
        # Set up orchestrator
        print("\n🔧 Setting up context orchestrator...")
        orchestrator = await setup_orchestrator()
        
        # Initialize mock LLM
        llm = MockLLM()
        
        # Example queries
        queries = [
            "What are the latest sales figures?",
            "How is customer satisfaction performing?",
            "What are the key metrics for Q4?",
        ]
        
        # Process each query
        for query in queries:
            try:
                response = await process_user_query(orchestrator, llm, query)
                print(f"\n💬 LLM Response: {response}")
                print("-" * 60)
            except Exception as e:
                print(f"⚠️  Error processing query '{query}': {e}")
                continue
        
        # Demonstrate basic features only (skip complex ones that might hang)
        print("\n📊 Demonstrating basic features...")
        
        # Test basic methods
        try:
            sources = orchestrator.list_sources()
            print(f"✅ Available sources: {sources}")
        except Exception as e:
            print(f"⚠️  Could not list sources: {e}")
        
        try:
            config = orchestrator.config
            print(f"✅ Config privacy level: {config.privacy_level}")
        except Exception as e:
            print(f"⚠️  Could not get config: {e}")
        
        # Clean up
        try:
            await orchestrator.close()
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
        
        print(f"\n✅ Example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
