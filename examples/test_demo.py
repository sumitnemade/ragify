#!/usr/bin/env python3
"""
Simple test script to verify demo components work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragify import ContextOrchestrator, PrivacyLevel
from ragify.sources import DocumentSource, DatabaseSource, APISource, RealtimeSource


async def test_basic_components():
    """Test basic components work."""
    print("🧪 Testing basic Ragify components...")
    
    try:
        # Test orchestrator initialization
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.RESTRICTED
        )
        print("✅ Orchestrator initialized")
        
        # Test document source
        doc_source = DocumentSource(
            name="test_docs",
            path=".",
            file_patterns=["*.md"]
        )
        print("✅ Document source created")
        
        # Test database source
        db_source = DatabaseSource(
            name="test_db",
            connection_string="sqlite:///test.db",
            db_type="sqlite"
        )
        print("✅ Database source created")
        
        # Test API source
        api_source = APISource(
            name="test_api",
            url="https://httpbin.org/get",
            auth_type="none"
        )
        print("✅ API source created")
        
        # Test real-time source
        realtime_source = RealtimeSource(
            name="test_realtime",
            connection_type="websocket",
            url="ws://localhost:8080"
        )
        print("✅ Real-time source created")
        
        print("🎉 All basic components work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False


async def test_simple_query():
    """Test a simple query works."""
    print("\n🔍 Testing simple query...")
    
    try:
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://"
        )
        print("✅ Orchestrator created")
        
        # Create a simple document source
        doc_source = DocumentSource(
            name="test",
            url="README.md"  # Use a specific file instead of directory
        )
        print("✅ DocumentSource created")
        
        # Add source (this is not async)
        orchestrator.add_source(doc_source)
        print("✅ Source added")
        
        # Try a simple query
        try:
            print("🔍 Calling get_context...")
            response = await orchestrator.get_context(
                query="test",
                user_id="test_user",
                session_id="test_session"
            )
            
            print(f"✅ Query successful: {len(response.context.chunks)} chunks found")
            return True
        except Exception as e:
            import traceback
            print(f"❌ Query failed with error: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        import traceback
        print(f"❌ Query test setup failed: {e}")
        print(f"❌ Setup traceback: {traceback.format_exc()}")
        return False


async def main():
    """Main test function."""
    print("🚀 Testing Ragify Demo Components")
    print("=" * 50)
    
    # Test basic components
    components_ok = await test_basic_components()
    
    # Test simple query
    query_ok = await test_simple_query()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"   Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"   Query: {'✅ PASS' if query_ok else '❌ FAIL'}")
    
    if components_ok and query_ok:
        print("\n🎉 All tests passed! Demo should work correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
