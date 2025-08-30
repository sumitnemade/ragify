"""
Test script to verify RAGify setup and dependencies.

Run this script to check if all components are working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import streamlit
        print("✅ Streamlit imported successfully")
        
        import openai
        print("✅ OpenAI imported successfully")
        
        import numpy
        print("✅ NumPy imported successfully")
        
        import pandas
        print("✅ Pandas imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ragify_imports():
    """Test if RAGify components can be imported."""
    print("\n🔍 Testing RAGify imports...")
    
    try:
        # Add src to path
        src_path = str(Path(__file__).parent.parent.parent / 'src')
        sys.path.insert(0, src_path)
        
        # Test RAGify imports
        from ragify import ContextOrchestrator
        print("✅ ContextOrchestrator imported successfully")
        
        from src.ragify.models import PrivacyLevel, SourceType
        print("✅ RAGify models imported successfully")
        
        from src.ragify.sources.document import DocumentSource
        print("✅ DocumentSource imported successfully")
        
        from src.ragify.storage.vector_db import VectorDatabase
        print("✅ VectorDatabase imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ RAGify import failed: {e}")
        return False

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment variables...")
    
    required_vars = [
        'OPENAI_API_KEY'
    ]
    
    optional_vars = [
        'VECTOR_DB_URL',
        'CACHE_URL',
        'PRIVACY_LEVEL'
    ]
    
    all_good = True
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (required)")
            all_good = False
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set: {value}")
        else:
            print(f"ℹ️  {var} is not set (will use defaults)")
    
    return all_good

def test_ragify_components():
    """Test if RAGify components can be initialized."""
    print("\n🔍 Testing RAGify component initialization...")
    
    try:
        # Add src to path
        src_path = str(Path(__file__).parent.parent.parent / 'src')
        sys.path.insert(0, src_path)
        
        from ragify import ContextOrchestrator
        from src.ragify.models import PrivacyLevel
        
        # Test orchestrator initialization
        orchestrator = ContextOrchestrator(
            vector_db_url="memory://",
            cache_url="memory://",
            privacy_level=PrivacyLevel.PRIVATE
        )
        print("✅ ContextOrchestrator initialized successfully")
        
        # Test vector database initialization
        from src.ragify.storage.vector_db import VectorDatabase
        vector_db = VectorDatabase("memory://")
        print("✅ VectorDatabase initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🔍 Testing OpenAI connection...")
    
    try:
        import openai
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OpenAI API key not set, skipping connection test")
            return True
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAGify Chat Assistant - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("RAGify Imports", test_ragify_imports),
        ("Environment Variables", test_environment),
        ("RAGify Components", test_ragify_components),
        ("OpenAI Connection", test_openai_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Copy env_example.txt to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run: streamlit run rag_chat_assistant.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Set OPENAI_API_KEY in .env file")
        print("3. Check Python path and RAGify installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
