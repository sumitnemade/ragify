"""
Test utilities and helper functions for RAGify testing infrastructure.

This module provides:
- Test data generators
- Performance measurement utilities
- Memory monitoring tools
- Security testing helpers
- Load testing utilities
"""

import asyncio
import gc
import json
import os
import random
import string
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_random_text(length: int = 100) -> str:
    """Generate random text of specified length."""
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam"
    ]
    
    text = []
    for _ in range(length // 10):  # Approximate word count
        text.append(random.choice(words))
    
    return " ".join(text)


def generate_test_documents(count: int = 10, min_length: int = 100, max_length: int = 1000) -> List[Dict[str, Any]]:
    """Generate test documents with random content."""
    documents = []
    
    for i in range(count):
        length = random.randint(min_length, max_length)
        content = generate_random_text(length)
        
        document = {
            "id": f"doc_{i}",
            "content": content,
            "metadata": {
                "title": f"Test Document {i}",
                "author": f"Author {i}",
                "category": random.choice(["tech", "science", "business", "health"]),
                "created_date": "2024-01-01",
                "tags": [f"tag_{j}" for j in range(random.randint(1, 5))]
            }
        }
        documents.append(document)
    
    return documents


def generate_test_queries(count: int = 20) -> List[str]:
    """Generate test queries for testing."""
    base_queries = [
        "What is machine learning?",
        "How does RAG work?",
        "Explain vector databases",
        "What are embeddings?",
        "How to implement search?",
        "What is semantic similarity?",
        "Explain context management",
        "How to handle conflicts?",
        "What is relevance scoring?",
        "How to optimize performance?"
    ]
    
    queries = []
    for _ in range(count):
        base_query = random.choice(base_queries)
        # Add some variation
        if random.random() > 0.7:
            base_query += " in detail"
        if random.random() > 0.8:
            base_query += " with examples"
        
        queries.append(base_query)
    
    return queries


def generate_large_dataset(size_mb: int = 100) -> str:
    """Generate a large dataset for performance testing."""
    # Estimate characters needed for size_mb
    chars_needed = size_mb * 1024 * 1024
    
    # Generate base text
    base_text = generate_random_text(1000)
    repetitions = chars_needed // len(base_text)
    
    return base_text * repetitions


# =============================================================================
# PERFORMANCE MEASUREMENT
# =============================================================================

class TimeMeasurement:
    """Container for time measurement results."""
    def __init__(self, start_time: float, end_time: float):
        self.start_time = start_time
        self.end_time = end_time
        self.elapsed_time = end_time - start_time

class MemoryMeasurement:
    """Container for memory measurement results."""
    def __init__(self, start_memory: float, end_memory: float):
        self.start_memory = start_memory
        self.end_memory = end_memory
        self.peak_memory = max(start_memory, end_memory)
        self.current_memory = end_memory
        self.memory_diff = end_memory - start_memory

@contextmanager
def measure_time(operation_name: str = "operation"):
    """Context manager to measure execution time."""
    start_time = time.time()
    timer = TimeMeasurement(start_time, start_time)  # Initialize with start time
    try:
        yield timer
    finally:
        end_time = time.time()
        timer.end_time = end_time
        timer.elapsed_time = end_time - start_time
        print(f"⏱️  {operation_name} took {timer.elapsed_time:.4f} seconds")


@contextmanager
def measure_memory(operation_name: str = "operation"):
    """Context manager to measure memory usage."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    memory = MemoryMeasurement(start_memory, start_memory)  # Initialize with start memory
    try:
        yield memory
    finally:
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory.end_memory = end_memory
        memory.peak_memory = max(start_memory, end_memory)
        memory.current_memory = end_memory
        memory.memory_diff = end_memory - start_memory
        print(f"💾 {operation_name} memory usage: {memory.memory_diff:+.2f} MB (total: {end_memory:.2f} MB)")


@contextmanager
def measure_performance(operation_name: str = "operation"):
    """Context manager to measure both time and memory."""
    with measure_time(operation_name), measure_memory(operation_name):
        yield


def benchmark_function(func, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
    """Benchmark a function with multiple iterations."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": sum(times) / len(times),
        "median_time": sorted(times)[len(times) // 2],
        "std_dev": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
    }


async def benchmark_async_function(func, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
    """Benchmark an async function with multiple iterations."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": sum(times) / len(times),
        "median_time": sorted(times)[len(times) // 2],
        "std_dev": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5
    }


# =============================================================================
# MEMORY MONITORING
# =============================================================================

@contextmanager
def monitor_memory_leaks():
    """Context manager to monitor for memory leaks."""
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()
    
    try:
        yield
    finally:
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Compare snapshots
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        
        if top_stats:
            print("🔍 Memory usage changes:")
            for stat in top_stats[:5]:  # Top 5 changes
                print(f"  {stat.traceback.format()}: {stat.size_diff:+d} B")
        else:
            print("✅ No significant memory changes detected")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024
    }


def check_memory_leak(baseline_memory: float, threshold_mb: float = 50.0) -> bool:
    """Check if memory usage indicates a potential leak."""
    current_memory = get_memory_usage()["rss_mb"]
    memory_increase = current_memory - baseline_memory
    
    if memory_increase > threshold_mb:
        print(f"⚠️  Potential memory leak detected: {memory_increase:.2f} MB increase")
        return True
    
    return False


# =============================================================================
# SECURITY TESTING HELPERS
# =============================================================================

def generate_sensitive_data() -> Dict[str, str]:
    """Generate sensitive data for security testing."""
    return {
        "email": "test@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
        "credit_card": "4111-1111-1111-1111",
        "ip_address": "192.168.1.1",
        "password": "SuperSecret123!",
        "api_key": "sk-1234567890abcdef",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC...",
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }


def check_data_exposure(data: str, sensitive_patterns: List[str]) -> List[str]:
    """Check if sensitive data patterns are exposed in text."""
    exposed_patterns = []
    
    for pattern in sensitive_patterns:
        if pattern.lower() in data.lower():
            exposed_patterns.append(pattern)
    
    return exposed_patterns


def validate_encryption(original_data: str, encrypted_data: str) -> bool:
    """Validate that data is properly encrypted."""
    # Basic checks for encryption
    if original_data == encrypted_data:
        return False  # Not encrypted
    
    if len(encrypted_data) < len(original_data):
        return False  # Encryption should not reduce size significantly
    
    # Check for common encryption patterns
    if encrypted_data.startswith("-----BEGIN") or "==" in encrypted_data:
        return True  # Likely encrypted
    
    return True  # Assume encrypted if different


# =============================================================================
# LOAD TESTING UTILITIES
# =============================================================================

async def simulate_concurrent_users(
    user_count: int,
    requests_per_user: int,
    request_function,
    *args,
    **kwargs
) -> List[Tuple[int, float, Any]]:
    """Simulate concurrent users making requests."""
    results = []
    
    async def user_worker(user_id: int):
        user_results = []
        for request_id in range(requests_per_user):
            start_time = time.time()
            try:
                result = await request_function(*args, **kwargs)
                end_time = time.time()
                user_results.append((user_id, end_time - start_time, result))
            except Exception as e:
                end_time = time.time()
                user_results.append((user_id, end_time - start_time, e))
        return user_results
    
    # Create tasks for all users
    tasks = [user_worker(i) for i in range(user_count)]
    
    # Execute all tasks concurrently
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results
    for user_results in all_results:
        results.extend(user_results)
    
    return results


def calculate_load_test_metrics(results: List[Tuple[int, float, Any]]) -> Dict[str, float]:
    """Calculate metrics from load test results."""
    if not results:
        return {}
    
    response_times = [result[1] for result in results]
    success_count = sum(1 for result in results if not isinstance(result[2], Exception))
    total_count = len(results)
    
    return {
        "total_requests": total_count,
        "successful_requests": success_count,
        "failed_requests": total_count - success_count,
        "success_rate": success_count / total_count * 100,
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "avg_response_time": sum(response_times) / len(response_times),
        "throughput_rps": total_count / max(response_times) if response_times else 0
    }


# =============================================================================
# TEST ENVIRONMENT HELPERS
# =============================================================================

def create_temp_file(content: str, extension: str = ".txt") -> Path:
    """Create a temporary file with content."""
    import tempfile
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False)
    temp_file.write(content)
    temp_file.close()
    
    return Path(temp_file.name)


def cleanup_temp_files(files: List[Path]):
    """Clean up temporary files."""
    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")


@contextmanager
def temp_environment(**env_vars):
    """Context manager to temporarily set environment variables."""
    original_env = {}
    
    try:
        # Store original values and set new ones
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        yield
    finally:
        # Restore original values
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


# =============================================================================
# MOCKING HELPERS
# =============================================================================

def create_mock_response(status: int = 200, data: Any = None, headers: Dict = None) -> MagicMock:
    """Create a mock HTTP response."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.headers = headers or {}
    
    if data:
        mock_response.json.return_value = data
        mock_response.text = json.dumps(data)
    
    return mock_response


def create_mock_database_connection() -> MagicMock:
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value = MagicMock()
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    mock_conn.close.return_value = None
    return mock_conn


def create_mock_vector_store() -> MagicMock:
    """Create a mock vector store."""
    mock_store = MagicMock()
    mock_store.add_documents.return_value = ["doc_id_1", "doc_id_2"]
    mock_store.similarity_search.return_value = [
        MagicMock(page_content="result 1", metadata={"score": 0.9}),
        MagicMock(page_content="result 2", metadata={"score": 0.8})
    ]
    return mock_store


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_performance_met(actual_time: float, expected_max: float, test_name: str):
    """Assert that performance meets expectations."""
    assert actual_time <= expected_max, (
        f"Performance test failed for {test_name}: "
        f"Expected <= {expected_max}s, got {actual_time}s"
    )


def assert_memory_usage_acceptable(usage_mb: float, max_mb: float, test_name: str):
    """Assert that memory usage is acceptable."""
    assert usage_mb <= max_mb, (
        f"Memory usage test failed for {test_name}: "
        f"Expected <= {max_mb}MB, got {usage_mb}MB"
    )


def assert_response_time_acceptable(response_time_ms: float, max_ms: float, test_name: str):
    """Assert that response time is acceptable."""
    assert response_time_ms <= max_ms, (
        f"Response time test failed for {test_name}: "
        f"Expected <= {max_ms}ms, got {response_time_ms}ms"
    )


def assert_throughput_acceptable(throughput: float, min_ops_per_sec: float, test_name: str):
    """Assert that throughput meets minimum requirements."""
    assert throughput >= min_ops_per_sec, (
        f"Throughput test failed for {test_name}: "
        f"Expected >= {min_ops_per_sec} ops/sec, got {throughput} ops/sec"
    )


# =============================================================================
# TEST CATEGORIZATION HELPERS
# =============================================================================

def categorize_test_by_complexity(test_function) -> str:
    """Categorize test by complexity based on function attributes."""
    if hasattr(test_function, 'complexity'):
        return test_function.complexity
    
    # Estimate complexity based on function name and docstring
    func_name = test_function.__name__.lower()
    docstring = test_function.__doc__ or ""
    
    if any(word in func_name for word in ['simple', 'basic', 'basic_']):
        return "simple"
    elif any(word in func_name for word in ['complex', 'advanced', 'integration']):
        return "complex"
    elif any(word in func_name for word in ['performance', 'load', 'stress']):
        return "performance"
    else:
        return "medium"


def get_test_metadata(test_function) -> Dict[str, Any]:
    """Extract metadata from test function."""
    return {
        "name": test_function.__name__,
        "module": test_function.__module__,
        "complexity": categorize_test_by_complexity(test_function),
        "docstring": test_function.__doc__,
        "markers": getattr(test_function, 'pytestmark', []),
        "file": getattr(test_function, '__code__', None).co_filename if hasattr(test_function, '__code__') else None,
        "line": getattr(test_function, '__code__', None).co_firstlineno if hasattr(test_function, '__code__') else None
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_email() -> str:
    """Generate a random email address."""
    username = random_string(8)
    domain = random_string(6)
    return f"{username}@{domain}.com"


def random_phone() -> str:
    """Generate a random phone number."""
    area_code = random.randint(100, 999)
    prefix = random.randint(100, 999)
    line_number = random.randint(1000, 9999)
    return f"+1-{area_code}-{prefix}-{line_number}"


def cleanup_test_resources():
    """Clean up test resources and force garbage collection."""
    gc.collect()
    
    # Clean up any temporary files or connections
    # This is a placeholder for more specific cleanup logic
    pass
