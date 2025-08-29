"""
Custom exceptions for the Intelligent Context Orchestration plugin.
"""

from typing import Any, Dict, Optional


class ICOException(Exception):
    """Base exception for all ICO-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ContextNotFoundError(ICOException):
    """Raised when requested context cannot be found."""
    
    def __init__(self, query: str, user_id: Optional[str] = None):
        message = f"Context not found for query: {query}"
        if user_id:
            message += f" (user: {user_id})"
        super().__init__(message, error_code="CONTEXT_NOT_FOUND")


class SourceConnectionError(ICOException):
    """Raised when unable to connect to a data source."""
    
    def __init__(self, source_name: str, error: str):
        message = f"Failed to connect to source '{source_name}': {error}"
        super().__init__(message, error_code="SOURCE_CONNECTION_ERROR")


class SourceAuthenticationError(ICOException):
    """Raised when authentication fails for a data source."""
    
    def __init__(self, source_name: str, error: str):
        message = f"Authentication failed for source '{source_name}': {error}"
        super().__init__(message, error_code="SOURCE_AUTH_ERROR")


class ContextValidationError(ICOException):
    """Raised when context validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str):
        message = f"Context validation failed for field '{field}' with value '{value}': {reason}"
        super().__init__(message, error_code="CONTEXT_VALIDATION_ERROR")


class PrivacyViolationError(ICOException):
    """Raised when privacy rules are violated."""
    
    def __init__(self, operation: str, privacy_level: str, required_level: str):
        message = f"Privacy violation in {operation}: required {required_level}, got {privacy_level}"
        super().__init__(message, error_code="PRIVACY_VIOLATION")


class TokenLimitExceededError(ICOException):
    """Raised when context exceeds token limits."""
    
    def __init__(self, current_tokens: int, max_tokens: int):
        message = f"Token limit exceeded: {current_tokens} > {max_tokens}"
        super().__init__(message, error_code="TOKEN_LIMIT_EXCEEDED")


class RelevanceScoringError(ICOException):
    """Raised when relevance scoring fails."""
    
    def __init__(self, error: str):
        message = f"Relevance scoring failed: {error}"
        super().__init__(message, error_code="RELEVANCE_SCORING_ERROR")


class CacheError(ICOException):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, error: str):
        message = f"Cache {operation} failed: {error}"
        super().__init__(message, error_code="CACHE_ERROR")


class VectorDBError(ICOException):
    """Raised when vector database operations fail."""
    
    def __init__(self, operation: str, error: str):
        message = f"Vector database {operation} failed: {error}"
        super().__init__(message, error_code="VECTOR_DB_ERROR")


class ConfigurationError(ICOException):
    """Raised when configuration is invalid."""
    
    def __init__(self, field: str, value: Any, reason: str):
        message = f"Configuration error for field '{field}' with value '{value}': {reason}"
        super().__init__(message, error_code="CONFIGURATION_ERROR")


class RateLimitError(ICOException):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, source_name: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for source '{source_name}'"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, error_code="RATE_LIMIT_ERROR")


class TimeoutError(ICOException):
    """Raised when operations timeout."""
    
    def __init__(self, operation: str, timeout: int):
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        super().__init__(message, error_code="TIMEOUT_ERROR")


class DataSourceError(ICOException):
    """Raised when data source operations fail."""
    
    def __init__(self, source_name: str, operation: str, error: str):
        message = f"Data source '{source_name}' {operation} failed: {error}"
        super().__init__(message, error_code="DATA_SOURCE_ERROR")


class EmbeddingError(ICOException):
    """Raised when embedding generation fails."""
    
    def __init__(self, error: str):
        message = f"Embedding generation failed: {error}"
        super().__init__(message, error_code="EMBEDDING_ERROR")


class AnalyticsError(ICOException):
    """Raised when analytics operations fail."""
    
    def __init__(self, operation: str, error: str):
        message = f"Analytics {operation} failed: {error}"
        super().__init__(message, error_code="ANALYTICS_ERROR")
