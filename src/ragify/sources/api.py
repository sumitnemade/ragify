"""
API Source for handling external API data sources.
"""

import asyncio
import json
import time
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import structlog

# HTTP clients
import aiohttp
import httpx

# Authentication and security
import base64
import hashlib
import hmac

from .base import BaseDataSource
from ..models import ContextChunk, SourceType
from ..exceptions import ICOException


class APISource(BaseDataSource):
    """
    API source for handling external API data sources.
    
    Supports REST APIs, GraphQL, and other HTTP-based data sources.
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType = SourceType.API,
        url: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        auth_type: str = "none",  # none, basic, bearer, api_key, oauth2
        auth_config: Optional[Dict[str, Any]] = None,
        rate_limit: Optional[Dict[str, int]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the API source.
        
        Args:
            name: Name of the API source
            source_type: Type of data source
            url: API endpoint URL
            headers: HTTP headers for requests
            timeout: Request timeout in seconds
            auth_type: Authentication type (none, basic, bearer, api_key, oauth2)
            auth_config: Authentication configuration
            rate_limit: Rate limiting configuration
            retry_config: Retry configuration
            **kwargs: Additional configuration
        """
        super().__init__(name, source_type, **kwargs)
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.auth_type = auth_type
        self.auth_config = auth_config or {}
        self.rate_limit = rate_limit or {}
        self.retry_config = retry_config or {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0
        }
        
        # Ensure all required retry config keys are present
        if 'retry_delay' not in self.retry_config:
            self.retry_config['retry_delay'] = 1.0
        if 'backoff_factor' not in self.retry_config:
            self.retry_config['backoff_factor'] = 2.0
        
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
        # HTTP sessions
        self.session: Optional[aiohttp.ClientSession] = None
        self.httpx_client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting
        self.last_request_time = 0.0
        self.request_count = 0
        
        # Authentication
        self.access_token = None
        self.token_expiry = None
    
    async def get_chunks(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Get context chunks from API.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            min_relevance: Minimum relevance threshold
            user_id: Optional user ID for personalization
            session_id: Optional session ID for continuity
            
        Returns:
            List of context chunks
        """
        try:
            # Validate query
            query = await self._validate_query(query)
            
            self.logger.info(f"Getting chunks from API for query: {query}")
            
            # Make API request
            response_data = await self._make_api_request(query, user_id, session_id)
            
            # Process response into chunks
            chunks = await self._process_api_response(response_data, query)
            
            # Filter by relevance
            relevant_chunks = await self._filter_chunks_by_relevance(
                chunks, min_relevance
            )
            
            # Apply max_chunks limit
            if max_chunks:
                relevant_chunks = relevant_chunks[:max_chunks]
            
            # Update statistics
            await self._update_stats(len(relevant_chunks))
            
            self.logger.info(f"Retrieved {len(relevant_chunks)} chunks from API")
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks from API: {e}")
            return []
    
    async def refresh(self) -> None:
        """Refresh the API source."""
        try:
            self.logger.info("Refreshing API source")
            
            # Clear any cached data
            # Refresh authentication tokens if needed
            if self.auth_type == 'oauth2' and self._is_token_expired():
                try:
                    await self._refresh_oauth2_token()
                except Exception as e:
                    self.logger.warning(f"Failed to refresh OAuth2 token: {e}")
            
            self.logger.info("API source refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh API source: {e}")
    
    async def close(self) -> None:
        """Close the API source."""
        try:
            self.logger.info("Closing API source")
            
            if self.session:
                await self.session.close()
                self.session = None
            
            if self.httpx_client:
                await self.httpx_client.aclose()
                self.httpx_client = None
                
        except Exception as e:
            self.logger.error(f"Error closing API source: {e}")
    
    async def _make_api_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Make API request to get data.
        
        Args:
            query: Search query
            user_id: Optional user ID
            session_id: Optional session ID
            
        Returns:
            API response data
        """
        # Initialize HTTP client if needed
        await self._ensure_http_client()
        
        # Apply rate limiting
        await self._apply_rate_limit()
        
        # Prepare request parameters
        params = {
            'query': query,
        }
        
        if user_id:
            params['user_id'] = user_id
        
        if session_id:
            params['session_id'] = session_id
        
        # Add authentication headers
        headers = await self._get_auth_headers()
        
        # Add custom headers
        headers.update(self.headers)
        
        # Make request with retry logic
        for attempt in range(self.retry_config['max_retries'] + 1):
            try:
                if self.httpx_client:
                    # Use httpx for better async support
                    response = await self.httpx_client.get(
                        self.url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
                else:
                    # Fallback to aiohttp
                    async with self.session.get(
                        self.url,
                        params=params,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
                        
            except Exception as e:
                self.logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_config['max_retries']:
                    # Wait before retry
                    delay = self.retry_config['retry_delay'] * (self.retry_config['backoff_factor'] ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # All retries failed, handle failure properly
                    await self._handle_api_failure(query, Exception(f"Max retries ({self.retry_config['max_retries']}) exceeded"))
                    
    
    async def _ensure_http_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if not self.httpx_client:
            self.httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if not self.rate_limit:
            return
        
        current_time = time.time()
        
        # Handle rate limit as integer (requests per minute) or dict
        if isinstance(self.rate_limit, int):
            # Simple rate limiting - requests per minute
            if self.request_count >= self.rate_limit:
                # Reset counter after a minute
                if current_time - self.last_request_time >= 60:
                    self.request_count = 0
                else:
                    await asyncio.sleep(60 - (current_time - self.last_request_time))
        elif isinstance(self.rate_limit, dict):
            # Advanced rate limiting configuration
            if 'requests_per_second' in self.rate_limit:
                time_since_last = current_time - self.last_request_time
                min_interval = 1.0 / self.rate_limit['requests_per_second']
                
                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    await asyncio.sleep(sleep_time)
            
            if 'requests_per_minute' in self.rate_limit:
                if self.request_count >= self.rate_limit['requests_per_minute']:
                    if current_time - self.last_request_time >= 60:
                        self.request_count = 0
                    else:
                        await asyncio.sleep(60 - (current_time - self.last_request_time))
        
        self.last_request_time = current_time
        self.request_count += 1
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type."""
        headers = {}
        
        if self.auth_type == "none":
            return headers
        
        elif self.auth_type == "basic":
            username = self.auth_config.get('username')
            password = self.auth_config.get('password')
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers['Authorization'] = f"Basic {credentials}"
        
        elif self.auth_type == "bearer":
            token = self.auth_config.get('token')
            if token:
                headers['Authorization'] = f"Bearer {token}"
        
        elif self.auth_type == "api_key":
            api_key = self.auth_config.get('api_key')
            header_name = self.auth_config.get('header_name', 'X-API-Key')
            if api_key:
                headers[header_name] = api_key
        
        elif self.auth_type == "oauth2":
            # Handle OAuth2 token refresh
            if await self._is_token_expired():
                await self._refresh_oauth2_token()
            
            if self.access_token:
                headers['Authorization'] = f"Bearer {self.access_token}"
        
        return headers
    
    async def _is_token_expired(self) -> bool:
        """Check if OAuth2 token is expired."""
        if not self.access_token or not self.token_expiry:
            return True
        
        # Add 5-minute buffer
        return datetime.now() >= self.token_expiry - timedelta(minutes=5)
    
    async def _refresh_oauth2_token(self) -> None:
        """Refresh OAuth2 access token."""
        try:
            token_url = self.auth_config.get('token_url')
            client_id = self.auth_config.get('client_id')
            client_secret = self.auth_config.get('client_secret')
            refresh_token = self.auth_config.get('refresh_token')
            
            if not all([token_url, client_id, client_secret, refresh_token]):
                self.logger.error("Missing OAuth2 configuration")
                return
            
            # Make token refresh request
            data = {
                'grant_type': 'refresh_token',
                'client_id': client_id,
                'client_secret': client_secret,
                'refresh_token': refresh_token
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(token_url, data=data)
                response.raise_for_status()
                
                token_data = response.json()
                self.access_token = token_data['access_token']
                
                # Calculate expiry time
                expires_in = token_data.get('expires_in', 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                
                self.logger.info("OAuth2 token refreshed successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to refresh OAuth2 token: {e}")
    
    async def _handle_api_failure(self, query: str, error: Exception) -> None:
        """Handle API failure with proper error logging and cleanup."""
        self.logger.error(f"API source {self.name} failed for query '{query}': {error}")
        
        # Mark source as temporarily unavailable
        self.last_error = error
        self.last_error_time = time.time()
        
        # Clean up resources
        await self._cleanup_http_clients()
        
        # Raise exception for proper error handling upstream
        raise ICOException(f"API source {self.name} failed: {error}")
    
    async def _cleanup_http_clients(self) -> None:
        """Clean up HTTP clients and sessions."""
        try:
            if self.httpx_client:
                await self.httpx_client.aclose()
                self.httpx_client = None
            
            if self.session:
                await self.session.close()
                self.session = None
        except Exception as e:
            self.logger.warning(f"Error during HTTP client cleanup: {e}")
    
    async def _process_api_response(
        self,
        response_data: Dict[str, Any],
        query: str
    ) -> List[ContextChunk]:
        """
        Process API response into context chunks.
        
        Args:
            response_data: API response data
            query: Original query
            
        Returns:
            List of context chunks
        """
        chunks = []
        
        # Extract data from response
        data_items = response_data.get('data', [])
        
        for item in data_items:
            content = item.get('content', '')
            if not content:
                continue
            
            # Create chunk
            chunk = await self._create_chunk(
                content=content,
                metadata={
                    'api_source': self.name,
                    'api_url': self.url,
                    'query': query,
                    'response_metadata': item.get('metadata', {}),
                },
                token_count=len(content.split())
            )
            
            # Add relevance score if provided
            relevance = item.get('relevance', 0.5)
            chunk.relevance_score = type('obj', (object,), {
                'score': relevance,
                'confidence_lower': max(0, relevance - 0.1),
                'confidence_upper': min(1, relevance + 0.1),
                'confidence_level': 0.95,
                'factors': {'api_relevance': relevance}
            })()
            
            chunks.append(chunk)
        
        return chunks
    
    async def _filter_chunks_by_relevance(
        self,
        chunks: List[ContextChunk],
        min_relevance: float
    ) -> List[ContextChunk]:
        """
        Filter chunks by relevance score.
        
        Args:
            chunks: List of chunks to filter
            min_relevance: Minimum relevance threshold
            
        Returns:
            Filtered list of chunks
        """
        if min_relevance <= 0.0:
            return chunks
        
        relevant_chunks = [
            chunk for chunk in chunks
            if chunk.relevance_score and chunk.relevance_score.score >= min_relevance
        ]
        
        # Sort by relevance
        relevant_chunks.sort(
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
            reverse=True
        )
        
        return relevant_chunks
    
    async def _create_chunk(self, content: str, metadata: Dict[str, Any], token_count: int) -> ContextChunk:
        """Create a ContextChunk object."""
        from ..models import ContextChunk, ContextSource, SourceType
        from uuid import uuid4
        
        return ContextChunk(
            id=uuid4(),  # Generate unique ID for each chunk
            content=content,
            source=ContextSource(
                name=self.name,
                source_type=SourceType.API,
                url=self.url
            ),
            metadata=metadata,
            token_count=token_count
        )
