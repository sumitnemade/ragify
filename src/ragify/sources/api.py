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
import secrets
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

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
        auth_type: str = "none",  # none, basic, bearer, api_key, oauth2, jwt, hmac
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
            auth_type: Authentication type (none, basic, bearer, api_key, oauth2, jwt, hmac)
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
        
        # Enhanced rate limiting with sliding window
        self.last_request_time = 0.0
        self.request_count = 0
        self.request_timestamps = []
        self.rate_limit_window = 60  # seconds
        
        # Enhanced authentication
        self.access_token = None
        self.token_expiry = None
        self.refresh_token = None
        self.id_token = None
        
        # API key rotation
        self.api_keys = []
        self.current_key_index = 0
        self.key_rotation_interval = 3600  # 1 hour
        self.last_key_rotation = time.time()
        
        # JWT handling
        self.jwt_secret = None
        self.jwt_algorithm = 'HS256'
        self.jwt_issuer = None
        self.jwt_audience = None
        
        # HMAC signing
        self.hmac_secret = None
        self.hmac_algorithm = 'sha256'
        
        # OAuth2 flow state
        self.oauth2_state = None
        self.oauth2_code_verifier = None
        self.oauth2_redirect_uri = None
        
        # Initialize authentication based on type
        self._initialize_auth()
    
    def _initialize_auth(self) -> None:
        """Initialize authentication based on auth type."""
        if self.auth_type == "oauth2":
            self._initialize_oauth2()
        elif self.auth_type == "jwt":
            self._initialize_jwt()
        elif self.auth_type == "hmac":
            self._initialize_hmac()
        elif self.auth_type == "api_key":
            self._initialize_api_key_rotation()
    
    def _initialize_oauth2(self) -> None:
        """Initialize OAuth2 configuration."""
        self.oauth2_redirect_uri = self.auth_config.get('redirect_uri')
        self.oauth2_state = secrets.token_urlsafe(32)
        self.oauth2_code_verifier = secrets.token_urlsafe(64)
        
        # PKCE code challenge
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self.oauth2_code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        self.auth_config['code_challenge'] = code_challenge
        self.auth_config['code_challenge_method'] = 'S256'
    
    def _initialize_jwt(self) -> None:
        """Initialize JWT configuration."""
        self.jwt_secret = self.auth_config.get('secret')
        self.jwt_algorithm = self.auth_config.get('algorithm', 'HS256')
        self.jwt_issuer = self.auth_config.get('issuer')
        self.jwt_audience = self.auth_config.get('audience')
        
        if not self.jwt_secret:
            self.logger.warning("JWT secret not provided, using default")
            self.jwt_secret = secrets.token_urlsafe(32)
    
    def _initialize_hmac(self) -> None:
        """Initialize HMAC signing configuration."""
        self.hmac_secret = self.auth_config.get('secret')
        self.hmac_algorithm = self.auth_config.get('algorithm', 'sha256')
        
        if not self.hmac_secret:
            self.logger.warning("HMAC secret not provided, using default")
            self.hmac_secret = secrets.token_urlsafe(32)
    
    def _initialize_api_key_rotation(self) -> None:
        """Initialize API key rotation."""
        api_key = self.auth_config.get('api_key')
        if api_key:
            self.api_keys = [api_key]
        else:
            # Generate multiple keys for rotation
            self.api_keys = [secrets.token_urlsafe(32) for _ in range(3)]
        
        self.current_key_index = 0
        self.last_key_rotation = time.time()
    
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
            
            # Rotate API keys if needed
            if self.auth_type == 'api_key':
                await self._rotate_api_keys()
            
            # Refresh JWT token if needed
            if self.auth_type == 'jwt' and self._is_jwt_expired():
                await self._refresh_jwt_token()
            
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
        
        # Apply enhanced rate limiting
        await self._apply_enhanced_rate_limit()
        
        # Prepare request parameters
        params = {
            'query': query,
        }
        
        if user_id:
            params['user_id'] = user_id
        
        if session_id:
            params['session_id'] = session_id
        
        # Add authentication headers
        headers = await self._get_enhanced_auth_headers()
        
        # Add custom headers
        headers.update(self.headers)
        
        # Add request signing if HMAC is enabled
        if self.auth_type == 'hmac':
            headers.update(await self._sign_request(params, headers))
        
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
    
    async def _apply_enhanced_rate_limit(self) -> None:
        """Apply enhanced rate limiting with sliding window."""
        if not self.rate_limit:
            return
        
        current_time = time.time()
        
        # Clean old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < self.rate_limit_window]
        
        # Handle different rate limit configurations
        if isinstance(self.rate_limit, int):
            # Simple rate limiting - requests per minute
            if len(self.request_timestamps) >= self.rate_limit:
                sleep_time = self.rate_limit_window - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        elif isinstance(self.rate_limit, dict):
            # Advanced rate limiting configuration
            if 'requests_per_second' in self.rate_limit:
                time_since_last = current_time - self.last_request_time
                min_interval = 1.0 / self.rate_limit['requests_per_second']
                
                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    await asyncio.sleep(sleep_time)
            
            if 'requests_per_minute' in self.rate_limit:
                if len(self.request_timestamps) >= self.rate_limit['requests_per_minute']:
                    sleep_time = self.rate_limit_window - (current_time - self.request_timestamps[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            if 'burst_limit' in self.rate_limit:
                # Burst limiting - allow short bursts but maintain average
                burst_window = self.rate_limit.get('burst_window', 10)
                recent_requests = [ts for ts in self.request_timestamps 
                                 if current_time - ts < burst_window]
                
                if len(recent_requests) > self.rate_limit['burst_limit']:
                    sleep_time = burst_window - (current_time - recent_requests[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
        
        self.last_request_time = current_time
        self.request_timestamps.append(current_time)
    
    async def _get_enhanced_auth_headers(self) -> Dict[str, str]:
        """Get enhanced authentication headers based on auth type."""
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
            # Use rotated API key
            await self._rotate_api_keys()
            api_key = self.api_keys[self.current_key_index]
            header_name = self.auth_config.get('header_name', 'X-API-Key')
            headers[header_name] = api_key
        
        elif self.auth_type == "oauth2":
            # Handle OAuth2 token refresh
            if await self._is_token_expired():
                await self._refresh_oauth2_token()
            
            if self.access_token:
                headers['Authorization'] = f"Bearer {self.access_token}"
        
        elif self.auth_type == "jwt":
            # Generate or refresh JWT token
            if await self._is_jwt_expired():
                await self._refresh_jwt_token()
            
            token = await self._generate_jwt_token()
            headers['Authorization'] = f"Bearer {token}"
        
        elif self.auth_type == "hmac":
            # HMAC signing will be added in _sign_request
            pass
        
        return headers
    
    async def _is_token_expired(self) -> bool:
        """Check if OAuth2 token is expired."""
        if not self.access_token or not self.token_expiry:
            return True
        
        # Add 5-minute buffer
        return datetime.now() >= self.token_expiry - timedelta(minutes=5)
    
    async def _is_jwt_expired(self) -> bool:
        """Check if JWT token is expired."""
        if not self.id_token:
            return True
        
        try:
            # Decode without audience validation to avoid issues
            payload = jwt.decode(
                self.id_token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm],
                options={"verify_aud": False}
            )
            exp = payload.get('exp')
            if exp:
                # Add 5-minute buffer
                current_time = datetime.now().timestamp()
                return current_time >= exp - 300
            else:
                # No expiration claim, consider expired
                return True
        except jwt.ExpiredSignatureError:
            return True
        except jwt.InvalidTokenError:
            return True
        
        # If we get here, token is valid and not expired
        return False
    
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
    
    async def _refresh_jwt_token(self) -> None:
        """Refresh JWT token."""
        try:
            # Generate new JWT token
            payload = {
                'iss': self.jwt_issuer or self.name,
                'aud': self.jwt_audience or 'ragify',
                'iat': datetime.now().timestamp(),
                'exp': (datetime.now() + timedelta(hours=1)).timestamp(),
                'sub': self.name
            }
            
            self.id_token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            self.logger.info("JWT token refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh JWT token: {e}")
    
    async def _generate_jwt_token(self) -> str:
        """Generate JWT token."""
        if not self.id_token:
            await self._refresh_jwt_token()
        
        return self.id_token
    
    async def _rotate_api_keys(self) -> None:
        """Rotate API keys based on interval."""
        current_time = time.time()
        
        if current_time - self.last_key_rotation >= self.key_rotation_interval:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.last_key_rotation = current_time
            
            # Generate new key if needed
            if len(self.api_keys) < 3:
                self.api_keys.append(secrets.token_urlsafe(32))
            
            self.logger.info(f"API key rotated to index {self.current_key_index}")
    
    async def _sign_request(self, params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, str]:
        """Sign request with HMAC for security."""
        try:
            # Create signature string
            timestamp = str(int(time.time()))
            nonce = secrets.token_urlsafe(16)
            
            # Sort parameters for consistent signing
            param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            header_str = '&'.join([f"{k}={v}" for k, v in sorted(headers.items())])
            
            signature_string = f"{timestamp}&{nonce}&{param_str}&{header_str}"
            
            # Create HMAC signature
            if self.hmac_algorithm == 'sha256':
                signature = hmac.new(
                    self.hmac_secret.encode(),
                    signature_string.encode(),
                    hashlib.sha256
                ).hexdigest()
            else:
                signature = hmac.new(
                    self.hmac_secret.encode(),
                    signature_string.encode(),
                    hashlib.sha1
                ).hexdigest()
            
            # Add signature headers
            return {
                'X-Timestamp': timestamp,
                'X-Nonce': nonce,
                'X-Signature': signature,
                'X-Signature-Algorithm': self.hmac_algorithm
            }
            
        except Exception as e:
            self.logger.error(f"Failed to sign request: {e}")
            return {}
    
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