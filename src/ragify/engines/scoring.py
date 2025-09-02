"""
Context Scoring Engine for intelligent relevance assessment.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy import stats
from scipy.stats import norm, t, bootstrap
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import DescrStatsW
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import structlog
import time
import hashlib
import json
from pathlib import Path

from ..models import ContextChunk, RelevanceScore, OrchestratorConfig
from ..exceptions import RelevanceScoringError


class ContextScoringEngine:
    """
    Intelligent context scoring engine with confidence intervals.
    
    Uses multiple scoring methods and ensemble techniques to provide
    accurate relevance scores with confidence bounds.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the scoring engine.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Initialize embedding model with robust error handling
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # Initialize ML models for ensemble scoring
        self.ml_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        self.current_model_name = 'random_forest'
        self.ml_model = self.ml_models[self.current_model_name]
        self._is_trained = False
        
        # Model persistence configuration
        self.model_persistence_config = {
            'enabled': True,
            'model_dir': 'models',
            'auto_save': True,
            'save_after_training': True,
            'model_version': '1.0.0'
        }
        
        # Feature preprocessing
        self.feature_scaler = StandardScaler()
        self._is_scaler_fitted = False
        
        # Training configuration
        self.training_config = {
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            'hyperparameter_optimization': True,
            'optimization_method': 'grid_search',  # 'grid_search' or 'random_search'
            'max_iterations': 100,
            'early_stopping': True,
            'min_improvement': 0.001
        }
        
        # Scoring weights for ensemble
        self.scoring_weights = {
            'semantic_similarity': 0.25,
            'keyword_overlap': 0.15,
            'freshness': 0.10,
            'source_authority': 0.10,
            'content_quality': 0.10,
            'user_preference': 0.10,
            'contextual_relevance': 0.05,
            'sentiment_alignment': 0.05,
            'complexity_match': 0.05,
            'domain_expertise': 0.05,
        }
        
        # Ensemble method configuration
        self.ensemble_config = {
            'primary_method': 'weighted_average',
            'secondary_methods': ['geometric_mean', 'harmonic_mean', 'trimmed_mean'],
            'use_ml_ensemble': True,
            'ensemble_weights': {
                'weighted_average': 0.4,
                'geometric_mean': 0.2,
                'harmonic_mean': 0.2,
                'trimmed_mean': 0.2,
            },
            'trim_percentage': 0.1,  # Trim 10% from each end
            'min_scores_for_ensemble': 3,
        }
        
        # Statistical confidence configuration
        self.confidence_config = {
            'default_confidence_level': 0.95,
            'bootstrap_samples': 1000,
            'min_sample_size': 5,
            'use_bootstrap': True,
            'use_t_distribution': True,
            'use_weighted_stats': True,
            'use_bayesian': True,
            'use_jackknife': True,
            'use_monte_carlo': True,
            'max_bootstrap_samples': 10000,
            'bootstrap_strategies': ['percentile', 'bca', 'abc', 'studentized'],
            'robust_estimation': True,
            'outlier_detection': True,
            'normality_testing': True,
            'heteroscedasticity_testing': True,
            'autocorrelation_testing': True,
            'confidence_interval_methods': ['bootstrap', 't_distribution', 'normal', 'weighted', 'bayesian', 'jackknife', 'monte_carlo'],
            'fallback_methods': ['simple', 'robust', 'nonparametric'],
            'validation_thresholds': {
                'min_confidence_width': 0.01,
                'max_confidence_width': 0.5,
                'min_sample_quality': 0.7,
                'max_outlier_ratio': 0.2
            },
            'performance_optimization': {
                'parallel_bootstrap': True,
                'adaptive_sampling': True,
                'early_stopping': True,
                'cache_results': True
            }
        }
        
        # Historical scoring data for statistical analysis
        self.scoring_history = []
        self.confidence_calibration_data = []
        
        # Training history and model performance tracking
        self.training_history = []
        self.model_performance = {}
        
        # Load existing model if available (will be done asynchronously when needed)
        # Note: Model loading is deferred to avoid blocking initialization

        # Embedding model configuration
        self.embedding_batch_size = 100 # Batch size for embedding requests
        self.embedding_cache_size = 10000 # Maximum number of embeddings to cache
        self.embedding_cache_ttl = 3600 # Cache TTL in seconds
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        self._embedding_semaphore = asyncio.Semaphore(10) # Semaphore for concurrent embedding requests
        self._last_embedding_cleanup = time.time()
        
        # Performance statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_generated': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
        }
    
    def _initialize_embedding_model(self) -> None:
        """Initialize embedding model with robust error handling."""
        try:
            # Try to load the embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")
        except ImportError as e:
            self.logger.warning(f"Embedding model dependencies not available: {e}")
            self.logger.info("Semantic similarity scoring will be disabled")
        except Exception as e:
            # Handle specific huggingface_hub compatibility issues
            if "split_torch_state_dict_into_shards" in str(e):
                self.logger.warning("HuggingFace Hub compatibility issue detected")
                self.logger.info("Semantic similarity scoring will be disabled")
            else:
                self.logger.warning(f"Failed to load embedding model: {e}")
                self.logger.info("Semantic similarity scoring will be disabled")
        
        if self.embedding_model is None:
            self.logger.info("Using fallback scoring methods (keyword-based only)")
    
    async def calculate_multi_factor_score(
        self,
        chunk: Dict[str, Any],
        query: str,
        user_id: Optional[str] = None,
    ) -> Any:
        """Calculate multi-factor score for a chunk (alias for score_chunks)."""
        # Convert chunk dict to ContextChunk for compatibility
        from ..models import ContextChunk, ContextSource, RelevanceScore
        from uuid import uuid4
        
        context_chunk = ContextChunk(
            id=uuid4(),
            content=chunk.get('content', ''),
            source=ContextSource(
                id=uuid4(),
                name=chunk.get('metadata', {}).get('source', 'unknown'),
                source_type='document'
            ),
            metadata=chunk.get('metadata', {})
        )
        
        scored_chunks = await self.score_chunks([context_chunk], query, user_id)
        
        if scored_chunks:
            scored_chunk = scored_chunks[0]
            return scored_chunk.relevance_score
        else:
            # Return a default relevance score
            return RelevanceScore(
                score=0.5,
                confidence_lower=0.4,
                confidence_upper=0.6
            )
    
    async def score_chunks(
        self,
        chunks: List[ContextChunk],
        query: str,
        user_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Score relevance of context chunks for a query.
        
        Args:
            chunks: List of context chunks to score
            query: User query
            user_id: Optional user ID for personalization
            
        Returns:
            List of chunks with relevance scores
        """
        if not chunks:
            return chunks
        
        self.logger.info(f"Scoring {len(chunks)} chunks for query: {query}")
        
        try:
            # Generate embeddings (if available)
            query_embedding = None
            chunk_embeddings = None
            
            if self.embedding_model:
                try:
                    query_embedding = await self._get_embedding(query)
                    chunk_embeddings = await self._get_embeddings([c.content for c in chunks])
                except Exception as e:
                    self.logger.warning(f"Embedding generation failed, falling back to keyword-based scoring: {e}")
                    query_embedding = None
                    chunk_embeddings = None
            
            # Score each chunk
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                relevance_score = await self._calculate_relevance_score(
                    chunk=chunk,
                    query=query,
                    query_embedding=query_embedding,
                    chunk_embedding=chunk_embeddings[i] if chunk_embeddings else None,
                    user_id=user_id,
                )
                
                chunk.relevance_score = relevance_score
                scored_chunks.append(chunk)
            
            # Sort by relevance score
            scored_chunks.sort(
                key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
                reverse=True
            )
            
            self.logger.info(
                f"Scored {len(scored_chunks)} chunks",
                avg_score=np.mean([c.relevance_score.score for c in scored_chunks if c.relevance_score]),
            )
            
            return scored_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to score chunks: {e}")
            raise RelevanceScoringError(str(e))
    
    async def _calculate_relevance_score(
        self,
        chunk: ContextChunk,
        query: str,
        query_embedding: Optional[List[float]],
        chunk_embedding: Optional[List[float]],
        user_id: Optional[str] = None,
    ) -> RelevanceScore:
        """
        Calculate comprehensive relevance score for a chunk.
        
        Args:
            chunk: Context chunk to score
            query: User query
            query_embedding: Query embedding vector
            chunk_embedding: Chunk embedding vector
            user_id: Optional user ID for personalization
            
        Returns:
            RelevanceScore with confidence interval
        """
        scores = {}
        
        # Semantic similarity
        if query_embedding and chunk_embedding:
            try:
                scores['semantic_similarity'] = await self._calculate_semantic_similarity(
                    query_embedding, chunk_embedding
                )
            except Exception as e:
                self.logger.warning(f"Semantic similarity calculation failed: {e}")
                scores['semantic_similarity'] = 0.5  # Fallback score
        else:
            scores['semantic_similarity'] = 0.5  # Default score when embeddings unavailable
        
        # Keyword overlap
        scores['keyword_overlap'] = await self._calculate_keyword_overlap(query, chunk.content)
        
        # Freshness score
        scores['freshness'] = await self._calculate_freshness_score(chunk)
        
        # Source authority
        scores['source_authority'] = await self._calculate_source_authority(chunk.source)
        
        # Content quality
        scores['content_quality'] = await self._calculate_content_quality(chunk)
        
        # User preference (if user_id provided)
        if user_id:
            scores['user_preference'] = await self._calculate_user_preference(
                chunk, user_id
            )
        else:
            scores['user_preference'] = 0.5  # Neutral score
        
        # Additional multi-factor scores
        scores['contextual_relevance'] = await self._calculate_contextual_relevance(
            chunk, query, query_embedding, chunk_embedding
        )
        
        scores['sentiment_alignment'] = await self._calculate_sentiment_alignment(
            chunk, query
        )
        
        scores['complexity_match'] = await self._calculate_complexity_match(
            chunk, query
        )
        
        scores['domain_expertise'] = await self._calculate_domain_expertise(
            chunk, query
        )
        
        # Calculate multi-ensemble score
        ensemble_score = await self._calculate_multi_ensemble_score(scores)
        
        # Calculate confidence interval
        confidence_interval = await self._calculate_confidence_interval(scores, ensemble_score)
        
        return RelevanceScore(
            score=ensemble_score,
            confidence_lower=confidence_interval[0],
            confidence_upper=confidence_interval[1],
            confidence_level=0.95,
            factors=scores,
        )
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text with caching and optimization."""
        if not self.embedding_model:
            return None
        
        # Check cache first
        cache_key = self._generate_embedding_cache_key(text)
        if cache_key in self.embedding_cache:
            cached_result = self.embedding_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.embedding_cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_result['embedding']
            else:
                # Expired cache entry
                del self.embedding_cache[cache_key]
        
        try:
            self.stats['cache_misses'] += 1
            # Use semaphore to limit concurrent embedding requests
            async with self._embedding_semaphore:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, self.embedding_model.encode, text
                )
                
                # Cache the result
                self._cache_embedding(cache_key, embedding.tolist())
                self.stats['embeddings_generated'] += 1
                
                return embedding.tolist()
        except Exception as e:
            self.logger.warning(f"Failed to get embedding: {e}")
            return None
    
    async def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for multiple texts with batching and optimization."""
        if not self.embedding_model:
            return None
        
        if not texts:
            return []
        
        try:
            # Check cache for all texts first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._generate_embedding_cache_key(text)
                if cache_key in self.embedding_cache:
                    cached_result = self.embedding_cache[cache_key]
                    if time.time() - cached_result['timestamp'] < self.embedding_cache_ttl:
                        self.stats['cache_hits'] += 1
                        cached_embeddings.append((i, cached_result['embedding']))
                        continue
                    else:
                        # Expired cache entry
                        del self.embedding_cache[cache_key]
                
                self.stats['cache_misses'] += 1
                uncached_texts.append(text)
                uncached_indices.append(i)
            
            # If all texts were cached, return them in order
            if not uncached_texts:
                cached_embeddings.sort(key=lambda x: x[0])
                return [emb for _, emb in cached_embeddings]
            
            # Process uncached texts in batches
            all_embeddings = [None] * len(texts)
            
            # Fill in cached embeddings
            for i, emb in cached_embeddings:
                all_embeddings[i] = emb
            
            # Process uncached texts in batches
            batch_size = self.embedding_batch_size
            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]
                
                # Use semaphore to limit concurrent embedding requests
                async with self._embedding_semaphore:
                    # Run in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    batch_embeddings = await loop.run_in_executor(
                        None, self.embedding_model.encode, batch_texts
                    )
                    
                    # Cache results and fill in embeddings
                    for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                        original_index = batch_indices[j]
                        cache_key = self._generate_embedding_cache_key(text)
                        self._cache_embedding(cache_key, embedding.tolist())
                        all_embeddings[original_index] = embedding.tolist()
                        self.stats['embeddings_generated'] += 1
            
            # Clean up cache periodically
            await self._cleanup_embedding_cache()
            
            return all_embeddings
            
        except Exception as e:
            self.logger.warning(f"Failed to get embeddings: {e}")
            return None
    
    def _generate_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        # Create a deterministic hash of the text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"emb_{text_hash}"
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Cache an embedding result."""
        # Clean up cache if it's too large
        if len(self.embedding_cache) >= self.embedding_cache_size:
            self._cleanup_embedding_cache_sync()
        
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def _cleanup_embedding_cache_sync(self) -> None:
        """Clean up expired cache entries synchronously."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.embedding_cache.items()
            if current_time - value['timestamp'] > self.embedding_cache_ttl
        ]
        
        for key in expired_keys:
            del self.embedding_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.embedding_cache) >= self.embedding_cache_size:
            sorted_items = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            items_to_remove = len(sorted_items) - self.embedding_cache_size + 100
            for key, _ in sorted_items[:items_to_remove]:
                del self.embedding_cache[key]
    
    async def _cleanup_embedding_cache(self) -> None:
        """Clean up expired cache entries asynchronously."""
        current_time = time.time()
        
        # Only cleanup every 5 minutes to avoid performance impact
        if current_time - self._last_embedding_cleanup < 300:
            return
        
        self._cleanup_embedding_cache_sync()
        self._last_embedding_cleanup = current_time
    
    async def _calculate_semantic_similarity(
        self,
        query_embedding: List[float],
        chunk_embedding: List[float],
    ) -> float:
        """Calculate semantic similarity between query and chunk."""
        try:
            similarity = cosine_similarity(
                [query_embedding], [chunk_embedding]
            )[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    async def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap between query and content."""
        try:
            # Simple keyword extraction and overlap calculation
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(content_words))
            return min(overlap / len(query_words), 1.0)
        except Exception as e:
            self.logger.warning(f"Failed to calculate keyword overlap: {e}")
            return 0.0
    
    async def _calculate_freshness_score(self, chunk: ContextChunk) -> float:
        """Calculate freshness score based on content age."""
        try:
            # Calculate days since creation
            from datetime import datetime, timezone
            age_days = (datetime.now(timezone.utc) - chunk.created_at).days
            
            # Exponential decay: newer content gets higher scores
            freshness_score = np.exp(-age_days / 365)  # Decay over 1 year
            return float(freshness_score)
        except Exception as e:
            self.logger.warning(f"Failed to calculate freshness score: {e}")
            return 0.5
    
    async def _calculate_source_authority(self, source) -> float:
        """Calculate source authority score."""
        try:
            # Base authority scores by source type
            base_authority_scores = {
                'document': 0.8,
                'api': 0.9,
                'database': 0.85,
                'realtime': 0.7,
                'vector': 0.6,
                'cache': 0.5,
            }
            
            base_score = base_authority_scores.get(source.source_type.value, 0.5)
            
            # Enhance with source-specific factors
            enhancement = 0.0
            
            # Domain authority (e.g., .edu, .gov domains)
            if hasattr(source, 'url') and source.url:
                domain = source.url.split('/')[2] if len(source.url.split('/')) > 2 else ''
                if domain.endswith('.edu'):
                    enhancement += 0.15
                elif domain.endswith('.gov'):
                    enhancement += 0.2
                elif domain.endswith('.org'):
                    enhancement += 0.1
                elif domain.endswith('.com'):
                    enhancement += 0.05
            
            # Source metadata enhancements
            if hasattr(source, 'metadata') and source.metadata:
                metadata = source.metadata
                
                # User ratings
                if 'user_rating' in metadata:
                    user_rating = float(metadata['user_rating'])
                    enhancement += user_rating * 0.1
                
                # Verification status
                if metadata.get('verified', False):
                    enhancement += 0.1
                
                # Update frequency
                if 'last_updated' in metadata:
                    try:
                        last_updated = datetime.fromisoformat(metadata['last_updated'])
                        days_old = (datetime.now(timezone.utc) - last_updated).days
                        if days_old < 30:
                            enhancement += 0.1
                        elif days_old < 90:
                            enhancement += 0.05
                    except:
                        pass
            
            # Source name patterns (e.g., official sources)
            if hasattr(source, 'name') and source.name:
                name_lower = source.name.lower()
                if any(keyword in name_lower for keyword in ['official', 'government', 'university', 'research']):
                    enhancement += 0.1
            
            final_score = min(1.0, base_score + enhancement)
            return float(final_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate source authority: {e}")
            return 0.5
    
    async def _calculate_content_quality(self, chunk: ContextChunk) -> float:
        """Calculate content quality score."""
        try:
            # Simple quality metrics
            content = chunk.content
            
            # Length score (not too short, not too long)
            length_score = min(len(content) / 1000, 1.0)  # Normalize to 1.0 at 1000 chars
            
            # Readability score (simple heuristic)
            sentences = content.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            readability_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)
            
            # Combine scores
            quality_score = (length_score + readability_score) / 2
            return float(quality_score)
        except Exception as e:
            self.logger.warning(f"Failed to calculate content quality: {e}")
            return 0.5
    
    async def _calculate_user_preference(self, chunk: ContextChunk, user_id: str) -> float:
        """Calculate user preference score based on historical interactions."""
        try:
            if not user_id:
                return 0.5
            
            # Get user interaction history (simulated)
            user_history = await self._get_user_interaction_history(user_id)
            
            if not user_history:
                return 0.5
            
            # Calculate preference based on historical interactions
            preference_score = 0.5  # Base neutral score
            
            # Source preference
            source_name = chunk.source.name if hasattr(chunk.source, 'name') else ''
            if source_name in user_history.get('preferred_sources', []):
                preference_score += 0.2
            
            # Content type preference
            content_type = chunk.metadata.get('content_type', 'text')
            if content_type in user_history.get('preferred_content_types', []):
                preference_score += 0.15
            
            # Topic preference (simple keyword matching)
            content_keywords = set(chunk.content.lower().split()[:10])  # First 10 words
            preferred_topics = user_history.get('preferred_topics', [])
            
            topic_overlap = len(content_keywords.intersection(set(preferred_topics)))
            if topic_overlap > 0:
                preference_score += min(0.1, topic_overlap * 0.02)
            
            # Recent interaction boost
            if chunk.source.name in user_history.get('recent_sources', []):
                preference_score += 0.05
            
            return min(1.0, max(0.0, preference_score))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate user preference: {e}")
            return 0.5
    
    async def _calculate_context_factor(self, chunk: ContextChunk, query: str) -> float:
        """Calculate context factor based on chunk position and surrounding context."""
        try:
            context_factor = 1.0  # Base factor
            
            # Position factor (chunks at the beginning might be more relevant)
            if hasattr(chunk, 'position') and chunk.position is not None:
                position = chunk.position
                if position < 0.2:  # First 20% of content
                    context_factor += 0.1
                elif position > 0.8:  # Last 20% of content (conclusions)
                    context_factor += 0.05
            
            # Metadata context indicators
            metadata = chunk.metadata
            
            # Title/heading indicators
            if metadata.get('is_title', False) or metadata.get('is_heading', False):
                context_factor += 0.15
            
            # Summary/conclusion indicators
            if metadata.get('is_summary', False) or metadata.get('is_conclusion', False):
                context_factor += 0.1
            
            # Section importance
            if metadata.get('section_importance', 0) > 0:
                context_factor += min(0.1, metadata['section_importance'] * 0.05)
            
            # Query context matching
            query_terms = set(query.lower().split())
            content_terms = set(chunk.content.lower().split())
            
            # Term frequency analysis
            term_overlap = len(query_terms.intersection(content_terms))
            if term_overlap > 0:
                context_factor += min(0.2, term_overlap * 0.05)
            
            return min(1.5, max(0.5, context_factor))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate context factor: {e}")
            return 1.0
    
    async def _get_user_interaction_history(self, user_id: str) -> Dict[str, Any]:
        """Get user interaction history from storage or generate realistic data."""
        try:
            # Try to retrieve from user interaction storage
            if hasattr(self, 'user_interaction_storage'):
                try:
                    history = await self.user_interaction_storage.get_user_history(user_id)
                    if history:
                        return history
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve user history from storage: {e}")
            
            # Generate realistic user interaction data based on user_id
            # This simulates what would be retrieved from a real user interaction database
            user_hash = hash(user_id) % 100  # Create deterministic user profile
            
            # Generate preferences based on user hash
            source_preferences = ['document', 'api', 'database', 'realtime']
            content_preferences = ['text', 'documentation', 'code', 'tutorial']
            topic_preferences = ['python', 'api', 'documentation', 'tutorial', 'development', 'data']
            
            # Select preferences based on user hash
            preferred_sources = source_preferences[:user_hash % 3 + 1]
            preferred_content_types = content_preferences[:user_hash % 2 + 1]
            preferred_topics = topic_preferences[:user_hash % 4 + 2]
            
            # Generate recent interactions
            recent_sources = preferred_sources[:user_hash % 2 + 1]
            
            # Generate interaction count based on user activity level
            interaction_count = 10 + (user_hash % 100)
            
            # Generate last interaction time
            days_ago = user_hash % 30
            last_interaction = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
            
            return {
                'preferred_sources': preferred_sources,
                'preferred_content_types': preferred_content_types,
                'preferred_topics': preferred_topics,
                'recent_sources': recent_sources,
                'interaction_count': interaction_count,
                'last_interaction': last_interaction,
                'user_activity_level': 'high' if interaction_count > 50 else 'medium' if interaction_count > 20 else 'low'
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get user interaction history: {e}")
            return {}
    
    async def _calculate_contextual_relevance(
        self,
        chunk: ContextChunk,
        query: str,
        query_embedding: Optional[List[float]],
        chunk_embedding: Optional[List[float]],
    ) -> float:
        """Calculate contextual relevance based on surrounding context."""
        try:
            if not query_embedding or not chunk_embedding:
                return 0.5
            
            # Calculate semantic similarity as base
            semantic_sim = await self._calculate_semantic_similarity(
                query_embedding, chunk_embedding
            )
            
            # Consider chunk position and context
            context_factor = await self._calculate_context_factor(chunk, query)
            
            # Consider query complexity vs chunk complexity
            query_complexity = len(query.split()) / 10.0  # Normalize
            chunk_complexity = len(chunk.content.split()) / 100.0  # Normalize
            complexity_match = 1.0 - abs(query_complexity - chunk_complexity)
            
            # Combine factors
            contextual_score = (semantic_sim * 0.6 + context_factor * 0.2 + complexity_match * 0.2)
            return float(contextual_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate contextual relevance: {e}")
            return 0.5
    
    async def _calculate_sentiment_alignment(self, chunk: ContextChunk, query: str) -> float:
        """Calculate sentiment alignment between query and content."""
        try:
            # Simple sentiment analysis (in production, use proper sentiment analysis)
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'poor']
            
            query_lower = query.lower()
            content_lower = chunk.content.lower()
            
            # Count sentiment words
            query_pos = sum(1 for word in positive_words if word in query_lower)
            query_neg = sum(1 for word in negative_words if word in query_lower)
            content_pos = sum(1 for word in positive_words if word in content_lower)
            content_neg = sum(1 for word in negative_words if word in content_lower)
            
            # Calculate sentiment scores
            query_sentiment = (query_pos - query_neg) / max(1, query_pos + query_neg)
            content_sentiment = (content_pos - content_neg) / max(1, content_pos + content_neg)
            
            # Calculate alignment (how well sentiments match)
            sentiment_alignment = 1.0 - abs(query_sentiment - content_sentiment) / 2.0
            return float(max(0.0, sentiment_alignment))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate sentiment alignment: {e}")
            return 0.5
    
    async def _calculate_complexity_match(self, chunk: ContextChunk, query: str) -> float:
        """Calculate complexity match between query and content."""
        try:
            # Calculate various complexity metrics
            query_words = query.split()
            content_words = chunk.content.split()
            
            # Average word length
            query_avg_length = np.mean([len(word) for word in query_words]) if query_words else 0
            content_avg_length = np.mean([len(word) for word in content_words]) if content_words else 0
            
            # Vocabulary complexity (unique words ratio)
            query_vocab_ratio = len(set(query_words)) / max(1, len(query_words))
            content_vocab_ratio = len(set(content_words)) / max(1, len(content_words))
            
            # Sentence complexity
            query_sentences = query.split('.')
            content_sentences = chunk.content.split('.')
            query_avg_sentence_length = np.mean([len(s.split()) for s in query_sentences if s.strip()])
            content_avg_sentence_length = np.mean([len(s.split()) for s in content_sentences if s.strip()])
            
            # Normalize and combine metrics
            length_match = 1.0 - abs(query_avg_length - content_avg_length) / 10.0
            vocab_match = 1.0 - abs(query_vocab_ratio - content_vocab_ratio)
            sentence_match = 1.0 - abs(query_avg_sentence_length - content_avg_sentence_length) / 20.0
            
            complexity_score = (length_match + vocab_match + sentence_match) / 3.0
            return float(max(0.0, min(1.0, complexity_score)))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate complexity match: {e}")
            return 0.5
    
    async def _calculate_domain_expertise(self, chunk: ContextChunk, query: str) -> float:
        """Calculate domain expertise alignment."""
        try:
            # Define domain keywords (in production, use proper domain classification)
            domains = {
                'technical': ['algorithm', 'implementation', 'code', 'function', 'method', 'class', 'api'],
                'academic': ['research', 'study', 'analysis', 'methodology', 'hypothesis', 'conclusion'],
                'business': ['strategy', 'market', 'revenue', 'profit', 'customer', 'product'],
                'medical': ['treatment', 'diagnosis', 'symptoms', 'patient', 'clinical', 'therapy'],
                'legal': ['law', 'regulation', 'compliance', 'contract', 'legal', 'statute'],
            }
            
            query_lower = query.lower()
            content_lower = chunk.content.lower()
            
            # Find matching domains
            query_domains = []
            content_domains = []
            
            for domain, keywords in domains.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_domains.append(domain)
                if any(keyword in content_lower for keyword in keywords):
                    content_domains.append(domain)
            
            # Calculate domain overlap
            if not query_domains or not content_domains:
                return 0.3  # Neutral score for no domain match
            
            domain_overlap = len(set(query_domains) & set(content_domains))
            total_domains = len(set(query_domains) | set(content_domains))
            
            domain_score = domain_overlap / total_domains if total_domains > 0 else 0.0
            
            # Boost score for exact domain matches
            if domain_overlap > 0:
                domain_score = min(1.0, domain_score + 0.2)
            
            return float(domain_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate domain expertise: {e}")
            return 0.5
    
    def _calculate_ensemble_score(self, scores: dict) -> float:
        """Calculate weighted ensemble score from individual scores."""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for score_type, score in scores.items():
                weight = self.scoring_weights.get(score_type, 0.0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.5
            
            return weighted_sum / total_weight
        except Exception as e:
            self.logger.warning(f"Failed to calculate ensemble score: {e}")
            return 0.5
    
    async def _calculate_multi_ensemble_score(self, scores: dict) -> float:
        """Calculate multi-ensemble score using multiple methods."""
        try:
            score_values = list(scores.values())
            n_scores = len(score_values)
            
            if n_scores < self.ensemble_config['min_scores_for_ensemble']:
                # Fallback to simple weighted average
                return self._calculate_ensemble_score(scores)
            
            ensemble_scores = {}
            
            # Method 1: Weighted Average
            ensemble_scores['weighted_average'] = self._calculate_ensemble_score(scores)
            
            # Method 2: Geometric Mean
            ensemble_scores['geometric_mean'] = self._calculate_geometric_mean(score_values)
            
            # Method 3: Harmonic Mean
            ensemble_scores['harmonic_mean'] = self._calculate_harmonic_mean(score_values)
            
            # Method 4: Trimmed Mean
            ensemble_scores['trimmed_mean'] = self._calculate_trimmed_mean(score_values)
            
            # Method 5: ML Ensemble (if available)
            if self.ensemble_config['use_ml_ensemble'] and self._is_trained:
                ml_score = await self._calculate_ml_ensemble_score(scores)
                if ml_score is not None:
                    ensemble_scores['ml_ensemble'] = ml_score
            
            # Combine ensemble methods
            final_score = self._combine_ensemble_methods(ensemble_scores)
            
            return float(final_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate multi-ensemble score: {e}")
            return self._calculate_ensemble_score(scores)
    
    def _calculate_geometric_mean(self, values: List[float]) -> float:
        """Calculate geometric mean of scores."""
        try:
            # Filter out zero values to avoid log(0)
            positive_values = [v for v in values if v > 0]
            if not positive_values:
                return 0.0
            
            # Calculate geometric mean
            log_sum = sum(np.log(v) for v in positive_values)
            geometric_mean = np.exp(log_sum / len(positive_values))
            
            return float(geometric_mean)
        except Exception as e:
            self.logger.warning(f"Failed to calculate geometric mean: {e}")
            return np.mean(values) if values else 0.5
    
    def _calculate_harmonic_mean(self, values: List[float]) -> float:
        """Calculate harmonic mean of scores."""
        try:
            # Filter out zero values to avoid division by zero
            positive_values = [v for v in values if v > 0]
            if not positive_values:
                return 0.0
            
            # Calculate harmonic mean
            reciprocal_sum = sum(1.0 / v for v in positive_values)
            harmonic_mean = len(positive_values) / reciprocal_sum
            
            return float(harmonic_mean)
        except Exception as e:
            self.logger.warning(f"Failed to calculate harmonic mean: {e}")
            return np.mean(values) if values else 0.5
    
    def _calculate_trimmed_mean(self, values: List[float]) -> float:
        """Calculate trimmed mean of scores."""
        try:
            if len(values) < 3:
                return np.mean(values) if values else 0.5
            
            # Sort values
            sorted_values = sorted(values)
            
            # Calculate trim size
            trim_size = int(len(sorted_values) * self.ensemble_config['trim_percentage'])
            trim_size = max(0, min(trim_size, len(sorted_values) // 2))
            
            # Trim from both ends
            trimmed_values = sorted_values[trim_size:-trim_size] if trim_size > 0 else sorted_values
            
            return float(np.mean(trimmed_values))
        except Exception as e:
            self.logger.warning(f"Failed to calculate trimmed mean: {e}")
            return np.mean(values) if values else 0.5
    
    async def _calculate_ml_ensemble_score(self, scores: dict) -> Optional[float]:
        """Calculate ML ensemble score using trained model."""
        try:
            if not self._is_trained:
                return None
            
            # Extract features for ML model
            features = await self._extract_features_for_ml(scores)
            
            # Predict score using ML model
            prediction = self.ml_model.predict([features])[0]
            
            return float(max(0.0, min(1.0, prediction)))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate ML ensemble score: {e}")
            return None
    
    def _combine_ensemble_methods(self, ensemble_scores: dict) -> float:
        """Combine multiple ensemble methods using weighted average."""
        try:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, score in ensemble_scores.items():
                weight = self.ensemble_config['ensemble_weights'].get(method, 0.1)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return np.mean(list(ensemble_scores.values()))
            
            return weighted_sum / total_weight
            
        except Exception as e:
            self.logger.warning(f"Failed to combine ensemble methods: {e}")
            return np.mean(list(ensemble_scores.values())) if ensemble_scores else 0.5
    
    async def _extract_features_for_ml(self, scores: dict) -> List[float]:
        """Extract features for ML ensemble training."""
        try:
            # Basic statistical features
            score_values = list(scores.values())
            
            features = [
                np.mean(score_values),  # Mean score
                np.std(score_values),   # Standard deviation
                np.min(score_values),   # Minimum score
                np.max(score_values),   # Maximum score
                np.median(score_values), # Median score
                len(score_values),      # Number of scores
            ]
            
            # Add individual score features
            for score_type in self.scoring_weights.keys():
                features.append(scores.get(score_type, 0.5))
            
            # Add interaction features
            features.extend([
                np.mean(score_values) * np.std(score_values),  # Mean * Std interaction
                np.max(score_values) - np.min(score_values),   # Score range
                np.percentile(score_values, 25),               # 25th percentile
                np.percentile(score_values, 75),               # 75th percentile
            ])
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract ML features: {e}")
            return [0.5] * 20  # Return neutral features
    
    async def _calculate_confidence_interval(
        self,
        scores: dict,
        ensemble_score: float,
    ) -> tuple[float, float]:
        """Calculate comprehensive confidence interval for the ensemble score."""
        try:
            score_values = list(scores.values())
            n_scores = len(score_values)
            
            if n_scores < self.confidence_config['min_sample_size']:
                # Fallback to simple interval for small sample sizes
                return await self._calculate_simple_confidence_interval(
                    scores, ensemble_score
                )
            
            # Try multiple statistical methods and combine results
            confidence_intervals = []
            
            # Method 1: Bootstrap confidence interval
            if self.confidence_config['use_bootstrap'] and n_scores >= 5:
                bootstrap_ci = await self._calculate_bootstrap_confidence_interval(
                    score_values, ensemble_score
                )
                if bootstrap_ci:
                    confidence_intervals.append(bootstrap_ci)
            
            # Method 2: T-distribution confidence interval
            if self.confidence_config['use_t_distribution']:
                t_ci = await self._calculate_t_confidence_interval(
                    score_values, ensemble_score
                )
                if t_ci:
                    confidence_intervals.append(t_ci)
            
            # Method 3: Normal distribution confidence interval
            normal_ci = await self._calculate_normal_confidence_interval(
                score_values, ensemble_score
            )
            if normal_ci:
                confidence_intervals.append(normal_ci)
            
            # Method 4: Weighted confidence interval
            if self.confidence_config['use_weighted_stats']:
                weighted_ci = await self._calculate_weighted_confidence_interval(
                    scores, ensemble_score
                )
                if weighted_ci:
                    confidence_intervals.append(weighted_ci)
            
            # Combine confidence intervals using robust statistics
            if confidence_intervals:
                return await self._combine_confidence_intervals(confidence_intervals)
            else:
                # Fallback to simple method
                return await self._calculate_simple_confidence_interval(
                    scores, ensemble_score
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence interval: {e}")
            return await self._calculate_simple_confidence_interval(scores, ensemble_score)
    
    async def _calculate_simple_confidence_interval(
        self,
        scores: dict,
        ensemble_score: float,
    ) -> tuple[float, float]:
        """Calculate simple confidence interval using standard error."""
        try:
            score_values = list(scores.values())
            std_dev = np.std(score_values)
            n = len(score_values)
            
            # Standard error
            standard_error = std_dev / np.sqrt(n)
            
            # Z-score for 95% confidence
            z_score = norm.ppf(0.975)  # 95% confidence level
            
            margin_of_error = z_score * standard_error
            
            lower_bound = max(0.0, ensemble_score - margin_of_error)
            upper_bound = min(1.0, ensemble_score + margin_of_error)
            
            return float(lower_bound), float(upper_bound)
        except Exception as e:
            self.logger.warning(f"Failed to calculate simple confidence interval: {e}")
            return 0.0, 1.0
    
    async def _calculate_bootstrap_confidence_interval(
        self,
        score_values: List[float],
        ensemble_score: float,
    ) -> Optional[tuple[float, float]]:
        """Calculate enhanced bootstrap confidence interval with multiple strategies."""
        try:
            # Validate data quality
            if not self._validate_data_quality(score_values):
                self.logger.warning("Data quality validation failed for bootstrap CI")
                return None
            
            n = len(score_values)
            if n < self.confidence_config['min_sample_size']:
                return None
            
            # Detect and handle outliers if enabled
            if self.confidence_config['outlier_detection']:
                clean_data, outlier_info = self._detect_and_handle_outliers(score_values)
                if outlier_info['outlier_ratio'] > self.confidence_config['validation_thresholds']['max_outlier_ratio']:
                    self.logger.warning(f"High outlier ratio detected: {outlier_info['outlier_ratio']:.3f}")
                score_values = clean_data
                n = len(score_values)
                if n < self.confidence_config['min_sample_size']:
                    return None
            
            # Test normality if enabled
            if self.confidence_config['normality_testing']:
                normality_result = self._test_normality(score_values)
                if not normality_result['is_normal']:
                    self.logger.info(f"Data appears non-normal (p={normality_result['p_value']:.4f})")
            
            # Select optimal bootstrap strategy
            normality_result = self._test_normality(score_values) if self.confidence_config['normality_testing'] else {'is_normal': True, 'p_value': 1.0}
            strategy = self._select_bootstrap_strategy(score_values, normality_result)
            self.logger.info(f"Selected bootstrap strategy: {strategy}")
            
            # Calculate confidence interval using selected strategy
            ci_result = None
            
            if strategy == 'percentile':
                ci_result = await self._percentile_bootstrap(score_values, ensemble_score)
            elif strategy == 'bca':
                ci_result = await self._bca_bootstrap(score_values, ensemble_score)
            elif strategy == 'abc':
                ci_result = await self._abc_bootstrap(score_values, ensemble_score)
            elif strategy == 'studentized':
                ci_result = await self._studentized_bootstrap(score_values, ensemble_score)
            else:
                # Fallback to percentile method
                ci_result = await self._percentile_bootstrap(score_values, ensemble_score)
            
            if ci_result is None:
                self.logger.warning("Primary bootstrap strategy failed, trying fallback methods")
                # Try fallback methods
                for fallback in self.confidence_config['fallback_methods']:
                    if fallback == 'simple':
                        ci_result = await self._percentile_bootstrap(score_values, ensemble_score)
                    elif fallback == 'robust':
                        ci_result = await self._calculate_robust_confidence_interval(score_values, ensemble_score)
                    elif fallback == 'nonparametric':
                        # Use basic percentile as nonparametric fallback
                        ci_result = await self._percentile_bootstrap(score_values, ensemble_score)
                    
                    if ci_result is not None:
                        self.logger.info(f"Fallback method '{fallback}' succeeded")
                        break
            
            if ci_result is None:
                self.logger.error("All bootstrap methods failed")
                return None
            
            # Validate the confidence interval
            if not self._validate_confidence_interval(ci_result, score_values):
                self.logger.warning("Generated confidence interval failed validation")
                # Try to generate a valid one using robust method
                ci_result = await self._calculate_robust_confidence_interval(score_values, ensemble_score)
            
            return ci_result
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate bootstrap confidence interval: {e}")
            # Final fallback to robust method
            try:
                return await self._calculate_robust_confidence_interval(score_values, ensemble_score)
            except Exception as fallback_error:
                self.logger.error(f"Even fallback method failed: {fallback_error}")
                return None
    
    async def _calculate_t_confidence_interval(
        self,
        score_values: List[float],
        ensemble_score: float,
    ) -> Optional[tuple[float, float]]:
        """Calculate t-distribution confidence interval."""
        try:
            n = len(score_values)
            if n < 2:
                return None
            
            # Calculate sample statistics
            sample_mean = np.mean(score_values)
            sample_std = np.std(score_values, ddof=1)  # Sample standard deviation
            
            # T-distribution confidence interval
            alpha = 1 - self.confidence_config['default_confidence_level']
            t_value = t.ppf(1 - alpha / 2, df=n - 1)
            
            margin_of_error = t_value * sample_std / np.sqrt(n)
            
            # Adjust for ensemble score
            center_offset = ensemble_score - sample_mean
            lower_bound = max(0.0, sample_mean - margin_of_error + center_offset)
            upper_bound = min(1.0, sample_mean + margin_of_error + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate t-confidence interval: {e}")
            return None
    
    async def _calculate_normal_confidence_interval(
        self,
        score_values: List[float],
        ensemble_score: float,
    ) -> Optional[tuple[float, float]]:
        """Calculate normal distribution confidence interval."""
        try:
            n = len(score_values)
            if n < 2:  # Require at least 2 samples for statistical validity
                return None
            
            # Calculate sample statistics
            sample_mean = np.mean(score_values)
            sample_std = np.std(score_values)
            
            # Normal distribution confidence interval
            alpha = 1 - self.confidence_config['default_confidence_level']
            z_value = norm.ppf(1 - alpha / 2)
            
            margin_of_error = z_value * sample_std / np.sqrt(n)
            
            # Adjust for ensemble score
            center_offset = ensemble_score - sample_mean
            lower_bound = max(0.0, sample_mean - margin_of_error + center_offset)
            upper_bound = min(1.0, sample_mean + margin_of_error + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate normal confidence interval: {e}")
            return None
    
    async def _calculate_weighted_confidence_interval(
        self,
        scores: dict,
        ensemble_score: float,
    ) -> Optional[tuple[float, float]]:
        """Calculate weighted confidence interval using scoring weights."""
        try:
            score_values = list(scores.values())
            weights = [self.scoring_weights.get(key, 0.1) for key in scores.keys()]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                return None
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Use weighted statistics
            weighted_stats = DescrStatsW(score_values, weights=normalized_weights)
            
            # Calculate weighted confidence interval
            alpha = 1 - self.confidence_config['default_confidence_level']
            lower_bound, upper_bound = weighted_stats.tconfint_mean(alpha=alpha)
            
            # Adjust for ensemble score
            center_offset = ensemble_score - weighted_stats.mean
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate weighted confidence interval: {e}")
            return None
    
    async def _combine_confidence_intervals(
        self,
        confidence_intervals: List[tuple[float, float]],
    ) -> tuple[float, float]:
        """Combine multiple confidence intervals using robust statistics."""
        try:
            if not confidence_intervals:
                return 0.0, 1.0
            
            # Extract lower and upper bounds
            lower_bounds = [ci[0] for ci in confidence_intervals]
            upper_bounds = [ci[1] for ci in confidence_intervals]
            
            # Use median for robust combination
            combined_lower = np.median(lower_bounds)
            combined_upper = np.median(upper_bounds)
            
            # Ensure bounds are valid
            combined_lower = max(0.0, min(combined_lower, combined_upper))
            combined_upper = min(1.0, max(combined_upper, combined_lower))
            
            return float(combined_lower), float(combined_upper)
            
        except Exception as e:
            self.logger.warning(f"Failed to combine confidence intervals: {e}")
            return 0.0, 1.0
    
    # ===== ENHANCED CONFIDENCE INTERVAL HELPER METHODS =====
    
    def _validate_data_quality(self, score_values: List[float]) -> bool:
        """Validate data quality for statistical analysis."""
        try:
            if not score_values or len(score_values) < 2:
                return False
            
            # Check for finite values
            if not all(np.isfinite(score) for score in score_values):
                return False
            
            # Check for reasonable range (0-1 for scores)
            if not all(0.0 <= score <= 1.0 for score in score_values):
                return False
            
            # Check for sufficient variation
            if np.std(score_values) < 1e-10:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Data quality validation failed: {e}")
            return False
    
    def _detect_and_handle_outliers(self, score_values: List[float]) -> tuple[List[float], dict]:
        """Detect and handle outliers using multiple methods."""
        try:
            if len(score_values) < 4:
                return score_values, {'outlier_ratio': 0.0, 'method': 'insufficient_data'}
            
            # Method 1: IQR-based outlier detection
            q1 = np.percentile(score_values, 25)
            q3 = np.percentile(score_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_iqr = [i for i, score in enumerate(score_values) 
                           if score < lower_bound or score > upper_bound]
            
            # Method 2: Z-score based outlier detection
            mean_score = np.mean(score_values)
            std_score = np.std(score_values)
            if std_score > 0:
                z_scores = [(score - mean_score) / std_score for score in score_values]
                outliers_zscore = [i for i, z in enumerate(z_scores) if abs(z) > 2.5]
            else:
                outliers_zscore = []
            
            # Method 3: Modified Z-score (more robust)
            median_score = np.median(score_values)
            mad = np.median([abs(score - median_score) for score in score_values])
            if mad > 0:
                modified_z_scores = [0.6745 * (score - median_score) / mad for score in score_values]
                outliers_mzscore = [i for i, mz in enumerate(modified_z_scores) if abs(mz) > 3.5]
            else:
                outliers_mzscore = []
            
            # Combine outlier detection methods
            all_outlier_indices = set(outliers_iqr) | set(outliers_zscore) | set(outliers_mzscore)
            outlier_ratio = len(all_outlier_indices) / len(score_values)
            
            # Handle outliers based on configuration
            if outlier_ratio <= self.confidence_config['validation_thresholds']['max_outlier_ratio']:
                # Remove outliers for analysis
                clean_scores = [score for i, score in enumerate(score_values) 
                              if i not in all_outlier_indices]
                if len(clean_scores) < 2:
                    clean_scores = score_values  # Keep original if too many outliers
            else:
                # Too many outliers, use robust methods
                clean_scores = score_values
            
            outlier_info = {
                'outlier_ratio': outlier_ratio,
                'outlier_indices': list(all_outlier_indices),
                'method': 'combined_detection',
                'iqr_outliers': len(outliers_iqr),
                'zscore_outliers': len(outliers_zscore),
                'mzscore_outliers': len(outliers_mzscore)
            }
            
            return clean_scores, outlier_info
            
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}")
            return score_values, {'outlier_ratio': 0.0, 'method': 'error_fallback'}
    
    def _test_normality(self, score_values: List[float]) -> dict:
        """Test for normality using multiple statistical tests."""
        try:
            if len(score_values) < 3:
                return {'is_normal': False, 'p_value': 0.0, 'method': 'insufficient_data'}
            
            # Method 1: Shapiro-Wilk test (most powerful for small samples)
            try:
                from scipy.stats import shapiro
                shapiro_stat, shapiro_p = shapiro(score_values)
                shapiro_normal = shapiro_p > 0.05
            except ImportError:
                shapiro_stat, shapiro_p = 0.0, 0.0
                shapiro_normal = False
            
            # Method 2: Anderson-Darling test
            try:
                from scipy.stats import anderson
                anderson_result = anderson(score_values)
                anderson_normal = anderson_result.statistic < anderson_result.critical_values[2]  # 5% level
                anderson_p = 0.05 if anderson_normal else 0.01
            except ImportError:
                anderson_result = None
                anderson_normal = False
                anderson_p = 0.0
            
            # Method 3: Kolmogorov-Smirnov test
            try:
                from scipy.stats import kstest
                ks_stat, ks_p = kstest(score_values, 'norm', 
                                     args=(np.mean(score_values), np.std(score_values)))
                ks_normal = ks_p > 0.05
            except ImportError:
                ks_stat, ks_p = 0.0, 0.0
                ks_normal = False
            
            # Combine test results
            normality_tests = [shapiro_normal, anderson_normal, ks_normal]
            normal_count = sum(normality_tests)
            is_normal = normal_count >= 2  # At least 2 out of 3 tests should pass
            
            # Calculate combined p-value (geometric mean)
            p_values = [shapiro_p, anderson_p, ks_p]
            valid_p_values = [p for p in p_values if p > 0]
            if valid_p_values:
                combined_p = np.exp(np.mean(np.log(valid_p_values)))
            else:
                combined_p = 0.0
            
            return {
                'is_normal': is_normal,
                'p_value': combined_p,
                'method': 'combined_tests',
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'normal': shapiro_normal},
                'anderson': {'statistic': anderson_result.statistic if anderson_result else 0.0, 'p_value': anderson_p, 'normal': anderson_normal},
                'ks': {'statistic': ks_stat, 'p_value': ks_p, 'normal': ks_normal},
                'normal_count': normal_count,
                'total_tests': len(normality_tests)
            }
            
        except Exception as e:
            self.logger.warning(f"Normality testing failed: {e}")
            return {'is_normal': False, 'p_value': 0.0, 'method': 'error_fallback'}
    
    def _test_homoscedasticity(self, score_values: List[float]) -> dict:
        """Test for homoscedasticity (constant variance)."""
        try:
            if len(score_values) < 4:
                return {'is_homoscedastic': True, 'p_value': 1.0, 'method': 'insufficient_data'}
            
            # Method 1: Levene's test for homogeneity of variances
            try:
                from scipy.stats import levene
                # Split data into two groups for testing
                mid_point = len(score_values) // 2
                group1 = score_values[:mid_point]
                group2 = score_values[mid_point:]
                
                if len(group1) >= 2 and len(group2) >= 2:
                    levene_stat, levene_p = levene(group1, group2)
                    is_homoscedastic = levene_p > 0.05
                else:
                    levene_stat, levene_p = 0.0, 1.0
                    is_homoscedastic = True
            except ImportError:
                levene_stat, levene_p = 0.0, 1.0
                is_homoscedastic = True
            
            # Method 2: Bartlett's test (assumes normality)
            try:
                from scipy.stats import bartlett
                if len(group1) >= 2 and len(group2) >= 2:
                    bartlett_stat, bartlett_p = bartlett(group1, group2)
                    bartlett_homoscedastic = bartlett_p > 0.05
                else:
                    bartlett_stat, bartlett_p = 0.0, 1.0
                    bartlett_homoscedastic = True
            except ImportError:
                bartlett_stat, bartlett_p = 0.0, 1.0
                bartlett_homoscedastic = True
            
            # Combine test results
            homoscedastic_tests = [is_homoscedastic, bartlett_homoscedastic]
            homoscedastic_count = sum(homoscedastic_tests)
            is_homoscedastic_final = homoscedastic_count >= 1  # At least 1 test should pass
            
            # Calculate combined p-value
            p_values = [levene_p, bartlett_p]
            valid_p_values = [p for p in p_values if p > 0]
            if valid_p_values:
                combined_p = np.exp(np.mean(np.log(valid_p_values)))
            else:
                combined_p = 1.0
            
            return {
                'is_homoscedastic': is_homoscedastic_final,
                'p_value': combined_p,
                'method': 'combined_tests',
                'levene': {'statistic': levene_stat, 'p_value': levene_p, 'homoscedastic': is_homoscedastic},
                'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_p, 'homoscedastic': bartlett_homoscedastic},
                'homoscedastic_count': homoscedastic_count,
                'total_tests': len(homoscedastic_tests)
            }
            
        except Exception as e:
            self.logger.warning(f"Homoscedasticity testing failed: {e}")
            return {'is_homoscedastic': True, 'p_value': 1.0, 'method': 'error_fallback'}
    
    def _select_bootstrap_strategy(self, score_values: List[float], normality_test: dict) -> str:
        """Select optimal bootstrap strategy based on data characteristics."""
        try:
            n = len(score_values)
            
            if n < 5:
                return 'percentile'
            elif n < 8:
                return 'percentile'
            elif n < 10:
                return 'abc' if normality_test['is_normal'] else 'percentile'
            elif n < 15:
                return 'bca' if normality_test['is_normal'] else 'abc'
            else:
                if normality_test['is_normal']:
                    return 'bca'  # Most accurate for normal data
                else:
                    return 'studentized'  # Most robust for non-normal data
                    
        except Exception as e:
            self.logger.warning(f"Bootstrap strategy selection failed: {e}")
            return 'percentile'
    
    async def _combine_bootstrap_results(self, bootstrap_results: dict, score_values: List[float], ensemble_score: float) -> tuple[float, float]:
        """Combine multiple bootstrap confidence interval results."""
        try:
            if not bootstrap_results:
                return 0.0, 1.0
            
            # Extract confidence intervals
            intervals = list(bootstrap_results.values())
            
            # Calculate robust statistics
            lower_bounds = [ci[0] for ci in intervals]
            upper_bounds = [ci[1] for ci in intervals]
            
            # Use trimmed mean for robust combination
            trim_percent = 0.1
            n_trim = max(1, int(len(intervals) * trim_percent))
            
            sorted_lower = sorted(lower_bounds)
            sorted_upper = sorted(upper_bounds)
            
            trimmed_lower = sorted_lower[n_trim:-n_trim] if len(sorted_lower) > 2 * n_trim else sorted_lower
            trimmed_upper = sorted_upper[n_trim:-n_trim] if len(sorted_upper) > 2 * n_trim else sorted_upper
            
            combined_lower = np.mean(trimmed_lower)
            combined_upper = np.mean(trimmed_upper)
            
            # Ensure bounds are valid
            combined_lower = max(0.0, min(combined_lower, combined_upper))
            combined_upper = min(1.0, max(combined_upper, combined_lower))
            
            return float(combined_lower), float(combined_upper)
            
        except Exception as e:
            self.logger.warning(f"Bootstrap results combination failed: {e}")
            return 0.0, 1.0
    
    def _validate_confidence_interval(self, confidence_interval: tuple[float, float], score_values: List[float]) -> bool:
        """Validate confidence interval quality."""
        try:
            lower_bound, upper_bound = confidence_interval
            
            # Check bounds are finite
            if not (np.isfinite(lower_bound) and np.isfinite(upper_bound)):
                return False
            
            # Check bounds are in valid range
            if not (0.0 <= lower_bound <= 1.0 and 0.0 <= upper_bound <= 1.0):
                return False
            
            # Check lower bound is less than upper bound
            if lower_bound >= upper_bound:
                return False
            
            # Check confidence interval width
            width = upper_bound - lower_bound
            if width < self.confidence_config['validation_thresholds']['min_confidence_width']:
                return False
            if width > self.confidence_config['validation_thresholds']['max_confidence_width']:
                return False
            
            # Check bounds are reasonable relative to data
            if score_values:
                data_mean = np.mean(score_values)
                if abs(lower_bound - data_mean) > 0.5 or abs(upper_bound - data_mean) > 0.5:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Confidence interval validation failed: {e}")
            return False
    
    async def _calculate_robust_confidence_interval(self, score_values: List[float], ensemble_score: float) -> tuple[float, float]:
        """Calculate robust confidence interval using nonparametric methods."""
        try:
            if len(score_values) < 2:
                return 0.0, 1.0
            
            # Use median and MAD for robust estimation
            median_score = np.median(score_values)
            mad = np.median([abs(score - median_score) for score in score_values])
            
            if mad <= 0:
                # Fallback to simple method
                return await self._calculate_simple_confidence_interval(
                    {'scores': score_values}, ensemble_score
                )
            
            # Calculate robust confidence interval
            # For 95% confidence, use 1.96 * MAD / sqrt(n)
            n = len(score_values)
            robust_se = 1.96 * mad / np.sqrt(n)
            
            lower_bound = max(0.0, median_score - robust_se)
            upper_bound = min(1.0, median_score + robust_se)
            
            # Adjust bounds relative to ensemble score
            center_offset = ensemble_score - median_score
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Robust confidence interval calculation failed: {e}")
            return 0.0, 1.0
    
    async def update_scoring_weights(self, new_weights: dict) -> None:
        """Update scoring weights based on feedback or optimization."""
        self.scoring_weights.update(new_weights)
        self.logger.info("Updated scoring weights", new_weights=new_weights)
    
    async def update_ensemble_config(self, new_config: dict) -> None:
        """Update ensemble configuration."""
        self.ensemble_config.update(new_config)
        self.logger.info("Updated ensemble configuration", new_config=new_config)
    
    async def optimize_ensemble_weights(self, validation_data: List[Dict[str, Any]]) -> None:
        """Optimize ensemble weights using validation data."""
        try:
            self.logger.info(f"Optimizing ensemble weights with {len(validation_data)} samples")
            
            # Simple grid search optimization
            best_score = 0.0
            best_weights = self.ensemble_config['ensemble_weights'].copy()
            
            # Test different weight combinations
            weight_combinations = [
                {'weighted_average': 0.4, 'geometric_mean': 0.2, 'harmonic_mean': 0.2, 'trimmed_mean': 0.2},
                {'weighted_average': 0.5, 'geometric_mean': 0.2, 'harmonic_mean': 0.2, 'trimmed_mean': 0.1},
                {'weighted_average': 0.3, 'geometric_mean': 0.3, 'harmonic_mean': 0.2, 'trimmed_mean': 0.2},
                {'weighted_average': 0.6, 'geometric_mean': 0.1, 'harmonic_mean': 0.1, 'trimmed_mean': 0.2},
                {'weighted_average': 0.2, 'geometric_mean': 0.4, 'harmonic_mean': 0.2, 'trimmed_mean': 0.2},
            ]
            
            for weights in weight_combinations:
                # Temporarily set weights
                original_weights = self.ensemble_config['ensemble_weights'].copy()
                self.ensemble_config['ensemble_weights'] = weights
                
                # Evaluate on validation data
                total_score = 0.0
                for sample in validation_data:
                    predicted_score = sample.get('predicted_score', 0.5)
                    actual_score = sample.get('actual_score', 0.5)
                    
                    # Calculate accuracy (how close prediction is to actual)
                    accuracy = 1.0 - abs(predicted_score - actual_score)
                    total_score += accuracy
                
                avg_score = total_score / len(validation_data)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_weights = weights.copy()
                
                # Restore original weights
                self.ensemble_config['ensemble_weights'] = original_weights
            
            # Apply best weights
            self.ensemble_config['ensemble_weights'] = best_weights
            self.logger.info(f"Optimized ensemble weights: {best_weights} (score: {best_score:.3f})")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize ensemble weights: {e}")
    
    async def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble scoring statistics."""
        try:
            stats = {
                'scoring_factors': len(self.scoring_weights),
                'ensemble_methods': len(self.ensemble_config['ensemble_weights']),
                'ml_ensemble_enabled': self.ensemble_config['use_ml_ensemble'],
                'ml_model_trained': self._is_trained,
                'primary_method': self.ensemble_config['primary_method'],
                'secondary_methods': self.ensemble_config['secondary_methods'],
                'ensemble_weights': self.ensemble_config['ensemble_weights'],
                'scoring_weights': self.scoring_weights,
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get ensemble statistics: {e}")
            return {}
    
    async def train_on_feedback(
        self,
        query_chunk_pairs: List[tuple],
        relevance_feedback: List[float],
        validation_split: Optional[float] = None,
        enable_cross_validation: bool = True,
        enable_hyperparameter_optimization: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive ML model training with validation, cross-validation, and hyperparameter optimization.
        
        Args:
            query_chunk_pairs: List of (query, chunk) pairs for training
            relevance_feedback: List of relevance scores for training
            validation_split: Fraction of data to use for validation (default from config)
            enable_cross_validation: Whether to perform cross-validation
            enable_hyperparameter_optimization: Whether to optimize hyperparameters
            
        Returns:
            Dictionary containing training results and performance metrics
        """
        try:
            if len(query_chunk_pairs) < 10:
                self.logger.warning("Insufficient training data. Need at least 10 samples.")
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Extract comprehensive features
            features = []
            for query, chunk in query_chunk_pairs:
                feature_vector = await self._extract_features(query, chunk)
                features.append(feature_vector)
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(relevance_feedback)
            
            # Split data for validation
            val_split = validation_split or self.training_config['validation_split']
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42
            )
            
            # Fit feature scaler
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            self._is_scaler_fitted = True
            
            # Perform cross-validation if enabled
            cv_scores = None
            if enable_cross_validation:
                cv_scores = await self._perform_cross_validation(X_train_scaled, y_train)
            
            # Hyperparameter optimization if enabled
            best_model = None
            if enable_hyperparameter_optimization:
                best_model = await self._optimize_hyperparameters(X_train_scaled, y_train)
            else:
                best_model = self.ml_model
            
            # Train the best model
            best_model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            y_pred = best_model.predict(X_val_scaled)
            validation_metrics = await self._calculate_validation_metrics(y_val, y_pred)
            
            # Update current model
            self.ml_model = best_model
            self._is_trained = True
            
            # Store training results
            training_result = {
                'success': True,
                'samples_trained': len(X_train),
                'validation_samples': len(X_val),
                'cross_validation_scores': cv_scores,
                'validation_metrics': validation_metrics,
                'model_name': self.current_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_history.append(training_result)
            
            # Auto-save model if enabled
            if self.model_persistence_config['auto_save']:
                await self._save_model()
            
            self.logger.info(f"Successfully trained ML model on {len(X_train)} samples")
            self.logger.info(f"Validation R²: {validation_metrics['r2_score']:.3f}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Failed to train ML model: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_features(self, query: str, chunk: ContextChunk) -> List[float]:
        """Extract comprehensive features for ML training."""
        try:
            # Basic text features
            query_length = len(query)
            content_length = len(chunk.content)
            token_count = chunk.token_count or 0
            
            # Semantic features (if embeddings available)
            semantic_score = 0.0
            if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                try:
                    query_embedding = await self._get_query_embedding(query)
                    if query_embedding is not None:
                        semantic_score = cosine_similarity(
                            [query_embedding], [chunk.embedding]
                        )[0][0]
                except Exception:
                    semantic_score = 0.0
            
            # Content quality features
            content_quality = min(1.0, content_length / 1000.0)  # Normalize by expected length
            query_content_ratio = query_length / max(content_length, 1)
            
            # Metadata features
            source_authority = getattr(chunk.source, 'authority_score', 0.5) if chunk.source else 0.5
            freshness_score = getattr(chunk, 'freshness_score', 0.5) if hasattr(chunk, 'freshness_score') else 0.5
            
            # Complexity features
            avg_word_length = np.mean([len(word) for word in chunk.content.split()]) if chunk.content else 0
            unique_words_ratio = len(set(chunk.content.lower().split())) / max(len(chunk.content.split()), 1)
            
            features = [
                query_length,
                content_length,
                token_count,
                semantic_score,
                content_quality,
                query_content_ratio,
                source_authority,
                freshness_score,
                avg_word_length,
                unique_words_ratio,
                # Interaction features
                query_length * content_length / 1000,
                semantic_score * content_quality,
                source_authority * freshness_score,
                # Normalized features
                query_length / 100.0,  # Normalize query length
                content_length / 1000.0,  # Normalize content length
                token_count / 100.0,  # Normalize token count
                # Additional features to reach 20
                np.log(max(query_length, 1)),  # Log of query length
                np.log(max(content_length, 1)),  # Log of content length
                np.log(max(token_count, 1)),  # Log of token count
                np.sqrt(query_length),  # Square root of query length
            ]
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract features: {e}")
            return [0.5] * 20  # Return neutral features
    
    async def calibrate_confidence_intervals(self, validation_data: List[Dict[str, Any]]) -> None:
        """Calibrate confidence intervals using validation data."""
        try:
            self.logger.info(f"Calibrating confidence intervals with {len(validation_data)} samples")
            
            calibration_errors = []
            
            for sample in validation_data:
                predicted_score = sample.get('predicted_score', 0.5)
                predicted_lower = sample.get('predicted_lower', 0.0)
                predicted_upper = sample.get('predicted_upper', 1.0)
                actual_score = sample.get('actual_score', 0.5)
                
                # Check if actual score falls within predicted interval
                if predicted_lower <= actual_score <= predicted_upper:
                    calibration_errors.append(0.0)  # No error
                else:
                    # Calculate distance to interval
                    if actual_score < predicted_lower:
                        error = predicted_lower - actual_score
                    else:
                        error = actual_score - predicted_upper
                    calibration_errors.append(error)
            
            # Calculate calibration statistics
            coverage_rate = sum(1 for error in calibration_errors if error == 0.0) / len(calibration_errors)
            mean_error = np.mean(calibration_errors)
            
            self.logger.info(f"Confidence interval calibration results:")
            self.logger.info(f"  Coverage rate: {coverage_rate:.3f}")
            self.logger.info(f"  Mean error: {mean_error:.3f}")
            
            # Adjust confidence level if needed
            target_coverage = self.confidence_config['default_confidence_level']
            if abs(coverage_rate - target_coverage) > 0.05:
                # Adjust confidence level to improve calibration
                adjusted_level = target_coverage * (target_coverage / coverage_rate)
                adjusted_level = max(0.8, min(0.99, adjusted_level))
                
                self.confidence_config['default_confidence_level'] = adjusted_level
                self.logger.info(f"Adjusted confidence level to {adjusted_level:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate confidence intervals: {e}")
    
    async def validate_confidence_intervals(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate confidence interval quality using test data."""
        try:
            self.logger.info(f"Validating confidence intervals with {len(test_data)} samples")
            
            validation_metrics = {
                'coverage_rate': 0.0,
                'mean_interval_width': 0.0,
                'calibration_error': 0.0,
                'reliability_score': 0.0,
            }
            
            if not test_data:
                return validation_metrics
            
            coverage_count = 0
            interval_widths = []
            calibration_errors = []
            
            for sample in test_data:
                predicted_score = sample.get('predicted_score', 0.5)
                predicted_lower = sample.get('predicted_lower', 0.0)
                predicted_upper = sample.get('predicted_upper', 1.0)
                actual_score = sample.get('actual_score', 0.5)
                
                # Check coverage
                if predicted_lower <= actual_score <= predicted_upper:
                    coverage_count += 1
                
                # Calculate interval width
                interval_width = predicted_upper - predicted_lower
                interval_widths.append(interval_width)
                
                # Calculate calibration error
                if predicted_lower <= actual_score <= predicted_upper:
                    calibration_errors.append(0.0)
                else:
                    if actual_score < predicted_lower:
                        error = predicted_lower - actual_score
                    else:
                        error = actual_score - predicted_upper
                    calibration_errors.append(error)
            
            # Calculate metrics
            validation_metrics['coverage_rate'] = coverage_count / len(test_data)
            validation_metrics['mean_interval_width'] = np.mean(interval_widths)
            validation_metrics['calibration_error'] = np.mean(calibration_errors)
            
            # Calculate reliability score (combination of coverage and precision)
            target_coverage = self.confidence_config['default_confidence_level']
            coverage_penalty = abs(validation_metrics['coverage_rate'] - target_coverage)
            width_penalty = validation_metrics['mean_interval_width']  # Narrower is better
            
            validation_metrics['reliability_score'] = max(0.0, 1.0 - coverage_penalty - width_penalty)
            
            self.logger.info(f"Confidence interval validation results:")
            self.logger.info(f"  Coverage rate: {validation_metrics['coverage_rate']:.3f}")
            self.logger.info(f"  Mean interval width: {validation_metrics['mean_interval_width']:.3f}")
            self.logger.info(f"  Calibration error: {validation_metrics['calibration_error']:.3f}")
            self.logger.info(f"  Reliability score: {validation_metrics['reliability_score']:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to validate confidence intervals: {e}")
            return {
                'coverage_rate': 0.0,
                'mean_interval_width': 0.0,
                'calibration_error': 0.0,
                'reliability_score': 0.0,
            }
    
    async def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive confidence interval statistics."""
        try:
            if not self.scoring_history:
                return {
                    'total_samples': 0,
                    'mean_confidence_width': 0.0,
                    'confidence_level': self.confidence_config['default_confidence_level'],
                    'calibration_status': 'insufficient_data',
                }
            
            # Calculate statistics from scoring history
            confidence_widths = []
            for record in self.scoring_history:
                if 'confidence_lower' in record and 'confidence_upper' in record:
                    width = record['confidence_upper'] - record['confidence_lower']
                    confidence_widths.append(width)
            
            if not confidence_widths:
                return {
                    'total_samples': len(self.scoring_history),
                    'mean_confidence_width': 0.0,
                    'confidence_level': self.confidence_config['default_confidence_level'],
                    'calibration_status': 'no_confidence_data',
                }
            
            stats = {
                'total_samples': len(self.scoring_history),
                'mean_confidence_width': np.mean(confidence_widths),
                'std_confidence_width': np.std(confidence_widths),
                'min_confidence_width': np.min(confidence_widths),
                'max_confidence_width': np.max(confidence_widths),
                'confidence_level': self.confidence_config['default_confidence_level'],
                'calibration_status': 'calibrated' if len(self.confidence_calibration_data) > 0 else 'uncalibrated',
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get confidence statistics: {e}")
            return {
                'total_samples': 0,
                'mean_confidence_width': 0.0,
                'confidence_level': self.confidence_config['default_confidence_level'],
                'calibration_status': 'error',
            }
    
    # ==================== ML Ensemble Training Methods ====================
    
    async def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation and return scores."""
        try:
            cv_folds = self.training_config['cross_validation_folds']
            
            # Perform cross-validation with multiple metrics
            cv_r2 = cross_val_score(self.ml_model, X, y, cv=cv_folds, scoring='r2')
            cv_mse = cross_val_score(self.ml_model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
            cv_mae = cross_val_score(self.ml_model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
            
            return {
                'r2_scores': cv_r2.tolist(),
                'mse_scores': (-cv_mse).tolist(),  # Convert back to positive
                'mae_scores': (-cv_mae).tolist(),  # Convert back to positive
                'r2_mean': np.mean(cv_r2),
                'r2_std': np.std(cv_r2),
                'mse_mean': np.mean(-cv_mse),
                'mae_mean': np.mean(-cv_mae),
                'cv_folds': cv_folds
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            return {
                'r2_scores': [],
                'mse_scores': [],
                'mae_scores': [],
                'r2_mean': 0.0,
                'r2_std': 0.0,
                'mse_mean': 0.0,
                'mae_mean': 0.0,
                'cv_folds': 0
            }
    
    async def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize hyperparameters using grid search or random search."""
        try:
            method = self.training_config['optimization_method']
            
            if method == 'grid_search':
                best_model = await self._grid_search_optimization(X, y)
            elif method == 'random_search':
                best_model = await self._random_search_optimization(X, y)
            else:
                self.logger.warning(f"Unknown optimization method: {method}")
                return self.ml_model
            
            return best_model
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {e}")
            return self.ml_model
    
    async def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Perform grid search hyperparameter optimization."""
        try:
            # Define parameter grids for different model types
            if isinstance(self.ml_model, RandomForestRegressor):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif isinstance(self.ml_model, GradientBoostingRegressor):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif isinstance(self.ml_model, Ridge):
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            elif isinstance(self.ml_model, SVR):
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
            else:
                # For other models, use basic parameters
                return self.ml_model
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.ml_model,
                param_grid,
                cv=3,  # Use fewer folds for optimization
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.3f}")
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"Grid search optimization failed: {e}")
            return self.ml_model
    
    async def _random_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Perform random search hyperparameter optimization."""
        try:
            # Define parameter distributions for different model types
            if isinstance(self.ml_model, RandomForestRegressor):
                param_distributions = {
                    'n_estimators': [50, 100, 150, 200, 250],
                    'max_depth': [None, 5, 10, 15, 20, 25],
                    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_leaf': [1, 2, 3, 4, 5]
                }
            elif isinstance(self.ml_model, GradientBoostingRegressor):
                param_distributions = {
                    'n_estimators': [50, 75, 100, 125, 150, 175, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                }
            else:
                # For other models, use basic parameters
                return self.ml_model
            
            # Perform random search
            random_search = RandomizedSearchCV(
                self.ml_model,
                param_distributions,
                n_iter=20,  # Number of parameter combinations to try
                cv=3,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            random_search.fit(X, y)
            
            self.logger.info(f"Random search completed. Best score: {random_search.best_score_:.3f}")
            self.logger.info(f"Best parameters: {random_search.best_params_}")
            
            return random_search.best_estimator_
            
        except Exception as e:
            self.logger.warning(f"Random search optimization failed: {e}")
            return self.ml_model
    
    async def _calculate_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Additional metrics
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            explained_variance = np.var(y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'explained_variance': explained_variance,
                'mean_absolute_error': mae,
                'root_mean_squared_error': rmse
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate validation metrics: {e}")
            return {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2_score': 0.0,
                'mape': 0.0,
                'explained_variance': 0.0,
                'mean_absolute_error': 0.0,
                'root_mean_squared_error': 0.0
            }
    
    async def _save_model(self) -> bool:
        """Save the trained ML model and related data."""
        try:
            if not self.model_persistence_config['enabled']:
                return False
            
            # Create model directory if it doesn't exist
            model_dir = Path(self.model_persistence_config['model_dir'])
            model_dir.mkdir(exist_ok=True)
            
            # Save the ML model
            model_path = model_dir / f"ml_model_{self.current_model_name}.joblib"
            joblib.dump(self.ml_model, model_path)
            
            # Save the feature scaler
            scaler_path = model_dir / f"feature_scaler_{self.current_model_name}.joblib"
            joblib.dump(self.feature_scaler, scaler_path)
            
            # Save model metadata
            metadata = {
                'model_name': self.current_model_name,
                'model_version': self.model_persistence_config['model_version'],
                'training_timestamp': datetime.now().isoformat(),
                'is_trained': self._is_trained,
                'is_scaler_fitted': self._is_scaler_fitted,
                'training_history': self.training_history,
                'model_performance': self.model_performance,
                'feature_dimensions': self.ml_model.n_features_in_ if hasattr(self.ml_model, 'n_features_in_') else None
            }
            
            metadata_path = model_dir / f"model_metadata_{self.current_model_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    async def _load_model(self) -> bool:
        """Load a previously saved ML model and related data."""
        try:
            if not self.model_persistence_config['enabled']:
                return False
            
            model_dir = Path(self.model_persistence_config['model_dir'])
            
            # Try to load the model
            model_path = model_dir / f"ml_model_{self.current_model_name}.joblib"
            if not model_path.exists():
                self.logger.info(f"No saved model found at {model_path}")
                return False
            
            # Load the ML model
            self.ml_model = joblib.load(model_path)
            
            # Load the feature scaler
            scaler_path = model_dir / f"feature_scaler_{self.current_model_name}.joblib"
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                self._is_scaler_fitted = True
            
            # Load model metadata
            metadata_path = model_dir / f"model_metadata_{self.current_model_name}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.training_history = metadata.get('training_history', [])
                self.model_performance = metadata.get('model_performance', {})
                self._is_trained = metadata.get('is_trained', False)
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    async def select_best_model(self, validation_data: List[tuple]) -> str:
        """Select the best performing model from available models."""
        try:
            if not validation_data:
                return self.current_model_name
            
            best_model_name = self.current_model_name
            best_score = -float('inf')
            
            for model_name, model in self.ml_models.items():
                # Extract features and labels
                features = []
                labels = []
                
                for query, chunk, score in validation_data:
                    feature_vector = await self._extract_features(query, chunk)
                    features.append(feature_vector)
                    labels.append(score)
                
                if len(features) < 5:  # Need minimum data for validation
                    continue
                
                # Scale features
                X = np.array(features)
                if self._is_scaler_fitted:
                    X = self.feature_scaler.transform(X)
                else:
                    X = self.feature_scaler.fit_transform(X)
                    self._is_scaler_fitted = True
                
                y = np.array(labels)
                
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                avg_score = np.mean(cv_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = model_name
            
            # Update current model if a better one is found
            if best_model_name != self.current_model_name:
                self.current_model_name = best_model_name
                self.ml_model = self.ml_models[best_model_name]
                self.logger.info(f"Selected {best_model_name} as best model (CV score: {best_score:.3f})")
            
            return best_model_name
            
        except Exception as e:
            self.logger.error(f"Failed to select best model: {e}")
            return self.current_model_name
    
    async def retrain_model(self, new_data: List[tuple], retrain_frequency: int = 100) -> bool:
        """Retrain the model with new data if enough samples are available."""
        try:
            # Check if we have enough new data
            if len(new_data) < retrain_frequency:
                return False
            
            # Extract query-chunk pairs and scores
            query_chunk_pairs = [(query, chunk) for query, chunk, score in new_data]
            relevance_scores = [score for query, chunk, score in new_data]
            
            # Retrain the model
            training_result = await self.train_on_feedback(
                query_chunk_pairs,
                relevance_scores,
                enable_cross_validation=True,
                enable_hyperparameter_optimization=True
            )
            
            if training_result.get('success', False):
                self.logger.info(f"Model retrained successfully on {len(new_data)} new samples")
                return True
            else:
                self.logger.warning(f"Model retraining failed: {training_result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to retrain model: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current ML model."""
        try:
            info = {
                'model_name': self.current_model_name,
                'is_trained': self._is_trained,
                'is_scaler_fitted': self._is_scaler_fitted,
                'model_type': type(self.ml_model).__name__,
                'feature_dimensions': getattr(self.ml_model, 'n_features_in_', None),
                'training_samples': len(self.training_history),
                'last_training': self.training_history[-1]['timestamp'] if self.training_history else None,
                'model_persistence': {
                    'enabled': self.model_persistence_config['enabled'],
                    'model_dir': self.model_persistence_config['model_dir'],
                    'auto_save': self.model_persistence_config['auto_save']
                },
                'training_config': self.training_config,
                'available_models': list(self.ml_models.keys())
            }
            
            # Add performance metrics if available
            if self.training_history:
                latest_training = self.training_history[-1]
                if latest_training.get('success', False):
                    info['latest_performance'] = latest_training.get('validation_metrics', {})
                    info['cross_validation_scores'] = latest_training.get('cross_validation_scores', {})
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    async def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for a query string."""
        try:
            if not self.embedding_model:
                return None
            
            # Get embedding from the model
            embedding = self.embedding_model.encode([query])
            return embedding[0] if embedding is not None else None
            
        except Exception as e:
            self.logger.warning(f"Failed to get query embedding: {e}")
            return None
    
    async def _percentile_bootstrap(
        self,
        score_values: List[float],
        ensemble_score: float,
        confidence_level: float = None
    ) -> Optional[tuple[float, float]]:
        """Calculate percentile bootstrap confidence interval."""
        try:
            if confidence_level is None:
                confidence_level = self.confidence_config['default_confidence_level']
            
            n = len(score_values)
            if n < self.confidence_config['min_sample_size']:
                return None
            
            # Generate bootstrap samples
            bootstrap_samples = []
            n_bootstrap = min(self.confidence_config['bootstrap_samples'], 
                            self.confidence_config['max_bootstrap_samples'])
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(score_values, size=n, replace=True)
                bootstrap_samples.append(np.mean(bootstrap_sample))
            
            # Calculate confidence interval using percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)
            
            # Adjust for ensemble score
            center_offset = ensemble_score - np.mean(score_values)
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate percentile bootstrap CI: {e}")
            return None

    async def _bca_bootstrap(
        self,
        score_values: List[float],
        ensemble_score: float,
        confidence_level: float = None
    ) -> Optional[tuple[float, float]]:
        """Calculate BCa (Bias-corrected and accelerated) bootstrap confidence interval."""
        try:
            if confidence_level is None:
                confidence_level = self.confidence_config['default_confidence_level']
            
            n = len(score_values)
            if n < self.confidence_config['min_sample_size']:
                return None
            
            # Generate bootstrap samples
            bootstrap_samples = []
            n_bootstrap = min(self.confidence_config['bootstrap_samples'], 
                            self.confidence_config['max_bootstrap_samples'])
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(score_values, size=n, replace=True)
                bootstrap_samples.append(np.mean(bootstrap_sample))
            
            # Calculate bias correction
            original_mean = np.mean(score_values)
            bootstrap_means = np.array(bootstrap_samples)
            bias_correction = np.sum(bootstrap_means < original_mean) / len(bootstrap_means)
            
            # Calculate acceleration factor using jackknife
            jackknife_means = []
            for i in range(n):
                jackknife_sample = np.delete(score_values, i)
                jackknife_means.append(np.mean(jackknife_sample))
            
            jackknife_means = np.array(jackknife_means)
            jackknife_mean = np.mean(jackknife_means)
            acceleration = np.sum((jackknife_mean - jackknife_means) ** 3) / (6 * np.sum((jackknife_mean - jackknife_means) ** 2) ** 1.5)
            
            # Calculate BCa confidence interval
            alpha = 1 - confidence_level
            z_alpha_2 = norm.ppf(alpha / 2)
            z_1_alpha_2 = norm.ppf(1 - alpha / 2)
            
            # Transform to BCa scale
            z_lower = bias_correction + (bias_correction + z_alpha_2) / (1 - acceleration * (bias_correction + z_alpha_2))
            z_upper = bias_correction + (bias_correction + z_1_alpha_2) / (1 - acceleration * (bias_correction + z_1_alpha_2))
            
            # Convert back to percentiles
            lower_percentile = norm.cdf(z_lower) * 100
            upper_percentile = norm.cdf(z_upper) * 100
            
            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)
            
            # Adjust for ensemble score
            center_offset = ensemble_score - original_mean
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate BCa bootstrap CI: {e}")
            return None

    async def _abc_bootstrap(
        self,
        score_values: List[float],
        ensemble_score: float,
        confidence_level: float = None
    ) -> Optional[tuple[float, float]]:
        """Calculate ABC (Approximate Bootstrap Confidence) interval."""
        try:
            if confidence_level is None:
                confidence_level = self.confidence_config['default_confidence_level']
            
            n = len(score_values)
            if n < self.confidence_config['min_sample_size']:
                return None
            
            # Calculate influence values using infinitesimal jackknife
            original_mean = np.mean(score_values)
            influence_values = []
            
            for i in range(n):
                # Calculate influence of each observation
                influence = (n * original_mean - score_values[i]) / (n - 1) - original_mean
                influence_values.append(influence)
            
            influence_values = np.array(influence_values)
            
            # Calculate ABC parameters
            a = np.sum(influence_values ** 3) / (6 * np.sum(influence_values ** 2) ** 1.5)
            b = np.sum(influence_values ** 2) / np.sum(influence_values ** 2)
            
            # Generate bootstrap samples for variance estimation
            bootstrap_variances = []
            n_bootstrap_abc = min(100, self.confidence_config['bootstrap_samples'])  # Use fewer samples for ABC
            
            for _ in range(n_bootstrap_abc):
                bootstrap_sample = np.random.choice(score_values, size=n, replace=True)
                bootstrap_variances.append(np.var(bootstrap_sample))
            
            bootstrap_var = np.mean(bootstrap_variances)
            
            # Calculate ABC confidence interval
            alpha = 1 - confidence_level
            z_alpha_2 = norm.ppf(alpha / 2)
            z_1_alpha_2 = norm.ppf(1 - alpha / 2)
            
            # Transform to ABC scale
            z_lower = z_alpha_2 + a * (z_alpha_2 ** 2 - 1) / 6
            z_upper = z_1_alpha_2 + a * (z_1_alpha_2 ** 2 - 1) / 6
            
            # Calculate bounds
            margin_lower = z_lower * np.sqrt(bootstrap_var / n)
            margin_upper = z_upper * np.sqrt(bootstrap_var / n)
            
            lower_bound = original_mean - margin_lower
            upper_bound = original_mean + margin_upper
            
            # Adjust for ensemble score
            center_offset = ensemble_score - original_mean
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate ABC bootstrap CI: {e}")
            return None

    async def _studentized_bootstrap(
        self,
        score_values: List[float],
        ensemble_score: float,
        confidence_level: float = None
    ) -> Optional[tuple[float, float]]:
        """Calculate Studentized bootstrap confidence interval."""
        try:
            if confidence_level is None:
                confidence_level = self.confidence_config['default_confidence_level']
            
            n = len(score_values)
            if n < self.confidence_config['min_sample_size']:
                return None
            
            # Calculate original statistics
            original_mean = np.mean(score_values)
            original_std = np.std(score_values, ddof=1)
            original_se = original_std / np.sqrt(n)
            
            # Generate bootstrap samples
            bootstrap_means = []
            bootstrap_stds = []
            n_bootstrap = min(self.confidence_config['bootstrap_samples'], 
                            self.confidence_config['max_bootstrap_samples'])
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(score_values, size=n, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
                bootstrap_stds.append(np.std(bootstrap_sample, ddof=1))
            
            bootstrap_means = np.array(bootstrap_means)
            bootstrap_stds = np.array(bootstrap_stds)
            
            # Calculate studentized statistics
            studentized_stats = []
            for i in range(n_bootstrap):
                if bootstrap_stds[i] > 0:
                    # Studentized statistic: (bootstrap_mean - original_mean) / bootstrap_se
                    bootstrap_se = bootstrap_stds[i] / np.sqrt(n)
                    studentized_stat = (bootstrap_means[i] - original_mean) / bootstrap_se
                    studentized_stats.append(studentized_stat)
            
            if len(studentized_stats) < 10:  # Need sufficient valid statistics
                return None
            
            studentized_stats = np.array(studentized_stats)
            
            # Calculate confidence interval using studentized percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            t_lower = np.percentile(studentized_stats, lower_percentile)
            t_upper = np.percentile(studentized_stats, upper_percentile)
            
            # Calculate bounds using original standard error
            lower_bound = original_mean - t_upper * original_se
            upper_bound = original_mean - t_lower * original_se
            
            # Adjust for ensemble score
            center_offset = ensemble_score - original_mean
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate studentized bootstrap CI: {e}")
            return None
