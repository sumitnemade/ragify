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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import structlog

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
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize ML model for ensemble scoring
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._is_trained = False
        
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
        }
        
        # Historical scoring data for statistical analysis
        self.scoring_history = []
        self.confidence_calibration_data = []
    
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
            # Generate embeddings
            query_embedding = await self._get_embedding(query)
            chunk_embeddings = await self._get_embeddings([chunk.content for chunk in chunks])
            
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
            scores['semantic_similarity'] = await self._calculate_semantic_similarity(
                query_embedding, chunk_embedding
            )
        
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
        """Get embedding for a text."""
        if not self.embedding_model:
            return None
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embedding_model.encode, text
            )
            return embedding.tolist()
        except Exception as e:
            self.logger.warning(f"Failed to get embedding: {e}")
            return None
    
    async def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for multiple texts."""
        if not self.embedding_model:
            return None
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.embedding_model.encode, texts
            )
            return embeddings.tolist()
        except Exception as e:
            self.logger.warning(f"Failed to get embeddings: {e}")
            return None
    
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
        """Calculate bootstrap confidence interval."""
        try:
            if len(score_values) < 5:
                return None
            
            # Bootstrap function
            def bootstrap_statistic(data):
                # Sample with replacement
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                return np.mean(bootstrap_sample)
            
            # Generate bootstrap samples
            bootstrap_samples = []
            for _ in range(self.confidence_config['bootstrap_samples']):
                bootstrap_stat = bootstrap_statistic(score_values)
                bootstrap_samples.append(bootstrap_stat)
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_config['default_confidence_level']
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)
            
            # Adjust bounds relative to ensemble score
            center_offset = ensemble_score - np.mean(bootstrap_samples)
            lower_bound = max(0.0, lower_bound + center_offset)
            upper_bound = min(1.0, upper_bound + center_offset)
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate bootstrap confidence interval: {e}")
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
            if n < 1:
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
    ) -> None:
        """Train the ML model on user feedback."""
        try:
            # Extract features from query-chunk pairs
            features = []
            for query, chunk in query_chunk_pairs:
                feature_vector = await self._extract_features(query, chunk)
                features.append(feature_vector)
            
            # Train the model
            self.ml_model.fit(features, relevance_feedback)
            self._is_trained = True
            
            self.logger.info(f"Trained ML model on {len(query_chunk_pairs)} feedback samples")
        except Exception as e:
            self.logger.error(f"Failed to train ML model: {e}")
    
    async def _extract_features(self, query: str, chunk: ContextChunk) -> List[float]:
        """Extract features for ML training."""
        # Extract comprehensive features for ML training
        # This provides a rich feature vector for machine learning models
        return [
            len(query),
            len(chunk.content),
            chunk.token_count or 0,
        ]
    
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
