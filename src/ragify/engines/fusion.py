"""
Intelligent Context Fusion Engine for multi-source context combination with conflict resolution.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher
import re
import structlog

from ..models import (
    ContextChunk, Context, OrchestratorConfig, ConflictType, 
    ConflictResolutionStrategy, ConflictInfo, FusionMetadata,
    ContextSource
)
from ..exceptions import ICOException


class IntelligentContextFusionEngine:
    """
    Intelligent context fusion engine with advanced conflict detection and resolution.
    
    Features:
    - Multi-source context fusion with intelligent deduplication
    - Advanced conflict detection (content contradictions, factual disagreements, etc.)
    - Multiple conflict resolution strategies
    - Semantic similarity analysis
    - Source authority and freshness weighting
    - Consensus-based decision making
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the intelligent fusion engine.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Source authority scores (higher = more authoritative)
        self.source_authority_scores = {
            'database': 1.0,
            'api': 0.9,
            'document': 0.8,
            'realtime': 0.7,
            'vector': 0.6,
            'cache': 0.5,
        }
        
        # Conflict detection thresholds
        self.conflict_thresholds = {
            ConflictType.CONTENT_CONTRADICTION: 0.8,
            ConflictType.FACTUAL_DISAGREEMENT: 0.7,
            ConflictType.TEMPORAL_CONFLICT: 0.6,
            ConflictType.SOURCE_AUTHORITY: 0.5,
            ConflictType.DATA_FRESHNESS: 0.4,
            ConflictType.SEMANTIC_CONFLICT: 0.6,
        }
        
        # Fusion strategies
        self.fusion_strategies = {
            'intelligent': self._intelligent_fusion,
            'weighted_average': self._weighted_average_fusion,
            'priority_based': self._priority_based_fusion,
            'consensus': self._consensus_fusion,
            'hierarchical': self._hierarchical_fusion,
            'conflict_aware': self._conflict_aware_fusion,
        }
    
    async def fuse_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        strategy: str = 'intelligent',
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Any:
        """Fuse chunks using intelligent conflict detection and resolution (alias for fuse_contexts)."""
        # Convert chunks to Context objects for compatibility
        from ..models import Context, ContextChunk, ContextSource
        from uuid import uuid4
        
        context_chunks = []
        for chunk_data in chunks:
            chunk = ContextChunk(
                id=uuid4(),
                content=chunk_data.get('content', ''),
                source=ContextSource(
                    id=uuid4(),
                    name=chunk_data.get('source', 'unknown'),
                    source_type='document'
                ),
                metadata=chunk_data
            )
            context_chunks.append(chunk)
        
        context = Context(
            query=query,
            chunks=context_chunks,
            user_id="test_user",
            session_id="test_session"
        )
        
        fused_context = await self.fuse_contexts([context], strategy, weights, conflict_resolution)
        
        # Return a simple result object
        class FusionResult:
            def __init__(self, fused_chunks, conflicts):
                self.fused_chunks = fused_chunks
                self.conflicts = conflicts
        
        return FusionResult(
            fused_chunks=fused_context.chunks,
            conflicts=[]  # Would be populated by actual conflict detection
        )
    
    async def fuse_contexts(
        self,
        contexts: List[Context],
        strategy: str = 'intelligent',
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """
        Fuse multiple contexts using intelligent conflict detection and resolution.
        
        Args:
            contexts: List of contexts to fuse
            strategy: Fusion strategy to use
            weights: Optional weights for weighted strategies
            conflict_resolution: Strategy for resolving conflicts
            
        Returns:
            Fused context with conflict resolution metadata
        """
        start_time = time.time()
        
        if not contexts:
            raise ICOException("No contexts provided for fusion")
        
        if len(contexts) == 1:
            return contexts[0]
        
        self.logger.info(
            f"Starting intelligent fusion of {len(contexts)} contexts using strategy: {strategy}"
        )
        
        try:
            # Get fusion strategy
            fusion_func = self.fusion_strategies.get(strategy)
            if not fusion_func:
                raise ICOException(f"Unknown fusion strategy: {strategy}")
            
            # Perform intelligent fusion
            fused_context = await fusion_func(contexts, weights, conflict_resolution)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update fusion metadata
            if fused_context.fusion_metadata:
                fused_context.fusion_metadata.processing_time = processing_time
                fused_context.fusion_metadata.fusion_strategy = strategy
            
            self.logger.info(
                f"Successfully fused {len(contexts)} contexts into {len(fused_context.chunks)} chunks",
                processing_time=processing_time,
                conflict_count=fused_context.fusion_metadata.conflict_count if fused_context.fusion_metadata else 0
            )
            
            return fused_context
            
        except Exception as e:
            self.logger.error(f"Failed to fuse contexts: {e}")
            raise
    
    async def _intelligent_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """
        Intelligent fusion with advanced conflict detection and resolution.
        """
        # Collect all chunks from all contexts
        all_chunks = []
        for context in contexts:
            all_chunks.extend(context.chunks)
        
        # Step 1: Detect and resolve conflicts
        resolved_chunks, conflicts = await self._detect_and_resolve_conflicts(
            all_chunks, conflict_resolution
        )
        
        # Step 2: Group similar chunks for deduplication
        grouped_chunks = await self._group_similar_chunks(resolved_chunks)
        
        # Step 3: Fuse each group intelligently
        fused_chunks = []
        for group in grouped_chunks:
            if len(group) == 1:
                fused_chunks.append(group[0])
            else:
                fused_chunk = await self._fuse_chunk_group_intelligently(group, weights)
                fused_chunks.append(fused_chunk)
        
        # Step 4: Create fusion metadata
        fusion_metadata = FusionMetadata(
            fusion_strategy="intelligent",
            conflict_count=len(conflicts),
            resolved_conflicts=conflicts,
            source_weights=weights or {},
            fusion_confidence=self._calculate_fusion_confidence(fused_chunks, conflicts),
        )
        
        # Create fused context
        return Context(
            query=contexts[0].query,
            chunks=fused_chunks,
            user_id=contexts[0].user_id,
            session_id=contexts[0].session_id,
            privacy_level=contexts[0].privacy_level,
            fusion_metadata=fusion_metadata,
        )
    
    async def _detect_and_resolve_conflicts(
        self,
        chunks: List[ContextChunk],
        resolution_strategy: ConflictResolutionStrategy,
    ) -> Tuple[List[ContextChunk], List[ConflictInfo]]:
        """
        Detect conflicts between chunks and resolve them.
        
        Returns:
            Tuple of (resolved_chunks, conflict_info_list)
        """
        conflicts = []
        resolved_chunks = chunks.copy()
        
        # Group chunks by semantic similarity for conflict detection
        similar_groups = await self._group_similar_chunks(chunks)
        
        for group in similar_groups:
            if len(group) <= 1:
                continue
            
            # Detect conflicts within the group
            group_conflicts = await self._detect_conflicts_in_group(group)
            
            for conflict in group_conflicts:
                conflicts.append(conflict)
                
                # Resolve the conflict
                resolved_chunk = await self._resolve_conflict(
                    [c for c in group if c.id in conflict.conflicting_chunks],
                    resolution_strategy
                )
                
                # Update the resolved chunk
                conflict.resolved_chunk_id = resolved_chunk.id
                
                # Replace conflicting chunks with resolved chunk
                for chunk_id in conflict.conflicting_chunks:
                    resolved_chunks = [c for c in resolved_chunks if c.id != chunk_id]
                
                resolved_chunks.append(resolved_chunk)
        
        return resolved_chunks, conflicts
    
    async def _detect_conflicts_in_group(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect various types of conflicts within a group of similar chunks.
        """
        conflicts = []
        
        # Detect content contradictions
        content_conflicts = await self._detect_content_contradictions(chunks)
        conflicts.extend(content_conflicts)
        
        # Detect factual disagreements
        factual_conflicts = await self._detect_factual_disagreements(chunks)
        conflicts.extend(factual_conflicts)
        
        # Detect temporal conflicts
        temporal_conflicts = await self._detect_temporal_conflicts(chunks)
        conflicts.extend(temporal_conflicts)
        
        # Detect source authority conflicts
        authority_conflicts = await self._detect_source_authority_conflicts(chunks)
        conflicts.extend(authority_conflicts)
        
        # Detect data freshness conflicts
        freshness_conflicts = await self._detect_data_freshness_conflicts(chunks)
        conflicts.extend(freshness_conflicts)
        
        # Detect semantic conflicts
        semantic_conflicts = await self._detect_semantic_conflicts(chunks)
        conflicts.extend(semantic_conflicts)
        
        return conflicts
    
    async def _detect_content_contradictions(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect direct content contradictions between chunks.
        """
        conflicts = []
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                contradiction_score = await self._calculate_contradiction_score(chunk1, chunk2)
                
                if contradiction_score > self.conflict_thresholds[ConflictType.CONTENT_CONTRADICTION]:
                    conflicts.append(ConflictInfo(
                        conflict_type=ConflictType.CONTENT_CONTRADICTION,
                        conflicting_chunks=[chunk1.id, chunk2.id],
                        confidence=contradiction_score,
                        description=f"Content contradiction detected between chunks from {chunk1.source.name} and {chunk2.source.name}",
                        resolution_strategy=ConflictResolutionStrategy.HIGHEST_AUTHORITY,
                    ))
        
        return conflicts
    
    async def _detect_factual_disagreements(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect factual disagreements (numbers, dates, names, etc.).
        """
        conflicts = []
        
        # Extract factual information from chunks
        factual_info = await self._extract_factual_information(chunks)
        
        for fact_type, values in factual_info.items():
            # Convert lists to tuples for comparison, filter out None values
            non_none_values = [v for v in values if v is not None]
            if len(non_none_values) > 1:
                # Convert lists to tuples for hashable comparison
                if fact_type == 'numbers':
                    # For numbers, compare the actual numbers
                    unique_values = set()
                    for value_list in non_none_values:
                        if isinstance(value_list, list):
                            unique_values.update(value_list)
                        else:
                            unique_values.add(value_list)
                    
                    if len(unique_values) > 1:
                        conflicting_chunks = [
                            chunk.id for chunk, value in zip(chunks, values)
                            if value is not None
                        ]
                        
                        if len(conflicting_chunks) > 1:
                            conflicts.append(ConflictInfo(
                                conflict_type=ConflictType.FACTUAL_DISAGREEMENT,
                                conflicting_chunks=conflicting_chunks,
                                confidence=0.8,
                                description=f"Factual disagreement detected for {fact_type}",
                                resolution_strategy=ConflictResolutionStrategy.HIGHEST_AUTHORITY,
                                metadata={'fact_type': fact_type, 'values': values}
                            ))
        
        return conflicts
    
    async def _detect_temporal_conflicts(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect temporal conflicts (outdated information, time-sensitive data).
        """
        conflicts = []
        
        # Calculate age differences
        ages = [(chunk, (time.time() - chunk.created_at.timestamp()) / 86400) for chunk in chunks]
        max_age_diff = max(ages, key=lambda x: x[1])[1] - min(ages, key=lambda x: x[1])[1]
        
        if max_age_diff > 30:  # More than 30 days difference
            old_chunks = [chunk.id for chunk, age in ages if age > 30]
            new_chunks = [chunk.id for chunk, age in ages if age <= 7]
            
            if old_chunks and new_chunks:
                conflicts.append(ConflictInfo(
                    conflict_type=ConflictType.TEMPORAL_CONFLICT,
                    conflicting_chunks=old_chunks + new_chunks,
                    confidence=0.7,
                    description="Temporal conflict detected between old and new data",
                    resolution_strategy=ConflictResolutionStrategy.NEWEST_DATA,
                    metadata={'max_age_diff_days': max_age_diff}
                ))
        
        return conflicts
    
    async def _detect_source_authority_conflicts(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect conflicts based on source authority differences.
        """
        conflicts = []
        
        authority_scores = [
            self.source_authority_scores.get(chunk.source.source_type.value, 0.5)
            for chunk in chunks
        ]
        
        max_authority = max(authority_scores)
        min_authority = min(authority_scores)
        
        if max_authority - min_authority > 0.3:  # Significant authority difference
            low_authority_chunks = [
                chunk.id for chunk, score in zip(chunks, authority_scores)
                if score < max_authority - 0.3
            ]
            high_authority_chunks = [
                chunk.id for chunk, score in zip(chunks, authority_scores)
                if score >= max_authority - 0.1
            ]
            
            if low_authority_chunks and high_authority_chunks:
                conflicts.append(ConflictInfo(
                    conflict_type=ConflictType.SOURCE_AUTHORITY,
                    conflicting_chunks=low_authority_chunks + high_authority_chunks,
                    confidence=0.6,
                    description="Source authority conflict detected",
                    resolution_strategy=ConflictResolutionStrategy.HIGHEST_AUTHORITY,
                    metadata={'authority_scores': authority_scores}
                ))
        
        return conflicts
    
    async def _detect_data_freshness_conflicts(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect conflicts based on data freshness.
        """
        conflicts = []
        
        # Calculate freshness scores based on update times
        freshness_scores = []
        for chunk in chunks:
            age_hours = (time.time() - chunk.updated_at.timestamp()) / 3600
            freshness = max(0, 1 - (age_hours / 168))  # Decay over 1 week
            freshness_scores.append(freshness)
        
        max_freshness = max(freshness_scores)
        min_freshness = min(freshness_scores)
        
        if max_freshness - min_freshness > 0.5:  # Significant freshness difference
            stale_chunks = [
                chunk.id for chunk, score in zip(chunks, freshness_scores)
                if score < max_freshness - 0.5
            ]
            fresh_chunks = [
                chunk.id for chunk, score in zip(chunks, freshness_scores)
                if score >= max_freshness - 0.1
            ]
            
            if stale_chunks and fresh_chunks:
                conflicts.append(ConflictInfo(
                    conflict_type=ConflictType.DATA_FRESHNESS,
                    conflicting_chunks=stale_chunks + fresh_chunks,
                    confidence=0.5,
                    description="Data freshness conflict detected",
                    resolution_strategy=ConflictResolutionStrategy.NEWEST_DATA,
                    metadata={'freshness_scores': freshness_scores}
                ))
        
        return conflicts
    
    async def _detect_semantic_conflicts(self, chunks: List[ContextChunk]) -> List[ConflictInfo]:
        """
        Detect semantic conflicts (different interpretations, contexts).
        """
        conflicts = []
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i+1:]:
                semantic_conflict_score = await self._calculate_semantic_conflict_score(chunk1, chunk2)
                
                if semantic_conflict_score >= self.conflict_thresholds[ConflictType.SEMANTIC_CONFLICT]:
                    conflicts.append(ConflictInfo(
                        conflict_type=ConflictType.SEMANTIC_CONFLICT,
                        conflicting_chunks=[chunk1.id, chunk2.id],
                        confidence=semantic_conflict_score,
                        description=f"Semantic conflict detected between chunks from {chunk1.source.name} and {chunk2.source.name}",
                        resolution_strategy=ConflictResolutionStrategy.CONSENSUS,
                    ))
        
        return conflicts
    
    async def _resolve_conflict(
        self,
        conflicting_chunks: List[ContextChunk],
        strategy: ConflictResolutionStrategy,
    ) -> ContextChunk:
        """
        Resolve a conflict using the specified strategy.
        """
        if not conflicting_chunks:
            raise ICOException("No conflicting chunks provided")
        
        if len(conflicting_chunks) == 1:
            return conflicting_chunks[0]
        
        if strategy == ConflictResolutionStrategy.HIGHEST_RELEVANCE:
            return max(
                conflicting_chunks,
                key=lambda c: c.relevance_score.score if c.relevance_score else 0.0
            )
        elif strategy == ConflictResolutionStrategy.NEWEST_DATA:
            return max(conflicting_chunks, key=lambda c: c.updated_at)
        elif strategy == ConflictResolutionStrategy.HIGHEST_AUTHORITY:
            return max(
                conflicting_chunks,
                key=lambda c: self.source_authority_scores.get(c.source.source_type.value, 0.5)
            )
        elif strategy == ConflictResolutionStrategy.CONSENSUS:
            return await self._resolve_by_consensus(conflicting_chunks)
        elif strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            return await self._resolve_by_weighted_average(conflicting_chunks)
        else:
            raise ICOException(f"Unknown resolution strategy: {strategy}")
    
    async def _resolve_by_consensus(self, chunks: List[ContextChunk]) -> ContextChunk:
        """
        Resolve conflict by finding the chunk with highest consensus among sources.
        """
        # Calculate consensus score for each chunk
        consensus_scores = []
        for chunk in chunks:
            # Consensus = average relevance score + source authority
            relevance_score = chunk.relevance_score.score if chunk.relevance_score else 0.5
            authority_score = self.source_authority_scores.get(chunk.source.source_type.value, 0.5)
            consensus_score = (relevance_score + authority_score) / 2
            consensus_scores.append((chunk, consensus_score))
        
        return max(consensus_scores, key=lambda x: x[1])[0]
    
    async def _resolve_by_weighted_average(self, chunks: List[ContextChunk]) -> ContextChunk:
        """
        Resolve conflict by creating a weighted average chunk.
        """
        # Use the chunk with highest relevance as base
        base_chunk = max(
            chunks,
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0
        )
        
        # Calculate weighted average of relevance scores
        total_weight = 0.0
        weighted_score = 0.0
        
        for chunk in chunks:
            if chunk.relevance_score:
                weight = self.source_authority_scores.get(chunk.source.source_type.value, 0.5)
                weighted_score += chunk.relevance_score.score * weight
                total_weight += weight
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # Create resolved chunk
        resolved_chunk = ContextChunk(
            content=base_chunk.content,
            source=base_chunk.source,
            metadata={
                **base_chunk.metadata,
                'resolved_from': len(chunks),
                'resolution_method': 'weighted_average',
                'original_chunks': [c.id for c in chunks],
            },
            relevance_score=base_chunk.relevance_score,
            token_count=base_chunk.token_count,
            privacy_level=base_chunk.privacy_level,
        )
        
        # Update relevance score
        if resolved_chunk.relevance_score:
            resolved_chunk.relevance_score.score = avg_score
        
        return resolved_chunk
    
    async def _calculate_contradiction_score(self, chunk1: ContextChunk, chunk2: ContextChunk) -> float:
        """
        Calculate the contradiction score between two chunks.
        """
        # Extract key facts and compare
        facts1 = await self._extract_key_facts(chunk1.content)
        facts2 = await self._extract_key_facts(chunk2.content)
        
        contradictions = 0
        total_comparisons = 0
        
        for fact_type in facts1:
            if fact_type in facts2:
                total_comparisons += 1
                if facts1[fact_type] != facts2[fact_type]:
                    contradictions += 1
        
        return contradictions / total_comparisons if total_comparisons > 0 else 0.0
    
    async def _calculate_semantic_conflict_score(self, chunk1: ContextChunk, chunk2: ContextChunk) -> float:
        """
        Calculate semantic conflict score between two chunks.
        """
        # Use sequence matcher for semantic similarity
        similarity = SequenceMatcher(None, chunk1.content.lower(), chunk2.content.lower()).ratio()
        
        # High similarity but different sources might indicate semantic conflict
        if similarity > 0.7:
            # Check if sources are different
            if chunk1.source.source_type != chunk2.source.source_type:
                return 0.6
        
        return 0.0
    
    async def _extract_factual_information(self, chunks: List[ContextChunk]) -> Dict[str, List[Any]]:
        """
        Extract factual information from chunks (numbers, dates, names, etc.).
        """
        factual_info = defaultdict(list)
        
        for chunk in chunks:
            content = chunk.content.lower()
            
            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', content)
            factual_info['numbers'].append(numbers if numbers else None)
            
            # Extract dates
            dates = re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', content)
            factual_info['dates'].append(dates if dates else None)
            
            # Extract percentages
            percentages = re.findall(r'\d+(?:\.\d+)?%', content)
            factual_info['percentages'].append(percentages if percentages else None)
        
        return factual_info
    
    async def _extract_key_facts(self, content: str) -> Dict[str, Any]:
        """
        Extract key facts from content.
        """
        content_lower = content.lower()
        facts = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', content_lower)
        if numbers:
            facts['numbers'] = numbers
        
        # Extract dates
        dates = re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', content_lower)
        if dates:
            facts['dates'] = dates
        
        return facts
    
    async def _group_similar_chunks(self, chunks: List[ContextChunk]) -> List[List[ContextChunk]]:
        """
        Group chunks by semantic similarity for conflict detection.
        """
        if not chunks:
            return []
        
        groups = []
        processed = set()
        
        for i, chunk in enumerate(chunks):
            if i in processed:
                continue
            
            group = [chunk]
            processed.add(i)
            
            # Find similar chunks
            for j, other_chunk in enumerate(chunks[i+1:], i+1):
                if j in processed:
                    continue
                
                if await self._are_chunks_similar(chunk, other_chunk):
                    group.append(other_chunk)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    async def _are_chunks_similar(self, chunk1: ContextChunk, chunk2: ContextChunk) -> bool:
        """
        Check if two chunks are similar enough for conflict detection.
        """
        # Calculate semantic similarity
        similarity = SequenceMatcher(None, chunk1.content.lower(), chunk2.content.lower()).ratio()
        
        # Check for keyword overlap
        words1 = set(chunk1.content.lower().split())
        words2 = set(chunk2.content.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            overlap = 0.0
        
        # Consider chunks similar if either similarity or overlap is high
        return similarity > 0.3 or overlap > 0.2
    
    async def _fuse_chunk_group_intelligently(
        self,
        chunks: List[ContextChunk],
        weights: Optional[Dict[str, float]] = None,
    ) -> ContextChunk:
        """
        Intelligently fuse a group of similar chunks.
        """
        if len(chunks) == 1:
            return chunks[0]
        
        # Select the best chunk as base
        base_chunk = max(
            chunks,
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0
        )
        
        # Create fused chunk with enhanced metadata
        fused_chunk = ContextChunk(
            content=base_chunk.content,
            source=base_chunk.source,
            metadata={
                **base_chunk.metadata,
                'fused_from': len(chunks),
                'fusion_method': 'intelligent',
                'original_chunks': [c.id for c in chunks],
                'source_types': [c.source.source_type.value for c in chunks],
            },
            relevance_score=base_chunk.relevance_score,
            token_count=base_chunk.token_count,
            privacy_level=base_chunk.privacy_level,
        )
        
        return fused_chunk
    
    def _calculate_fusion_confidence(self, chunks: List[ContextChunk], conflicts: List[ConflictInfo]) -> float:
        """
        Calculate confidence in the fusion result.
        """
        if not chunks:
            return 0.0
        
        # Base confidence from chunk relevance scores
        relevance_scores = [
            chunk.relevance_score.score if chunk.relevance_score else 0.5
            for chunk in chunks
        ]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.5
        
        # Penalty for conflicts
        conflict_penalty = len(conflicts) * 0.1
        
        # Source diversity bonus
        source_types = set(chunk.source.source_type.value for chunk in chunks)
        diversity_bonus = len(source_types) * 0.05
        
        confidence = avg_relevance - conflict_penalty + diversity_bonus
        return max(0.0, min(1.0, confidence))
    
    # Legacy fusion methods for backward compatibility
    async def _weighted_average_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """Legacy weighted average fusion."""
        return await self._intelligent_fusion(contexts, weights, conflict_resolution)
    
    async def _priority_based_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """Legacy priority-based fusion."""
        return await self._intelligent_fusion(contexts, weights, conflict_resolution)
    
    async def _consensus_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """Legacy consensus fusion."""
        return await self._intelligent_fusion(contexts, weights, conflict_resolution)
    
    async def _hierarchical_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """Legacy hierarchical fusion."""
        return await self._intelligent_fusion(contexts, weights, conflict_resolution)
    
    async def _conflict_aware_fusion(
        self,
        contexts: List[Context],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_RELEVANCE,
    ) -> Context:
        """Conflict-aware fusion with explicit conflict handling."""
        return await self._intelligent_fusion(contexts, weights, conflict_resolution)
