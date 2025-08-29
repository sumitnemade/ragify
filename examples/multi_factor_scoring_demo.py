#!/usr/bin/env python3
"""
Multi-Factor Scoring with Ensemble Methods Demo for Ragify Framework

This demo showcases advanced multi-factor scoring with ensemble methods
for enhanced relevance assessment and ranking.
"""

import asyncio
import numpy as np
from datetime import datetime
from uuid import uuid4
from ragify.engines.scoring import ContextScoringEngine
from ragify.models import OrchestratorConfig, ContextChunk, ContextSource, SourceType, PrivacyLevel

async def demo_multi_factor_scoring():
    """Demonstrate multi-factor scoring with different content types."""
    
    print("🧪 Multi-Factor Scoring Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Create test chunks with different characteristics
    test_chunks = [
        ContextChunk(
            content="Machine learning algorithms are excellent computational methods that enable computers to learn patterns from data without being explicitly programmed. They form the foundation of artificial intelligence applications and provide amazing results.",
            source=ContextSource(
                id=str(uuid4()),
                name="ML Documentation",
                source_type=SourceType.DOCUMENT,
                url="docs://ml/algorithms",
            ),
            token_count=35,
        ),
        ContextChunk(
            content="Neural networks are a subset of machine learning that use interconnected nodes to process information. Deep learning uses multiple layers of neural networks for complex pattern recognition.",
            source=ContextSource(
                id=str(uuid4()),
                name="AI Research Paper",
                source_type=SourceType.DOCUMENT,
                url="papers://ai/neural-networks",
            ),
            token_count=32,
        ),
        ContextChunk(
            content="Natural language processing combines computational linguistics with machine learning to enable computers to understand and generate human language. It's used in chatbots and translation systems.",
            source=ContextSource(
                id=str(uuid4()),
                name="NLP Guide",
                source_type=SourceType.DOCUMENT,
                url="guide://nlp/basics",
            ),
            token_count=30,
        ),
        ContextChunk(
            content="This is a terrible implementation that doesn't work properly. The code is awful and produces horrible results. Avoid using this approach.",
            source=ContextSource(
                id=str(uuid4()),
                name="Bad Example",
                source_type=SourceType.DOCUMENT,
                url="examples://bad/implementation",
            ),
            token_count=25,
        ),
    ]
    
    print(f"📄 Created {len(test_chunks)} test chunks with different characteristics")
    
    # Test different queries
    queries = [
        "machine learning algorithms",
        "neural networks and deep learning",
        "natural language processing",
        "good implementation examples",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Score chunks
            scored_chunks = await scoring_engine.score_chunks(
                chunks=test_chunks,
                query=query,
                user_id="demo_user"
            )
            
            # Display results
            for j, chunk in enumerate(scored_chunks[:3], 1):  # Show top 3
                if chunk.relevance_score:
                    score = chunk.relevance_score.score
                    lower = chunk.relevance_score.confidence_lower
                    upper = chunk.relevance_score.confidence_upper
                    
                    print(f"  {j}. Score: {score:.3f} [{lower:.3f}, {upper:.3f}]")
                    print(f"     Content: {chunk.content[:60]}...")
                    
                    # Show individual factors
                    if chunk.relevance_score.factors:
                        print(f"     Factors:")
                        for factor, value in chunk.relevance_score.factors.items():
                            print(f"       • {factor}: {value:.3f}")
                    print()
                else:
                    print(f"  {j}. No relevance score available")
        
        except Exception as e:
            print(f"❌ Scoring failed: {e}")

async def demo_ensemble_methods():
    """Demonstrate different ensemble methods."""
    
    print(f"\n🔬 Ensemble Methods Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Test data with known scores
    test_scores = {
        'semantic_similarity': 0.85,
        'keyword_overlap': 0.72,
        'freshness': 0.91,
        'source_authority': 0.78,
        'content_quality': 0.83,
        'user_preference': 0.67,
        'contextual_relevance': 0.79,
        'sentiment_alignment': 0.88,
        'complexity_match': 0.74,
        'domain_expertise': 0.82,
    }
    
    print(f"📊 Test scores: {test_scores}")
    print()
    
    # Test individual ensemble methods
    methods = [
        ("Weighted Average", scoring_engine._calculate_ensemble_score),
        ("Geometric Mean", scoring_engine._calculate_geometric_mean),
        ("Harmonic Mean", scoring_engine._calculate_harmonic_mean),
        ("Trimmed Mean", scoring_engine._calculate_trimmed_mean),
    ]
    
    for method_name, method_func in methods:
        try:
            if method_name == "Weighted Average":
                result = method_func(test_scores)
            else:
                result = method_func(list(test_scores.values()))
            
            print(f"✅ {method_name:15}: {result:.3f}")
            
        except Exception as e:
            print(f"❌ {method_name:15}: Error - {e}")
    
    # Test multi-ensemble score
    try:
        multi_score = await scoring_engine._calculate_multi_ensemble_score(test_scores)
        print(f"✅ Multi-Ensemble: {multi_score:.3f}")
    except Exception as e:
        print(f"❌ Multi-Ensemble: Error - {e}")

async def demo_ensemble_optimization():
    """Demonstrate ensemble weight optimization."""
    
    print(f"\n⚙️  Ensemble Optimization Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Generate synthetic validation data
    np.random.seed(42)
    n_samples = 100
    
    validation_data = []
    for i in range(n_samples):
        # Generate realistic scores
        predicted_score = np.random.uniform(0.3, 0.9)
        actual_score = predicted_score + np.random.normal(0, 0.1)
        actual_score = max(0.0, min(1.0, actual_score))
        
        validation_data.append({
            'predicted_score': predicted_score,
            'actual_score': actual_score,
        })
    
    print(f"📊 Generated {len(validation_data)} validation samples")
    
    # Get initial ensemble statistics
    try:
        initial_stats = await scoring_engine.get_ensemble_statistics()
        print(f"📈 Initial ensemble statistics:")
        print(f"  Scoring factors: {initial_stats['scoring_factors']}")
        print(f"  Ensemble methods: {initial_stats['ensemble_methods']}")
        print(f"  ML ensemble enabled: {initial_stats['ml_ensemble_enabled']}")
        print(f"  ML model trained: {initial_stats['ml_model_trained']}")
        print(f"  Primary method: {initial_stats['primary_method']}")
        print(f"  Ensemble weights: {initial_stats['ensemble_weights']}")
    except Exception as e:
        print(f"❌ Failed to get initial statistics: {e}")
    
    # Test ensemble optimization
    try:
        await scoring_engine.optimize_ensemble_weights(validation_data)
        print("✅ Ensemble optimization completed")
        
        # Get updated statistics
        updated_stats = await scoring_engine.get_ensemble_statistics()
        print(f"📈 Updated ensemble weights: {updated_stats['ensemble_weights']}")
        
    except Exception as e:
        print(f"❌ Ensemble optimization failed: {e}")

async def demo_scoring_factors():
    """Demonstrate individual scoring factors."""
    
    print(f"\n🎯 Individual Scoring Factors Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Test chunks with different characteristics
    test_chunks = [
        ContextChunk(
            content="This is a great algorithm that provides excellent results. The implementation is wonderful and produces amazing outcomes.",
            source=ContextSource(
                id=str(uuid4()),
                name="Positive Content",
                source_type=SourceType.DOCUMENT,
                url="content://positive/example",
            ),
            token_count=20,
        ),
        ContextChunk(
            content="This terrible algorithm produces awful results. The implementation is horrible and creates bad outcomes.",
            source=ContextSource(
                id=str(uuid4()),
                name="Negative Content",
                source_type=SourceType.DOCUMENT,
                url="content://negative/example",
            ),
            token_count=20,
        ),
        ContextChunk(
            content="The algorithm implementation uses complex mathematical functions and advanced computational methods for data processing.",
            source=ContextSource(
                id=str(uuid4()),
                name="Technical Content",
                source_type=SourceType.DOCUMENT,
                url="content://technical/example",
            ),
            token_count=25,
        ),
    ]
    
    print(f"📄 Created {len(test_chunks)} test chunks for factor analysis")
    
    # Test different queries to trigger different factors
    test_cases = [
        {
            'query': 'great algorithm',
            'description': 'Positive sentiment query'
        },
        {
            'query': 'terrible implementation',
            'description': 'Negative sentiment query'
        },
        {
            'query': 'complex mathematical functions',
            'description': 'Technical domain query'
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print("-" * 40)
        
        try:
            # Score chunks
            scored_chunks = await scoring_engine.score_chunks(
                chunks=test_chunks,
                query=test_case['query'],
                user_id="demo_user"
            )
            
            # Display results with factor breakdown
            for j, chunk in enumerate(scored_chunks, 1):
                if chunk.relevance_score:
                    score = chunk.relevance_score.score
                    print(f"  {j}. Overall Score: {score:.3f}")
                    
                    # Show individual factors
                    if chunk.relevance_score.factors:
                        print(f"     Factor Breakdown:")
                        for factor, value in chunk.relevance_score.factors.items():
                            print(f"       • {factor}: {value:.3f}")
                    print()
                else:
                    print(f"  {j}. No relevance score available")
        
        except Exception as e:
            print(f"❌ Scoring failed: {e}")

async def demo_ensemble_configuration():
    """Demonstrate ensemble configuration options."""
    
    print(f"\n⚙️  Ensemble Configuration Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Test data
    test_scores = {
        'semantic_similarity': 0.85,
        'keyword_overlap': 0.72,
        'freshness': 0.91,
        'source_authority': 0.78,
        'content_quality': 0.83,
        'user_preference': 0.67,
        'contextual_relevance': 0.79,
        'sentiment_alignment': 0.88,
        'complexity_match': 0.74,
        'domain_expertise': 0.82,
    }
    
    print(f"📊 Test scores: {test_scores}")
    print()
    
    # Test different ensemble configurations
    configurations = [
        {
            'name': 'Default Configuration',
            'config': {
                'ensemble_weights': {
                    'weighted_average': 0.4,
                    'geometric_mean': 0.2,
                    'harmonic_mean': 0.2,
                    'trimmed_mean': 0.2,
                }
            }
        },
        {
            'name': 'Weighted Average Heavy',
            'config': {
                'ensemble_weights': {
                    'weighted_average': 0.7,
                    'geometric_mean': 0.1,
                    'harmonic_mean': 0.1,
                    'trimmed_mean': 0.1,
                }
            }
        },
        {
            'name': 'Balanced Geometric',
            'config': {
                'ensemble_weights': {
                    'weighted_average': 0.2,
                    'geometric_mean': 0.4,
                    'harmonic_mean': 0.2,
                    'trimmed_mean': 0.2,
                }
            }
        },
        {
            'name': 'Harmonic Mean Focus',
            'config': {
                'ensemble_weights': {
                    'weighted_average': 0.2,
                    'geometric_mean': 0.2,
                    'harmonic_mean': 0.4,
                    'trimmed_mean': 0.2,
                }
            }
        },
    ]
    
    for config_test in configurations:
        try:
            # Update configuration
            await scoring_engine.update_ensemble_config(config_test['config'])
            
            # Calculate score with new configuration
            score = await scoring_engine._calculate_multi_ensemble_score(test_scores)
            
            print(f"✅ {config_test['name']:20}: {score:.3f}")
            
        except Exception as e:
            print(f"❌ {config_test['name']:20}: Error - {e}")

async def demo_advanced_scoring_scenarios():
    """Demonstrate advanced scoring scenarios."""
    
    print(f"\n🚀 Advanced Scoring Scenarios Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Create diverse test scenarios
    scenarios = [
        {
            'name': 'High Quality Technical Content',
            'chunks': [
                ContextChunk(
                    content="The implementation of advanced machine learning algorithms requires sophisticated mathematical understanding and careful optimization techniques. This approach demonstrates excellent engineering practices.",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Technical Documentation",
                        source_type=SourceType.DOCUMENT,
                        url="docs://technical/advanced",
                    ),
                    token_count=30,
                ),
            ],
            'query': 'advanced machine learning implementation'
        },
        {
            'name': 'Mixed Quality Content',
            'chunks': [
                ContextChunk(
                    content="This is a basic implementation that works but could be improved. The code is functional but not optimal.",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Basic Guide",
                        source_type=SourceType.DOCUMENT,
                        url="guide://basic/implementation",
                    ),
                    token_count=25,
                ),
            ],
            'query': 'basic implementation guide'
        },
        {
            'name': 'Domain-Specific Content',
            'chunks': [
                ContextChunk(
                    content="In medical applications, machine learning algorithms must meet strict regulatory compliance requirements and demonstrate clinical validation through rigorous testing protocols.",
                    source=ContextSource(
                        id=str(uuid4()),
                        name="Medical AI Research",
                        source_type=SourceType.DOCUMENT,
                        url="research://medical/ai",
                    ),
                    token_count=28,
                ),
            ],
            'query': 'medical machine learning compliance'
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 Scenario {i}: {scenario['name']}")
        print(f"Query: '{scenario['query']}'")
        print("-" * 40)
        
        try:
            # Score chunks
            scored_chunks = await scoring_engine.score_chunks(
                chunks=scenario['chunks'],
                query=scenario['query'],
                user_id="demo_user"
            )
            
            # Display results
            for j, chunk in enumerate(scored_chunks, 1):
                if chunk.relevance_score:
                    score = chunk.relevance_score.score
                    lower = chunk.relevance_score.confidence_lower
                    upper = chunk.relevance_score.confidence_upper
                    
                    print(f"  Score: {score:.3f} [{lower:.3f}, {upper:.3f}]")
                    print(f"  Content: {chunk.content[:80]}...")
                    
                    # Show top factors
                    if chunk.relevance_score.factors:
                        sorted_factors = sorted(
                            chunk.relevance_score.factors.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        print(f"  Top Factors:")
                        for factor, value in sorted_factors[:3]:
                            print(f"    • {factor}: {value:.3f}")
                    print()
                else:
                    print(f"  No relevance score available")
        
        except Exception as e:
            print(f"❌ Scoring failed: {e}")

async def main():
    """Run the complete multi-factor scoring demo."""
    
    print("🎯 Ragify Multi-Factor Scoring with Ensemble Methods Demo")
    print("=" * 70)
    print("This demo showcases advanced multi-factor scoring with ensemble methods")
    print("for enhanced relevance assessment and ranking.\n")
    
    # Run all demos
    await demo_multi_factor_scoring()
    await demo_ensemble_methods()
    await demo_ensemble_optimization()
    await demo_scoring_factors()
    await demo_ensemble_configuration()
    await demo_advanced_scoring_scenarios()
    
    print(f"\n🎉 Complete multi-factor scoring demo finished!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✅ Multi-factor scoring (10 different factors)")
    print(f"   ✅ Multiple ensemble methods (Weighted, Geometric, Harmonic, Trimmed)")
    print(f"   ✅ Ensemble weight optimization")
    print(f"   ✅ Sentiment alignment analysis")
    print(f"   ✅ Complexity matching")
    print(f"   ✅ Domain expertise detection")
    print(f"   ✅ Contextual relevance assessment")
    print(f"   ✅ ML ensemble integration")
    print(f"   ✅ Configuration management")
    print(f"   ✅ Advanced scoring scenarios")
    print(f"\n📚 Usage Examples:")
    print(f"   # Configure ensemble weights")
    print(f"   await scoring_engine.update_ensemble_config({'ensemble_weights': {'weighted_average': 0.6, 'geometric_mean': 0.2, 'harmonic_mean': 0.1, 'trimmed_mean': 0.1}})")
    print(f"   # Optimize ensemble weights")
    print(f"   await scoring_engine.optimize_ensemble_weights(validation_data)")
    print(f"   # Get ensemble statistics")
    print(f"   stats = await scoring_engine.get_ensemble_statistics()")
    print(f"   # Update scoring weights")
    print(f"   await scoring_engine.update_scoring_weights({'semantic_similarity': 0.3, 'keyword_overlap': 0.2})")

if __name__ == "__main__":
    asyncio.run(main())
