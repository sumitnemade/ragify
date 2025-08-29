#!/usr/bin/env python3
"""
Statistical Confidence Bounds Demo for Ragify Framework

This demo showcases comprehensive statistical confidence bounds
for relevance scores using multiple statistical methods.
"""

import asyncio
import numpy as np
from datetime import datetime
from uuid import uuid4
from ragify.engines.scoring import ContextScoringEngine
from ragify.models import OrchestratorConfig, ContextChunk, ContextSource, SourceType, PrivacyLevel

async def demo_confidence_interval_methods():
    """Demonstrate different confidence interval calculation methods."""
    
    print("🧪 Confidence Interval Methods Demo")
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
    }
    
    ensemble_score = 0.79
    
    print(f"📊 Test scores: {test_scores}")
    print(f"📊 Ensemble score: {ensemble_score}")
    print()
    
    # Test different confidence interval methods
    methods = [
        ("Simple", scoring_engine._calculate_simple_confidence_interval),
        ("Bootstrap", scoring_engine._calculate_bootstrap_confidence_interval),
        ("T-distribution", scoring_engine._calculate_t_confidence_interval),
        ("Normal", scoring_engine._calculate_normal_confidence_interval),
        ("Weighted", scoring_engine._calculate_weighted_confidence_interval),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        try:
            if method_name == "Bootstrap":
                result = await method_func(list(test_scores.values()), ensemble_score)
            elif method_name == "Weighted":
                result = await method_func(test_scores, ensemble_score)
            else:
                result = await method_func(list(test_scores.values()), ensemble_score)
            
            if result:
                results[method_name] = result
                interval_width = result[1] - result[0]
                print(f"✅ {method_name:12}: [{result[0]:.3f}, {result[1]:.3f}] (width: {interval_width:.3f})")
            else:
                print(f"❌ {method_name:12}: Failed")
                
        except Exception as e:
            print(f"❌ {method_name:12}: Error - {e}")
    
    # Test combined confidence interval
    try:
        confidence_intervals = [ci for ci in results.values() if ci]
        if confidence_intervals:
            combined_ci = await scoring_engine._combine_confidence_intervals(confidence_intervals)
            interval_width = combined_ci[1] - combined_ci[0]
            print(f"✅ {'Combined':12}: [{combined_ci[0]:.3f}, {combined_ci[1]:.3f}] (width: {interval_width:.3f})")
        else:
            print("❌ Combined: No valid intervals to combine")
    except Exception as e:
        print(f"❌ Combined: Error - {e}")
    
    return results

async def demo_confidence_calibration():
    """Demonstrate confidence interval calibration."""
    
    print(f"\n🔧 Confidence Interval Calibration Demo")
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
        predicted_lower = max(0.0, predicted_score - np.random.uniform(0.05, 0.15))
        predicted_upper = min(1.0, predicted_score + np.random.uniform(0.05, 0.15))
        
        # Generate actual score (with some noise)
        actual_score = predicted_score + np.random.normal(0, 0.1)
        actual_score = max(0.0, min(1.0, actual_score))
        
        validation_data.append({
            'predicted_score': predicted_score,
            'predicted_lower': predicted_lower,
            'predicted_upper': predicted_upper,
            'actual_score': actual_score,
        })
    
    print(f"📊 Generated {len(validation_data)} validation samples")
    
    # Test calibration
    try:
        await scoring_engine.calibrate_confidence_intervals(validation_data)
        print("✅ Confidence interval calibration completed")
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
    
    # Test validation
    try:
        validation_metrics = await scoring_engine.validate_confidence_intervals(validation_data)
        print(f"✅ Validation completed:")
        print(f"  Coverage rate: {validation_metrics['coverage_rate']:.3f}")
        print(f"  Mean interval width: {validation_metrics['mean_interval_width']:.3f}")
        print(f"  Calibration error: {validation_metrics['calibration_error']:.3f}")
        print(f"  Reliability score: {validation_metrics['reliability_score']:.3f}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

async def demo_confidence_statistics():
    """Demonstrate confidence interval statistics."""
    
    print(f"\n📈 Confidence Interval Statistics Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Add some scoring history
    scoring_engine.scoring_history = [
        {
            'score': 0.85,
            'confidence_lower': 0.78,
            'confidence_upper': 0.92,
            'timestamp': datetime.utcnow(),
        },
        {
            'score': 0.72,
            'confidence_lower': 0.65,
            'confidence_upper': 0.79,
            'timestamp': datetime.utcnow(),
        },
        {
            'score': 0.91,
            'confidence_lower': 0.87,
            'confidence_upper': 0.95,
            'timestamp': datetime.utcnow(),
        },
        {
            'score': 0.68,
            'confidence_lower': 0.61,
            'confidence_upper': 0.75,
            'timestamp': datetime.utcnow(),
        },
        {
            'score': 0.94,
            'confidence_lower': 0.89,
            'confidence_upper': 0.99,
            'timestamp': datetime.utcnow(),
        },
    ]
    
    # Add calibration data
    scoring_engine.confidence_calibration_data = [
        {'predicted': 0.8, 'actual': 0.82},
        {'predicted': 0.7, 'actual': 0.71},
        {'predicted': 0.9, 'actual': 0.88},
        {'predicted': 0.6, 'actual': 0.62},
        {'predicted': 0.85, 'actual': 0.83},
    ]
    
    try:
        stats = await scoring_engine.get_confidence_statistics()
        print(f"✅ Confidence statistics retrieved:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Mean confidence width: {stats['mean_confidence_width']:.3f}")
        print(f"  Std confidence width: {stats.get('std_confidence_width', 0.0):.3f}")
        print(f"  Min confidence width: {stats.get('min_confidence_width', 0.0):.3f}")
        print(f"  Max confidence width: {stats.get('max_confidence_width', 0.0):.3f}")
        print(f"  Confidence level: {stats['confidence_level']:.3f}")
        print(f"  Calibration status: {stats['calibration_status']}")
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")

async def demo_real_scoring_with_confidence():
    """Demonstrate real scoring with confidence intervals."""
    
    print(f"\n🎯 Real Scoring with Confidence Intervals Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Create test chunks with different content types
    test_chunks = [
        ContextChunk(
            content="Machine learning algorithms are computational methods that enable computers to learn patterns from data without being explicitly programmed. They form the foundation of artificial intelligence applications.",
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
    ]
    
    print(f"📄 Created {len(test_chunks)} test chunks with different content")
    
    try:
        # Score chunks
        scored_chunks = await scoring_engine.score_chunks(
            chunks=test_chunks,
            query="machine learning algorithms and neural networks",
            user_id="demo_user"
        )
        
        print(f"✅ Scored {len(scored_chunks)} chunks")
        print()
        
        # Display results with confidence intervals
        for i, chunk in enumerate(scored_chunks, 1):
            if chunk.relevance_score:
                score = chunk.relevance_score.score
                lower = chunk.relevance_score.confidence_lower
                upper = chunk.relevance_score.confidence_upper
                confidence_level = chunk.relevance_score.confidence_level
                interval_width = upper - lower
                
                print(f"📊 Chunk {i}:")
                print(f"  Score: {score:.3f} [{lower:.3f}, {upper:.3f}] (confidence: {confidence_level:.1%})")
                print(f"  Interval width: {interval_width:.3f}")
                print(f"  Content: {chunk.content[:80]}...")
                
                # Show individual factors
                if chunk.relevance_score.factors:
                    print(f"  Factors:")
                    for factor, value in chunk.relevance_score.factors.items():
                        print(f"    • {factor}: {value:.3f}")
                print()
            else:
                print(f"❌ Chunk {i}: No relevance score available")
        
    except Exception as e:
        print(f"❌ Scoring failed: {e}")

async def demo_confidence_configuration():
    """Demonstrate confidence interval configuration options."""
    
    print(f"\n⚙️  Confidence Configuration Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Test different confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    
    test_scores = {
        'semantic_similarity': 0.85,
        'keyword_overlap': 0.72,
        'freshness': 0.91,
        'source_authority': 0.78,
        'content_quality': 0.83,
        'user_preference': 0.67,
    }
    
    ensemble_score = 0.79
    
    print(f"📊 Testing different confidence levels:")
    print(f"📊 Test scores: {test_scores}")
    print(f"📊 Ensemble score: {ensemble_score}")
    print()
    
    for confidence_level in confidence_levels:
        try:
            # Update confidence level
            scoring_engine.confidence_config['default_confidence_level'] = confidence_level
            
            # Calculate confidence interval
            lower, upper = await scoring_engine._calculate_simple_confidence_interval(
                test_scores, ensemble_score
            )
            
            interval_width = upper - lower
            print(f"  {confidence_level:.0%} confidence: [{lower:.3f}, {upper:.3f}] (width: {interval_width:.3f})")
            
        except Exception as e:
            print(f"  ❌ {confidence_level:.0%} confidence: Error - {e}")
    
    # Test configuration options
    print(f"\n🔧 Testing configuration options:")
    
    # Test with bootstrap disabled
    scoring_engine.confidence_config['use_bootstrap'] = False
    try:
        lower, upper = await scoring_engine._calculate_confidence_interval(test_scores, ensemble_score)
        interval_width = upper - lower
        print(f"  Bootstrap disabled: [{lower:.3f}, {upper:.3f}] (width: {interval_width:.3f})")
    except Exception as e:
        print(f"  ❌ Bootstrap disabled: Error - {e}")
    
    # Test with t-distribution disabled
    scoring_engine.confidence_config['use_t_distribution'] = False
    try:
        lower, upper = await scoring_engine._calculate_confidence_interval(test_scores, ensemble_score)
        interval_width = upper - lower
        print(f"  T-distribution disabled: [{lower:.3f}, {upper:.3f}] (width: {interval_width:.3f})")
    except Exception as e:
        print(f"  ❌ T-distribution disabled: Error - {e}")

async def demo_advanced_statistical_methods():
    """Demonstrate advanced statistical methods."""
    
    print(f"\n🔬 Advanced Statistical Methods Demo")
    print("=" * 50)
    
    # Initialize scoring engine
    config = OrchestratorConfig(
        vector_db_url="memory://",
        privacy_level=PrivacyLevel.PUBLIC
    )
    scoring_engine = ContextScoringEngine(config)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    # Create multiple scoring scenarios
    scenarios = [
        {
            'name': 'High Variance',
            'scores': {
                'semantic_similarity': 0.95,
                'keyword_overlap': 0.30,
                'freshness': 0.85,
                'source_authority': 0.45,
                'content_quality': 0.90,
                'user_preference': 0.20,
            }
        },
        {
            'name': 'Low Variance',
            'scores': {
                'semantic_similarity': 0.75,
                'keyword_overlap': 0.70,
                'freshness': 0.80,
                'source_authority': 0.75,
                'content_quality': 0.78,
                'user_preference': 0.72,
            }
        },
        {
            'name': 'Mixed Quality',
            'scores': {
                'semantic_similarity': 0.60,
                'keyword_overlap': 0.85,
                'freshness': 0.95,
                'source_authority': 0.40,
                'content_quality': 0.65,
                'user_preference': 0.80,
            }
        },
    ]
    
    print(f"📊 Testing different scoring scenarios:")
    print()
    
    for scenario in scenarios:
        scores = scenario['scores']
        ensemble_score = sum(scores.values()) / len(scores)
        
        print(f"🎯 {scenario['name']}:")
        print(f"  Scores: {scores}")
        print(f"  Ensemble: {ensemble_score:.3f}")
        
        try:
            # Calculate confidence interval
            lower, upper = await scoring_engine._calculate_confidence_interval(scores, ensemble_score)
            interval_width = upper - lower
            
            # Calculate variance of scores
            score_values = list(scores.values())
            variance = np.var(score_values)
            
            print(f"  Confidence: [{lower:.3f}, {upper:.3f}] (width: {interval_width:.3f})")
            print(f"  Score variance: {variance:.3f}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

async def main():
    """Run the complete confidence bounds demo."""
    
    print("🎯 Ragify Statistical Confidence Bounds Demo")
    print("=" * 60)
    print("This demo showcases comprehensive statistical confidence bounds")
    print("for relevance scores using multiple statistical methods.\n")
    
    # Run all demos
    await demo_confidence_interval_methods()
    await demo_confidence_calibration()
    await demo_confidence_statistics()
    await demo_real_scoring_with_confidence()
    await demo_confidence_configuration()
    await demo_advanced_statistical_methods()
    
    print(f"\n🎉 Complete confidence bounds demo finished!")
    print(f"\n💡 Key Features Demonstrated:")
    print(f"   ✅ Multiple confidence interval methods (Bootstrap, T-distribution, Normal, Weighted)")
    print(f"   ✅ Confidence interval calibration and validation")
    print(f"   ✅ Statistical confidence bounds for relevance scores")
    print(f"   ✅ Configuration options for different confidence levels")
    print(f"   ✅ Real scoring with comprehensive confidence intervals")
    print(f"   ✅ Advanced statistical methods and scenarios")
    print(f"   ✅ Robust statistical methods with fallback mechanisms")
    print(f"\n📚 Usage Examples:")
    print(f"   # Configure confidence level")
    print(f"   scoring_engine.confidence_config['default_confidence_level'] = 0.95")
    print(f"   # Enable/disable methods")
    print(f"   scoring_engine.confidence_config['use_bootstrap'] = True")
    print(f"   scoring_engine.confidence_config['use_t_distribution'] = True")
    print(f"   # Calibrate confidence intervals")
    print(f"   await scoring_engine.calibrate_confidence_intervals(validation_data)")
    print(f"   # Get confidence statistics")
    print(f"   stats = await scoring_engine.get_confidence_statistics()")

if __name__ == "__main__":
    asyncio.run(main())
