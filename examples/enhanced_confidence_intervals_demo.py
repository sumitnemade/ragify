"""
Enhanced Confidence Intervals Demo for RAGify.

This example demonstrates the enterprise-grade confidence interval capabilities
including advanced bootstrap strategies, robust error handling, and statistical validation.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any

from ragify.engines.scoring import ContextScoringEngine
from ragify.models import OrchestratorConfig, PrivacyLevel


async def demonstrate_enhanced_confidence_intervals():
    """Demonstrate enhanced confidence interval capabilities."""
    print("🚀 **Enhanced Confidence Intervals Demo**")
    print("=" * 50)
    
    # Initialize scoring engine with config
    config = OrchestratorConfig(
        max_context_size=1000,
        cache_ttl=300,
        privacy_level=PrivacyLevel.PRIVATE,
        enable_logging=True
    )
    scoring_engine = ContextScoringEngine(config=config)
    
    # Display configuration
    print("\n📋 **Configuration Overview:**")
    print(f"Default Confidence Level: {scoring_engine.confidence_config['default_confidence_level']}")
    print(f"Bootstrap Samples: {scoring_engine.confidence_config['bootstrap_samples']}")
    print(f"Min Sample Size: {scoring_engine.confidence_config['min_sample_size']}")
    print(f"Bootstrap Strategies: {', '.join(scoring_engine.confidence_config['bootstrap_strategies'])}")
    print(f"Robust Estimation: {scoring_engine.confidence_config['robust_estimation']}")
    print(f"Outlier Detection: {scoring_engine.confidence_config['outlier_detection']}")
    print(f"Normality Testing: {scoring_engine.confidence_config['normality_testing']}")
    
    # Test data scenarios
    test_scenarios = [
        ("Normal Distribution", generate_normal_data(20, 0.8, 0.1)),
        ("Skewed Distribution", generate_skewed_data(20, 2, 5)),
        ("Data with Outliers", generate_data_with_outliers(20, 0.8, 0.1, 4)),
        ("Uniform Distribution", generate_uniform_data(20, 0.6, 0.4)),
        ("Bimodal Distribution", generate_bimodal_data(20, 0.3, 0.7, 0.1))
    ]
    
    for scenario_name, data in test_scenarios:
        print(f"\n🔍 **Scenario: {scenario_name}**")
        print(f"Data: {[f'{x:.3f}' for x in data[:10]]}{'...' if len(data) > 10 else ''}")
        print(f"Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
        
        await analyze_data_quality(scoring_engine, data)
        await demonstrate_confidence_intervals(scoring_engine, data, np.mean(data))


def generate_normal_data(n: int, mean: float, std: float) -> List[float]:
    """Generate normally distributed data."""
    np.random.seed(42)
    return np.random.normal(mean, std, n).clip(0, 1).tolist()


def generate_skewed_data(n: int, alpha: float, beta: float) -> List[float]:
    """Generate beta-distributed (skewed) data."""
    np.random.seed(42)
    return np.random.beta(alpha, beta, n).tolist()


def generate_data_with_outliers(n: int, mean: float, std: float, n_outliers: int) -> List[float]:
    """Generate data with outliers."""
    np.random.seed(42)
    normal_data = np.random.normal(mean, std, n - n_outliers).clip(0, 1).tolist()
    
    # Add outliers
    outliers = [0.05, 0.95, 0.02, 0.98]
    return normal_data + outliers[:n_outliers]


def generate_uniform_data(n: int, low: float, high: float) -> List[float]:
    """Generate uniformly distributed data."""
    np.random.seed(42)
    return np.random.uniform(low, high, n).tolist()


def generate_bimodal_data(n: int, mean1: float, mean2: float, std: float) -> List[float]:
    """Generate bimodal data."""
    np.random.seed(42)
    n1 = n // 2
    n2 = n - n1
    
    data1 = np.random.normal(mean1, std, n1).clip(0, 1)
    data2 = np.random.normal(mean2, std, n2).clip(0, 1)
    
    return np.concatenate([data1, data2]).tolist()


async def analyze_data_quality(scoring_engine: ContextScoringEngine, data: List[float]):
    """Analyze data quality using enhanced methods."""
    print("\n📊 **Data Quality Analysis:**")
    
    # Data quality validation
    is_valid = scoring_engine._validate_data_quality(data)
    print(f"Data Quality Valid: {is_valid}")
    
    if is_valid:
        # Outlier detection
        clean_data, outlier_info = scoring_engine._detect_and_handle_outliers(data)
        print(f"Outlier Ratio: {outlier_info['outlier_ratio']:.3f}")
        print(f"Outlier Detection Method: {outlier_info['method']}")
        
        if outlier_info['outlier_ratio'] > 0:
            print(f"Outliers Found: {outlier_info['outlier_indices']}")
        
        # Normality testing
        normality_test = scoring_engine._test_normality(clean_data)
        print(f"Normality Test: {'Normal' if normality_test['is_normal'] else 'Non-Normal'}")
        print(f"P-Value: {normality_test['p_value']:.4f}")
        print(f"Test Method: {normality_test['method']}")
        
        # Homoscedasticity testing
        homoscedasticity_test = scoring_engine._test_homoscedasticity(clean_data)
        print(f"Homoscedasticity: {'Constant Variance' if homoscedasticity_test['is_homoscedastic'] else 'Heteroscedastic'}")
        print(f"P-Value: {homoscedasticity_test['p_value']:.4f}")
        
        # Bootstrap strategy selection
        strategy = scoring_engine._select_bootstrap_strategy(clean_data, normality_test)
        print(f"Recommended Bootstrap Strategy: {strategy}")


async def demonstrate_confidence_intervals(scoring_engine: ContextScoringEngine, data: List[float], ensemble_score: float):
    """Demonstrate various confidence interval methods."""
    print("\n🎯 **Confidence Interval Methods:**")
    
    methods = [
        ("Bootstrap", scoring_engine._calculate_bootstrap_confidence_interval),
        ("T-Distribution", scoring_engine._calculate_t_confidence_interval),
        ("Normal Distribution", scoring_engine._calculate_normal_confidence_interval),
        ("Weighted", scoring_engine._calculate_weighted_confidence_interval),
        ("Robust", scoring_engine._calculate_robust_confidence_interval)
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        try:
            start_time = time.time()
            ci = await method_func(data, ensemble_score)
            end_time = time.time()
            
            if ci:
                lower, upper = ci
                width = upper - lower
                results[method_name] = {
                    'ci': ci,
                    'width': width,
                    'time': end_time - start_time
                }
                
                print(f"{method_name:20} CI: [{lower:.4f}, {upper:.4f}] Width: {width:.4f} Time: {(end_time - start_time)*1000:.2f}ms")
            else:
                print(f"{method_name:20} CI: Failed/Not applicable")
                
        except Exception as e:
            print(f"{method_name:20} CI: Error - {e}")
    
    # Compare methods
    if results:
        print(f"\n📈 **Method Comparison:**")
        best_method = min(results.keys(), key=lambda x: results[x]['width'])
        fastest_method = min(results.keys(), key=lambda x: results[x]['time'])
        
        print(f"Most Precise: {best_method} (width: {results[best_method]['width']:.4f})")
        print(f"Fastest: {fastest_method} (time: {results[fastest_method]['time']*1000:.2f}ms)")
        
        # Validate confidence intervals
        print(f"\n✅ **Validation Results:**")
        for method_name, result in results.items():
            is_valid = scoring_engine._validate_confidence_interval(result['ci'], data)
            print(f"{method_name:20} Valid: {is_valid}")


async def demonstrate_advanced_bootstrap_features(scoring_engine: ContextScoringEngine):
    """Demonstrate advanced bootstrap features."""
    print("\n🚀 **Advanced Bootstrap Features:**")
    
    # Generate test data
    data = generate_normal_data(25, 0.8, 0.1)
    ensemble_score = np.mean(data)
    
    print(f"Test Data: {len(data)} samples, mean: {ensemble_score:.3f}")
    
    # Test different bootstrap strategies
    strategies = scoring_engine.confidence_config['bootstrap_strategies']
    
    for strategy in strategies:
        print(f"\n🔧 **Strategy: {strategy.upper()}**")
        
        # Simulate bootstrap results for demonstration
        if strategy == 'percentile':
            ci = await scoring_engine._percentile_bootstrap(data, ensemble_score)
        elif strategy == 'bca':
            ci = await scoring_engine._bca_bootstrap(data, ensemble_score)
        elif strategy == 'abc':
            ci = await scoring_engine._abc_bootstrap(data, ensemble_score)
        elif strategy == 'studentized':
            ci = await scoring_engine._studentized_bootstrap(data, ensemble_score)
        else:
            ci = None
        
        if ci:
            lower, upper = ci
            width = upper - lower
            print(f"  CI: [{lower:.4f}, {upper:.4f}] Width: {width:.4f}")
        else:
            print(f"  CI: Failed/Not applicable")


async def demonstrate_error_handling_and_fallbacks(scoring_engine: ContextScoringEngine):
    """Demonstrate error handling and fallback mechanisms."""
    print("\n🛡️ **Error Handling & Fallbacks:**")
    
    # Test with problematic data
    problematic_scenarios = [
        ("Empty Data", []),
        ("Single Value", [0.8]),
        ("NaN Values", [0.8, np.nan, 0.9]),
        ("Infinite Values", [0.8, np.inf, 0.9]),
        ("Out of Range", [0.8, 1.5, 0.9]),
        ("No Variation", [0.8, 0.8, 0.8, 0.8, 0.8])
    ]
    
    for scenario_name, data in problematic_scenarios:
        print(f"\n🔍 **Scenario: {scenario_name}**")
        
        try:
            # Test main confidence interval method
            ci = await scoring_engine._calculate_confidence_interval(data, 0.8)
            
            if ci:
                lower, upper = ci
                print(f"  Result: [{lower:.4f}, {upper:.4f}]")
            else:
                print(f"  Result: None (fallback used)")
                
        except Exception as e:
            print(f"  Error: {e}")


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n⚡ **Performance Optimization:**")
    
    # Test with different sample sizes
    sample_sizes = [10, 50, 100, 500, 1000]
    
    for n in sample_sizes:
        print(f"\n📊 **Sample Size: {n}**")
        
        # Generate data
        data = generate_normal_data(n, 0.8, 0.1)
        ensemble_score = np.mean(data)
        
        # Time different methods
        methods = [
            ("Bootstrap", "bootstrap"),
            ("T-Distribution", "t_distribution"),
            ("Normal", "normal")
        ]
        
        for method_name, method_key in methods:
            try:
                start_time = time.time()
                
                if method_key == "bootstrap":
                    ci = await scoring_engine._calculate_bootstrap_confidence_interval(data, ensemble_score)
                elif method_key == "t_distribution":
                    ci = await scoring_engine._calculate_t_confidence_interval(data, ensemble_score)
                elif method_key == "normal":
                    ci = await scoring_engine._calculate_normal_confidence_interval(data, ensemble_score)
                
                end_time = time.time()
                
                if ci:
                    print(f"  {method_name:15} Time: {(end_time - start_time)*1000:.2f}ms")
                else:
                    print(f"  {method_name:15} Time: Failed")
                    
            except Exception as e:
                print(f"  {method_name:15} Time: Error - {e}")


async def main():
    """Main demonstration function."""
    print("🎯 **RAGify Enhanced Confidence Intervals - Enterprise Grade Demo**")
    print("=" * 70)
    
    try:
        # Core demonstrations
        await demonstrate_enhanced_confidence_intervals()
        
        # Advanced features
        config = OrchestratorConfig(
            max_context_size=1000,
            cache_ttl=300,
            privacy_level=PrivacyLevel.PRIVATE,
            enable_logging=True
        )
        scoring_engine = ContextScoringEngine(config=config)
        await demonstrate_advanced_bootstrap_features(scoring_engine)
        
        # Error handling
        await demonstrate_error_handling_and_fallbacks(scoring_engine)
        
        # Performance
        await demonstrate_performance_optimization()
        
        print("\n🎉 **Demo Completed Successfully!**")
        print("\n✅ **Key Features Demonstrated:**")
        print("  • Advanced bootstrap strategies (BCa, ABC, Studentized)")
        print("  • Comprehensive data quality validation")
        print("  • Robust outlier detection and handling")
        print("  • Multiple normality and homoscedasticity tests")
        print("  • Intelligent method selection")
        print("  • Comprehensive error handling and fallbacks")
        print("  • Performance optimization features")
        print("  • Enterprise-grade statistical rigor")
        
    except Exception as e:
        print(f"\n❌ **Demo failed with error: {e}**")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
