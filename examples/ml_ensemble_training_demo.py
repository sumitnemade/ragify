#!/usr/bin/env python3
"""
ML Ensemble Training Demo for RAGify

This example demonstrates the comprehensive ML ensemble training capabilities
including model training, persistence, cross-validation, hyperparameter optimization,
and model selection.
"""

import asyncio
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ragify.engines.scoring import ContextScoringEngine
from ragify.models import OrchestratorConfig, ContextChunk, ContextSource
from uuid import uuid4


async def create_training_data():
    """Create comprehensive training data for ML ensemble training."""
    print("🔄 Creating training data...")
    
    # Create mock context chunks with realistic data
    chunks = []
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning concepts",
        "What are the applications of AI?",
        "How to implement ML algorithms?",
        "What is supervised learning?",
        "Explain unsupervised learning",
        "How to evaluate ML models?",
        "What is cross-validation?",
        "How to handle overfitting?",
        "What is regularization?",
        "How to tune hyperparameters?",
        "What is ensemble learning?",
        "Explain random forests",
        "How does gradient boosting work?",
        "What is transfer learning?",
        "How to preprocess data?",
        "What is feature engineering?",
        "How to handle missing data?",
        "What is data augmentation?"
    ]
    
    # Create realistic content for each chunk
    contents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns through training on large datasets.",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
        "Artificial Intelligence has numerous applications including autonomous vehicles, medical diagnosis, recommendation systems, fraud detection, natural language processing, and robotics. It's transforming industries across the globe.",
        "Implementing ML algorithms involves several steps: data collection and preprocessing, feature engineering, model selection, training, validation, and deployment. Each step requires careful consideration and best practices.",
        "Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on new, unseen data. It's used for classification and regression tasks.",
        "Unsupervised learning finds hidden patterns in unlabeled data without predefined outputs. Common techniques include clustering, dimensionality reduction, and association rule learning.",
        "ML model evaluation involves metrics like accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation and holdout sets help ensure robust performance assessment.",
        "Cross-validation is a technique to assess how well a model will generalize to new data by dividing the dataset into multiple folds and training/testing on different combinations.",
        "Overfitting occurs when a model learns the training data too well but fails to generalize. Techniques like regularization, early stopping, and data augmentation help prevent this.",
        "Regularization adds constraints to model parameters to prevent overfitting. Common methods include L1 (Lasso), L2 (Ridge), and dropout in neural networks.",
        "Hyperparameter tuning involves finding optimal values for model parameters that aren't learned during training. Techniques include grid search, random search, and Bayesian optimization.",
        "Ensemble learning combines multiple models to improve prediction accuracy and robustness. Popular methods include bagging, boosting, and stacking.",
        "Random forests are ensemble methods that build multiple decision trees and combine their predictions. They're robust, handle non-linear relationships, and provide feature importance.",
        "Gradient boosting builds models sequentially, each correcting the errors of the previous one. It's powerful for structured data and often achieves state-of-the-art performance.",
        "Transfer learning leverages knowledge from pre-trained models on large datasets to improve performance on smaller, related tasks. It's widely used in computer vision and NLP.",
        "Data preprocessing includes cleaning, normalization, encoding categorical variables, and handling missing values. Proper preprocessing is crucial for model performance.",
        "Feature engineering creates new features from raw data to improve model performance. It requires domain knowledge and creativity to identify useful transformations.",
        "Missing data can be handled through imputation (mean, median, mode), deletion, or advanced methods like multiple imputation. The approach depends on the data and context.",
        "Data augmentation creates new training examples by applying transformations to existing data. It's particularly useful for computer vision and helps prevent overfitting."
    ]
    
    # Create relevance scores (simulating user feedback)
    relevance_scores = [
        0.95, 0.92, 0.88, 0.85, 0.90, 0.87, 0.83, 0.89, 0.91, 0.86,
        0.84, 0.88, 0.90, 0.93, 0.89, 0.85, 0.87, 0.86, 0.82, 0.88
    ]
    
    # Create context chunks
    for i, (query, content, score) in enumerate(zip(queries, contents, relevance_scores)):
        chunk = ContextChunk(
            id=uuid4(),
            content=content,
            source=ContextSource(
                id=uuid4(),
                name=f"training_source_{i}",
                source_type="document",
                authority_score=0.8 + (i % 3) * 0.1  # Varying authority scores
            ),
            token_count=len(content.split()),
            metadata={
                "source": f"training_source_{i}",
                "category": "machine_learning",
                "difficulty": "intermediate",
                "length": len(content)
            }
        )
        chunks.append((query, chunk, score))
    
    print(f"✅ Created {len(chunks)} training samples")
    return chunks


async def demonstrate_basic_training(scoring_engine, training_data):
    """Demonstrate basic ML model training."""
    print("\n🚀 Demonstrating Basic ML Model Training...")
    
    # Extract query-chunk pairs and scores
    query_chunk_pairs = [(query, chunk) for query, chunk, score in training_data]
    relevance_scores = [score for query, chunk, score in training_data]
    
    # Train the model with basic settings
    result = await scoring_engine.train_on_feedback(
        query_chunk_pairs,
        relevance_scores,
        enable_cross_validation=True,
        enable_hyperparameter_optimization=False  # Start simple
    )
    
    if result['success']:
        print(f"✅ Training successful!")
        print(f"   Samples trained: {result['samples_trained']}")
        print(f"   Validation samples: {result['validation_samples']}")
        print(f"   Model: {result['model_name']}")
        
        # Show validation metrics
        metrics = result['validation_metrics']
        print(f"   R² Score: {metrics['r2_score']:.3f}")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f}")
        
        # Show cross-validation results
        cv_scores = result['cross_validation_scores']
        print(f"   CV R² (mean ± std): {cv_scores['r2_mean']:.3f} ± {cv_scores['r2_std']:.3f}")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    return result


async def demonstrate_advanced_training(scoring_engine, training_data):
    """Demonstrate advanced ML model training with hyperparameter optimization."""
    print("\n🔬 Demonstrating Advanced ML Model Training...")
    
    # Extract query-chunk pairs and scores
    query_chunk_pairs = [(query, chunk) for query, chunk, score in training_data]
    relevance_scores = [score for query, chunk, score in training_data]
    
    # Train with all advanced features enabled
    result = await scoring_engine.train_on_feedback(
        query_chunk_pairs,
        relevance_scores,
        enable_cross_validation=True,
        enable_hyperparameter_optimization=True
    )
    
    if result['success']:
        print(f"✅ Advanced training successful!")
        print(f"   Samples trained: {result['samples_trained']}")
        print(f"   Validation samples: {result['validation_samples']}")
        print(f"   Model: {result['model_name']}")
        
        # Show validation metrics
        metrics = result['validation_metrics']
        print(f"   R² Score: {metrics['r2_score']:.3f}")
        print(f"   RMSE: {metrics['rmse']:.3f}")
        print(f"   MAE: {metrics['mae']:.3f}")
        
        # Show cross-validation results
        cv_scores = result['cross_validation_scores']
        print(f"   CV R² (mean ± std): {cv_scores['r2_mean']:.3f} ± {cv_scores['r2_std']:.3f}")
        
        # Show that hyperparameter optimization was performed
        if result['cross_validation_scores']:
            print(f"   Hyperparameter optimization: ✅ Enabled")
    else:
        print(f"❌ Advanced training failed: {result.get('error', 'Unknown error')}")
    
    return result


async def demonstrate_model_persistence(scoring_engine, temp_dir):
    """Demonstrate model persistence capabilities."""
    print("\n💾 Demonstrating Model Persistence...")
    
    # Configure model directory
    scoring_engine.model_persistence_config['model_dir'] = temp_dir
    
    # Save the model
    save_success = await scoring_engine._save_model()
    if save_success:
        print(f"✅ Model saved successfully to {temp_dir}")
        
        # Check what files were created
        model_dir = Path(temp_dir)
        model_files = list(model_dir.glob("*"))
        print(f"   Files created: {[f.name for f in model_files]}")
        
        # Create new engine and load model
        new_config = OrchestratorConfig(
            model_name="test-model",
            embedding_model="all-MiniLM-L6-v2"
        )
        new_engine = ContextScoringEngine(new_config)
        new_engine.model_persistence_config['model_dir'] = temp_dir
        
        # Load the model
        load_success = await new_engine._load_model()
        if load_success:
            print(f"✅ Model loaded successfully in new engine")
            print(f"   Is trained: {new_engine._is_trained}")
            print(f"   Is scaler fitted: {new_engine._is_scaler_fitted}")
        else:
            print(f"❌ Failed to load model")
    else:
        print(f"❌ Failed to save model")


async def demonstrate_model_selection(scoring_engine, training_data):
    """Demonstrate automatic model selection."""
    print("\n🤖 Demonstrating Automatic Model Selection...")
    
    # Create validation data
    validation_data = training_data[:10]  # Use first 10 samples
    
    # Select best model
    best_model_name = await scoring_engine.select_best_model(validation_data)
    
    print(f"✅ Best model selected: {best_model_name}")
    print(f"   Current model: {scoring_engine.current_model_name}")
    print(f"   Available models: {list(scoring_engine.ml_models.keys())}")
    
    # Show model switching
    if best_model_name != scoring_engine.current_model_name:
        print(f"   Model switched from {scoring_engine.current_model_name} to {best_model_name}")
    else:
        print(f"   Current model was already the best choice")


async def demonstrate_model_retraining(scoring_engine, training_data):
    """Demonstrate model retraining capabilities."""
    print("\n🔄 Demonstrating Model Retraining...")
    
    # Create new data for retraining
    new_data = training_data[:15]  # Use first 15 samples
    
    # Retrain with new data
    retrain_success = await scoring_engine.retrain_model(new_data, retrain_frequency=10)
    
    if retrain_success:
        print(f"✅ Model retrained successfully!")
        print(f"   New samples used: {len(new_data)}")
        print(f"   Retraining threshold: 10")
        
        # Show updated training history
        print(f"   Total training sessions: {len(scoring_engine.training_history)}")
        latest_training = scoring_engine.training_history[-1]
        print(f"   Latest training: {latest_training['timestamp']}")
    else:
        print(f"❌ Model retraining failed")


async def demonstrate_model_info(scoring_engine):
    """Demonstrate getting comprehensive model information."""
    print("\n📊 Demonstrating Model Information...")
    
    # Get comprehensive model info
    model_info = await scoring_engine.get_model_info()
    
    print(f"✅ Model information retrieved:")
    print(f"   Model name: {model_info['model_name']}")
    print(f"   Model type: {model_info['model_type']}")
    print(f"   Is trained: {model_info['is_trained']}")
    print(f"   Is scaler fitted: {model_info['is_scaler_fitted']}")
    print(f"   Training samples: {model_info['training_samples']}")
    print(f"   Available models: {model_info['available_models']}")
    
    # Show persistence configuration
    persistence = model_info['model_persistence']
    print(f"   Persistence enabled: {persistence['enabled']}")
    print(f"   Model directory: {persistence['model_dir']}")
    print(f"   Auto-save: {persistence['auto_save']}")
    
    # Show training configuration
    training_config = model_info['training_config']
    print(f"   Validation split: {training_config['validation_split']}")
    print(f"   CV folds: {training_config['cross_validation_folds']}")
    print(f"   Hyperparameter optimization: {training_config['hyperparameter_optimization']}")


async def demonstrate_feature_extraction(scoring_engine, training_data):
    """Demonstrate feature extraction capabilities."""
    print("\n🔍 Demonstrating Feature Extraction...")
    
    # Extract features from a sample
    sample_query, sample_chunk, sample_score = training_data[0]
    
    features = await scoring_engine._extract_features(sample_query, sample_chunk)
    
    print(f"✅ Feature extraction successful!")
    print(f"   Query: '{sample_query[:50]}...'")
    print(f"   Content length: {len(sample_chunk.content)}")
    print(f"   Features extracted: {len(features)}")
    print(f"   Feature types: {[type(f).__name__ for f in features[:5]]}...")
    
    # Show some feature values
    print(f"   Sample features:")
    print(f"     Query length: {features[0]}")
    print(f"     Content length: {features[1]}")
    print(f"     Token count: {features[2]}")
    print(f"     Semantic score: {features[3]:.3f}")
    print(f"     Content quality: {features[4]:.3f}")


async def demonstrate_cross_validation(scoring_engine, training_data):
    """Demonstrate cross-validation capabilities."""
    print("\n✂️ Demonstrating Cross-Validation...")
    
    # Extract features and labels
    query_chunk_pairs = [(query, chunk) for query, chunk, score in training_data]
    relevance_scores = [score for query, chunk, score in training_data]
    
    # Extract features
    features = []
    for query, chunk in query_chunk_pairs:
        feature_vector = await scoring_engine._extract_features(query, chunk)
        features.append(feature_vector)
    
    X = np.array(features)
    y = np.array(relevance_scores)
    
    # Scale features
    X_scaled = scoring_engine.feature_scaler.fit_transform(X)
    
    # Perform cross-validation
    cv_result = await scoring_engine._perform_cross_validation(X_scaled, y)
    
    print(f"✅ Cross-validation completed!")
    print(f"   CV folds: {cv_result['cv_folds']}")
    print(f"   R² scores: {[f'{score:.3f}' for score in cv_result['r2_scores']]}")
    print(f"   R² mean ± std: {cv_result['r2_mean']:.3f} ± {cv_result['r2_std']:.3f}")
    print(f"   MSE mean: {cv_result['mse_mean']:.3f}")
    print(f"   MAE mean: {cv_result['mae_mean']:.3f}")


async def main():
    """Main demonstration function."""
    print("🎯 RAGify ML Ensemble Training Demo")
    print("=" * 50)
    
    # Create temporary directory for model persistence
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize scoring engine
        config = OrchestratorConfig(
            model_name="demo-model",
            embedding_model="all-MiniLM-L6-v2"
        )
        scoring_engine = ContextScoringEngine(config)
        
        print(f"✅ Scoring engine initialized")
        print(f"   Available models: {list(scoring_engine.ml_models.keys())}")
        print(f"   Current model: {scoring_engine.current_model_name}")
        
        # Create training data
        training_data = await create_training_data()
        
        # Demonstrate various capabilities
        await demonstrate_basic_training(scoring_engine, training_data)
        await demonstrate_advanced_training(scoring_engine, training_data)
        await demonstrate_model_persistence(scoring_engine, temp_dir)
        await demonstrate_model_selection(scoring_engine, training_data)
        await demonstrate_model_retraining(scoring_engine, training_data)
        await demonstrate_model_info(scoring_engine)
        await demonstrate_feature_extraction(scoring_engine, training_data)
        await demonstrate_cross_validation(scoring_engine, training_data)
        
        print("\n🎉 ML Ensemble Training Demo Completed Successfully!")
        print("=" * 50)
        print("Key Features Demonstrated:")
        print("✅ Basic and advanced model training")
        print("✅ Cross-validation with multiple metrics")
        print("✅ Hyperparameter optimization")
        print("✅ Model persistence and loading")
        print("✅ Automatic model selection")
        print("✅ Model retraining")
        print("✅ Comprehensive feature extraction")
        print("✅ Real-time model information")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    asyncio.run(main())
