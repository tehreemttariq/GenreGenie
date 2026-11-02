"""
Training script for GenreGenie recommendation models using scikit-learn
Author: Member 1 (l242519 or l242590)
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from model import MatrixFactorizationSVD, CollaborativeFilteringKNN, CollaborativeFilteringBaseline
from dataset import prepare_data_for_training
from utils import save_model, save_training_history, set_seed


def evaluate_model(model, user_ids, movie_ids, true_ratings):
    """
    Evaluate model performance
    
    Returns:
        Dictionary with metrics
    """
    predictions = model.predict(user_ids, movie_ids)
    
    # Clip predictions to valid range [0.5, 5.0]
    predictions = np.clip(predictions, 0.5, 5.0)
    
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    mae = mean_absolute_error(true_ratings, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions
    }


def train_svd_model(train_matrix, val_data, test_data, n_factors=50, save_path=None):
    """
    Train SVD-based Matrix Factorization model
    
    Args:
        train_matrix: User-item ratings matrix
        val_data: Validation data dict
        test_data: Test data dict
        n_factors: Number of latent factors
        save_path: Path to save model
        
    Returns:
        model, results
    """
    print("="*60)
    print("TRAINING SVD MATRIX FACTORIZATION MODEL")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Latent factors: {n_factors}")
    print(f"  Matrix shape: {train_matrix.shape}")
    print(f"  Non-zero entries: {np.count_nonzero(train_matrix):,}")
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    
    model = MatrixFactorizationSVD(n_factors=n_factors)
    model.fit(train_matrix)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(
        model,
        val_data['user_ids'],
        val_data['movie_ids'],
        val_data['ratings']
    )
    
    print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"  Validation MAE:  {val_metrics['mae']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model,
        test_data['user_ids'],
        test_data['movie_ids'],
        test_data['ratings']
    )
    
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    
    # Save model
    if save_path:
        save_model(model, save_path)
    
    results = {
        'model_type': 'svd',
        'n_factors': n_factors,
        'training_time': training_time,
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae']
    }
    
    return model, results


def train_baseline_model(train_matrix, val_data, test_data, save_path=None):
    """
    Train Baseline Collaborative Filtering model
    
    Args:
        train_matrix: User-item ratings matrix
        val_data: Validation data dict
        test_data: Test data dict
        save_path: Path to save model
        
    Returns:
        model, results
    """
    print("="*60)
    print("TRAINING BASELINE COLLABORATIVE FILTERING MODEL")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Method: User + Item bias")
    print(f"  Matrix shape: {train_matrix.shape}")
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    
    model = CollaborativeFilteringBaseline()
    model.fit(train_matrix)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(
        model,
        val_data['user_ids'],
        val_data['movie_ids'],
        val_data['ratings']
    )
    
    print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"  Validation MAE:  {val_metrics['mae']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model,
        test_data['user_ids'],
        test_data['movie_ids'],
        test_data['ratings']
    )
    
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    
    # Save model
    if save_path:
        save_model(model, save_path)
    
    results = {
        'model_type': 'baseline',
        'training_time': training_time,
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae']
    }
    
    return model, results


def train_knn_model(train_matrix, val_data, test_data, n_neighbors=20, save_path=None):
    """
    Train KNN-based Collaborative Filtering model
    
    Args:
        train_matrix: User-item ratings matrix
        val_data: Validation data dict
        test_data: Test data dict
        n_neighbors: Number of neighbors
        save_path: Path to save model
        
    Returns:
        model, results
    """
    print("="*60)
    print("TRAINING KNN COLLABORATIVE FILTERING MODEL")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Neighbors: {n_neighbors}")
    print(f"  Matrix shape: {train_matrix.shape}")
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    
    model = CollaborativeFilteringKNN(n_neighbors=n_neighbors)
    model.fit(train_matrix)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(
        model,
        val_data['user_ids'],
        val_data['movie_ids'],
        val_data['ratings']
    )
    
    print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    print(f"  Validation MAE:  {val_metrics['mae']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model,
        test_data['user_ids'],
        test_data['movie_ids'],
        test_data['ratings']
    )
    
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    
    # Save model
    if save_path:
        save_model(model, save_path)
    
    results = {
        'model_type': 'knn',
        'n_neighbors': n_neighbors,
        'training_time': training_time,
        'val_rmse': val_metrics['rmse'],
        'val_mae': val_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae']
    }
    
    return model, results


def hyperparameter_search(train_matrix, val_data, test_data):
    """
    Search over different hyperparameters
    """
    print("="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60)
    
    # SVD configurations
    svd_configs = [
        {'n_factors': 20},
        {'n_factors': 50},
        {'n_factors': 100},
        {'n_factors': 150}
    ]
    
    results = []
    
    print("\nTesting SVD configurations...")
    for i, config in enumerate(svd_configs):
        print(f"\n[{i+1}/{len(svd_configs)}] Testing: {config}")
        
        model, result = train_svd_model(
            train_matrix, val_data, test_data,
            **config
        )
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_rmse')
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/hyperparameter_results.csv', index=False)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nBest configuration:")
    print(results_df.iloc[0].to_dict())
    
    return results_df


if __name__ == "__main__":
    # Set random seed
    set_seed(42)
    
    # Prepare data
    print("="*60)
    print("PREPARING DATA")
    print("="*60)
    train_matrix, val_data, test_data, mappings = prepare_data_for_training()
    
    # Save mappings for later use
    os.makedirs('models', exist_ok=True)
    with open('models/mappings.json', 'w') as f:
        # Convert to JSON-serializable format
        mappings_json = {
            'user_to_idx': {int(k): int(v) for k, v in mappings['user_to_idx'].items()},
            'movie_to_idx': {int(k): int(v) for k, v in mappings['movie_to_idx'].items()}
        }
        json.dump(mappings_json, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Train SVD model
    svd_model, svd_results = train_svd_model(
        train_matrix, val_data, test_data,
        n_factors=50,
        save_path='models/svd_model.pkl'
    )
    
    print("\n")
    
    # Train Baseline model (faster alternative to KNN)
    baseline_model, baseline_results = train_baseline_model(
        train_matrix, val_data, test_data,
        save_path='models/baseline_model.pkl'
    )
    
    # Save combined results
    all_results = {
        'svd': svd_results,
        'baseline': baseline_results
    }
    
    save_training_history(all_results, 'results/training_results.json')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    print("  - models/svd_model.pkl")
    print("  - models/baseline_model.pkl")
    print("  - models/mappings.json")
    print("\nResults saved:")
    print("  - results/training_results.json")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame([svd_results, baseline_results])
    print(comparison_df[['model_type', 'val_rmse', 'val_mae', 'test_rmse', 'test_mae']].to_string(index=False))
    
    # Optional: Run hyperparameter search
    # Uncomment to run:
    # print("\n")
    # hyperparameter_search(train_matrix, val_data, test_data)
