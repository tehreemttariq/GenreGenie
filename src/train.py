"""
Training script for GenreGenie recommendation models
Author: Member 1 (l242519 or l242590)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from model import MatrixFactorization, NeuralCollaborativeFiltering
from dataset import load_and_prepare_data
from utils import set_seed, save_model, save_training_history, get_device, count_parameters


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    
    for user_ids, movie_ids, ratings in train_loader:
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        ratings = ratings.to(device)
        
        # Forward pass
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for user_ids, movie_ids, ratings in val_loader:
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)
            
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model_type='mf', n_epochs=30, learning_rate=0.001, n_factors=50, 
                dropout=0.2, save_path=None):
    """
    Main training function
    
    Args:
        model_type: 'mf' or 'ncf'
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        n_factors: Embedding dimension
        dropout: Dropout rate
        save_path: Path to save best model
    """
    print("="*60)
    print(f"Training {model_type.upper()} Model")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Load data
    train_loader, val_loader, test_loader, n_users, n_movies, mappings = load_and_prepare_data()
    
    print(f"\nModel configuration:")
    print(f"  Users: {n_users}")
    print(f"  Movies: {n_movies}")
    print(f"  Embedding dim: {n_factors}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create model
    if model_type == 'mf':
        model = MatrixFactorization(n_users, n_movies, n_factors, dropout)
    elif model_type == 'ncf':
        model = NeuralCollaborativeFiltering(n_users, n_movies, n_factors, [64, 32], dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    print(f"\nTrainable parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'config': {
            'model_type': model_type,
            'n_epochs': n_epochs,
            'learning_rate': learning_rate,
            'n_factors': n_factors,
            'dropout': dropout,
            'batch_size': 256
        }
    }
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("-"*60)
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                save_model(model, save_path, epoch, optimizer, val_loss)
    
    print("-"*60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save training history
    save_training_history(history)
    
    return model, history


def hyperparameter_search():
    """
    Search over different hyperparameters
    """
    print("="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60)
    
    # Define hyperparameter grid
    configs = [
        {'learning_rate': 0.001, 'n_factors': 32, 'dropout': 0.1},
        {'learning_rate': 0.001, 'n_factors': 50, 'dropout': 0.2},
        {'learning_rate': 0.005, 'n_factors': 50, 'dropout': 0.2},
        {'learning_rate': 0.001, 'n_factors': 64, 'dropout': 0.2},
        {'learning_rate': 0.001, 'n_factors': 50, 'dropout': 0.3},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing config: {config}")
        
        model, history = train_model(
            model_type='mf',
            n_epochs=20,
            **config
        )
        
        best_val_loss = min(history['val_loss'])
        
        results.append({
            **config,
            'best_val_loss': best_val_loss
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_loss')
    results_df.to_csv('results/hyperparameter_results.csv', index=False)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print(f"\nBest config: {results_df.iloc[0].to_dict()}")
    
    return results_df


if __name__ == "__main__":
    # Train Matrix Factorization
    print("\n" + "="*60)
    print("TRAINING MATRIX FACTORIZATION MODEL")
    print("="*60 + "\n")
    
    mf_model, mf_history = train_model(
        model_type='mf',
        n_epochs=30,
        learning_rate=0.001,
        n_factors=50,
        dropout=0.2,
        save_path='models/matrix_factorization_best.pth'
    )
    
    print("\n" + "="*60)
    print("TRAINING NEURAL COLLABORATIVE FILTERING MODEL")
    print("="*60 + "\n")
    
    # Train Neural CF
    ncf_model, ncf_history = train_model(
        model_type='ncf',
        n_epochs=30,
        learning_rate=0.001,
        n_factors=50,
        dropout=0.2,
        save_path='models/neural_cf_best.pth'
    )
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    print("  - models/matrix_factorization_best.pth")
    print("  - models/neural_cf_best.pth")
    print("\nTraining history saved:")
    print("  - results/training_history.json")
    
    # Optional: Run hyperparameter search
    # Uncomment to run:
    # print("\n")
    # hyperparameter_search()
