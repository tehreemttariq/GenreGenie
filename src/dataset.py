"""
Dataset loading and preprocessing for PyTorch
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MovieRatingsDataset(Dataset):
    """
    PyTorch Dataset for movie ratings
    """
    
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


def load_and_prepare_data(ratings_path='data/cleaned/ratings_cleaned.csv'):
    """
    Load ratings data and prepare for training
    
    Returns:
        train_loader, val_loader, test_loader, n_users, n_movies
    """
    print("Loading ratings data...")
    
    # Load ratings
    ratings_df = pd.read_csv(ratings_path)
    
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Users: {ratings_df['userId'].nunique()}")
    print(f"Movies: {ratings_df['movieId'].nunique()}")
    
    # Create user and movie ID mappings (0-indexed for PyTorch)
    user_ids = ratings_df['userId'].unique()
    movie_ids = ratings_df['movieId'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    
    # Map to indices
    ratings_df['user_idx'] = ratings_df['userId'].map(user_to_idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_to_idx)
    
    # Extract arrays
    users = ratings_df['user_idx'].values
    movies = ratings_df['movie_idx'].values
    ratings = ratings_df['rating'].values
    
    # Split data: 70% train, 15% validation, 15% test
    X = np.column_stack([users, movies])
    y = ratings
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(ratings_df)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(ratings_df)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(ratings_df)*100:.1f}%)")
    
    # Create datasets
    train_dataset = MovieRatingsDataset(X_train[:, 0], X_train[:, 1], y_train)
    val_dataset = MovieRatingsDataset(X_val[:, 0], X_val[:, 1], y_val)
    test_dataset = MovieRatingsDataset(X_test[:, 0], X_test[:, 1], y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    n_users = len(user_ids)
    n_movies = len(movie_ids)
    
    # Save mappings for later use
    mappings = {
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx
    }
    
    return train_loader, val_loader, test_loader, n_users, n_movies, mappings


def create_dataloaders(users, movies, ratings, batch_size=256):
    """
    Create DataLoader from arrays
    """
    dataset = MovieRatingsDataset(users, movies, ratings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
