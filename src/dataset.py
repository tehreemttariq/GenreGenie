"""
Data loading and preprocessing for scikit-learn models
Author: Member 1 (l242519 or l242590)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_ratings_data(ratings_path='data/cleaned/ratings_cleaned.csv'):
    """
    Load ratings data from CSV
    
    Returns:
        DataFrame with ratings
    """
    print("Loading ratings data...")
    ratings_df = pd.read_csv(ratings_path)
    
    print(f"Total ratings: {len(ratings_df):,}")
    print(f"Unique users: {ratings_df['userId'].nunique():,}")
    print(f"Unique movies: {ratings_df['movieId'].nunique():,}")
    print(f"Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    
    return ratings_df


def create_user_item_matrix(ratings_df):
    """
    Create user-item ratings matrix
    
    Returns:
        ratings_matrix, user_mapping, movie_mapping
    """
    print("\nCreating user-item matrix...")
    
    # Create mappings
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    # Map to indices
    user_indices = ratings_df['userId'].map(user_to_idx).values
    movie_indices = ratings_df['movieId'].map(movie_to_idx).values
    ratings = ratings_df['rating'].values
    
    # Create matrix
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    ratings_matrix = np.zeros((n_users, n_movies))
    ratings_matrix[user_indices, movie_indices] = ratings
    
    print(f"Matrix shape: {ratings_matrix.shape}")
    print(f"Sparsity: {100 * (1 - np.count_nonzero(ratings_matrix) / ratings_matrix.size):.2f}%")
    
    return ratings_matrix, user_to_idx, movie_to_idx


def split_data(ratings_df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        ratings_df: DataFrame with ratings
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    print("\nSplitting data...")
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state
    )
    
    print(f"Train: {len(train_df):,} ({len(train_df)/len(ratings_df)*100:.1f}%)")
    print(f"Val:   {len(val_df):,} ({len(val_df)/len(ratings_df)*100:.1f}%)")
    print(f"Test:  {len(test_df):,} ({len(test_df)/len(ratings_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def prepare_data_for_training(ratings_path='data/cleaned/ratings_cleaned.csv'):
    """
    Complete data preparation pipeline
    
    Returns:
        train_matrix, val_data, test_data, mappings
    """
    # Load data
    ratings_df = load_ratings_data(ratings_path)
    
    # Split data
    train_df, val_df, test_df = split_data(ratings_df)
    
    # Create user-item matrix from training data
    train_matrix, user_to_idx, movie_to_idx = create_user_item_matrix(train_df)
    
    # Prepare validation and test data - only keep items in training set
    def filter_data(df):
        """Filter to only include users and movies seen in training"""
        df = df[df['userId'].isin(user_to_idx.keys())]
        df = df[df['movieId'].isin(movie_to_idx.keys())]
        
        return {
            'user_ids': df['userId'].map(user_to_idx).values,
            'movie_ids': df['movieId'].map(movie_to_idx).values,
            'ratings': df['rating'].values
        }
    
    val_data = filter_data(val_df)
    test_data = filter_data(test_df)
    
    print(f"\nFiltered validation: {len(val_data['ratings'])} samples")
    print(f"Filtered test: {len(test_data['ratings'])} samples")
    
    mappings = {
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx,
        'idx_to_user': {v: k for k, v in user_to_idx.items()},
        'idx_to_movie': {v: k for k, v in movie_to_idx.items()}
    }
    
    return train_matrix, val_data, test_data, mappings
