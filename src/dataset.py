
#Data loading and preprocessing for scikit-learn models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_ratings_data(ratings_path='data/cleaned/ratings_cleaned.csv'):
    #load csv
    print("Loading ratings data...")
    ratings_df = pd.read_csv(ratings_path)
    
    print(f"Total ratings: {len(ratings_df):,}")
    print(f"Unique users: {ratings_df['userId'].nunique():,}")
    print(f"Unique movies: {ratings_df['movieId'].nunique():,}")
    print(f"Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    
    return ratings_df


def create_user_item_matrix(ratings_df):
    print("\nCreating user-item matrix...")
    
    # save unique user id aNd movies 
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    #enumerate() assigns indexes starting from 0.
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    # map() replaces the original IDs with their new indexes in the DataFrame.
    user_indices = ratings_df['userId'].map(user_to_idx).values
    movie_indices = ratings_df['movieId'].map(movie_to_idx).values
    ratings = ratings_df['rating'].values
    
    # Create matrix
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    #stores actual ratings at the corresponding user and movie positions.
    #0 means the user hasn't rated that movie.
    ratings_matrix = np.zeros((n_users, n_movies))
    ratings_matrix[user_indices, movie_indices] = ratings
    #counts how many matrix values are non-zero (actual ratings)
    #sparsity is the percentage of empty (zero) positions in the matrix.
    print(f"Matrix shape: {ratings_matrix.shape}")
    print(f"Sparsity: {100 * (1 - np.count_nonzero(ratings_matrix) / ratings_matrix.size):.2f}%")
    return ratings_matrix, user_to_idx, movie_to_idx


def split_data(ratings_df, test_size=0.15, val_size=0.15, random_state=42):
    #15% for testing and 85% for training + validation.
    print("\nSplitting data...")
    
    # First split is train+val vs test
    train_val_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split is train vs val
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

    ratings_df = load_ratings_data(ratings_path)
    
    # split data
    train_df, val_df, test_df = split_data(ratings_df)
    
    # creates the user-item matrix using training data only.
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
