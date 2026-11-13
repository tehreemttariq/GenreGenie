"""Fold-in new user vector from their ratings"""
import numpy as np
from model_loader import load_model

def fold_in_user(ratings_dict, model_data, mappings, lambda_reg=1.0):
    """
    Compute latent vector for new user using ridge regression.
    
    Args:
        ratings_dict: {movie_id: rating, ...} from database
        model_data: loaded SVD model (has item_factors, global_mean)
        mappings: {user_to_idx, movie_to_idx}
        lambda_reg: regularization strength (default 1.0)
    
    Returns:
        user_vector: (50,) numpy array, or None if error
    """
    if not ratings_dict or len(ratings_dict) < 1:
        return None
    
    try:
        # Get item indices for rated movies
        movie_ids = list(ratings_dict.keys())
        ratings = np.array(list(ratings_dict.values()), dtype=np.float64)
        
        # Map movie IDs to indices
        movie_to_idx = mappings['movie_to_idx']
        movie_indices = []
        for mid in movie_ids:
            idx = movie_to_idx.get(str(mid))
            if idx is None:
                continue
            movie_indices.append(idx)
        
        if len(movie_indices) == 0:
            return None
        
        movie_indices = np.array(movie_indices)
        
        # Get item factors for rated movies
        item_factors = model_data['item_factors']  # (8607, 50)
        X = item_factors[movie_indices]  # (k, 50)
        
        # Center ratings by global mean
        global_mean = model_data['global_mean']
        y = ratings[:len(movie_indices)] - global_mean  # (k,)
        
        # Ridge regression: (X^T X + lambda*I)^{-1} X^T y
        n_factors = X.shape[1]  # 50
        A = X.T @ X + lambda_reg * np.eye(n_factors)
        b = X.T @ y
        
        # Solve linear system
        user_vector = np.linalg.solve(A, b)
        
        return user_vector
    
    except Exception as e:
        print(f"Error in fold_in_user: {e}")
        return None

def recommend_new_user(user_vector, rated_movie_ids, model_data, mappings, n_recs=10):
    """
    Get top N recommendations for new user using their folded-in vector.
    
    Args:
        user_vector: (50,) from fold_in_user()
        rated_movie_ids: set/list of movie IDs already rated (to exclude)
        model_data: loaded SVD model
        mappings: user_to_idx, movie_to_idx
        n_recs: number of recommendations
    
    Returns:
        list of (movie_id, score) tuples, sorted by score descending
    """
    if user_vector is None:
        return []
    
    try:
        item_factors = model_data['item_factors']
        global_mean = model_data['global_mean']
        
        # Score all movies
        scores = global_mean + user_vector @ item_factors.T  # (8607,)
        
        # Get indices of rated movies
        movie_to_idx = mappings['movie_to_idx']
        rated_indices = set()
        for mid in rated_movie_ids:
            idx = movie_to_idx.get(str(mid))
            if idx is not None:
                rated_indices.add(idx)
        
        # Get top recommendations (excluding rated)
        idx_to_movie = {v: k for k, v in movie_to_idx.items()}
        
        top_indices = np.argsort(-scores)
        recommendations = []
        
        for idx in top_indices:
            if idx not in rated_indices and len(recommendations) < n_recs:
                movie_id = int(idx_to_movie[idx])
                recommendations.append((movie_id, float(scores[idx])))
        
        return recommendations
    
    except Exception as e:
        print(f"Error in recommend_new_user: {e}")
        return []

def recommend_cold_start(favorite_genres, model_data, mappings, n_recs=10):
    """
    Cold-start recommendations for users with < 5 ratings.
    Show popular movies in their favorite genres.
    
    Args:
        favorite_genres: list of genre strings
        model_data: loaded SVD model
        mappings: movie_to_idx
        n_recs: number to return
    
    Returns:
        list of (movie_id, popularity_score) tuples
    """
    # Simple fallback: return top-scored movies overall
    # In production, would filter by genre from metadata
    item_factors = model_data['item_factors']
    global_mean = model_data['global_mean']
    
    # Use average item factor as "global" recommendation
    avg_factor = np.mean(item_factors, axis=0)
    scores = global_mean + avg_factor @ item_factors.T
    
    idx_to_movie = {v: k for k, v in mappings['movie_to_idx'].items()}
    top_indices = np.argsort(-scores)[:n_recs]
    
    recommendations = [(int(idx_to_movie[idx]), float(scores[idx])) for idx in top_indices]
    return recommendations
