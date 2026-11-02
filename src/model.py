"""
GenreGenie Recommendation Models using Scikit-Learn
Simple implementations using SVD and KNN
Author: Member 1 (l242519 or l242590)
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


class MatrixFactorizationSVD:
    """
    Matrix Factorization using Singular Value Decomposition (SVD)
    Simple and effective collaborative filtering
    """
    
    def __init__(self, n_factors=50):
        """
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_means = None
        self.item_means = None
    
    def fit(self, ratings_matrix):
        """
        Fit the model on user-item ratings matrix
        
        Args:
            ratings_matrix: numpy array of shape (n_users, n_items)
        """
        # Store global mean
        self.global_mean = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Center the ratings
        centered_matrix = ratings_matrix.copy()
        mask = ratings_matrix > 0
        centered_matrix[mask] -= self.global_mean
        
        # Apply SVD
        self.user_factors = self.svd.fit_transform(centered_matrix)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for user-item pairs
        
        Args:
            user_ids: array of user indices
            item_ids: array of item indices
            
        Returns:
            Predicted ratings
        """
        # Ensure indices are integers
        user_ids = np.array(user_ids, dtype=int)
        item_ids = np.array(item_ids, dtype=int)
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            pred = self.global_mean + np.dot(
                self.user_factors[user_id], 
                self.item_factors[item_id]
            )
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save(self, path):
        """Save model to disk"""
        model_data = {
            'n_factors': self.n_factors,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'global_mean': self.global_mean,
            'svd': self.svd
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n_factors=model_data['n_factors'])
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.global_mean = model_data['global_mean']
        model.svd = model_data['svd']
        
        return model


class CollaborativeFilteringBaseline:
    """
    Simple Baseline Collaborative Filtering
    Uses user and item averages for predictions
    """
    
    def __init__(self):
        """
        Simple baseline predictor
        """
        self.global_mean = None
        self.user_means = None
        self.item_means = None
        self.ratings_matrix = None
    
    def fit(self, ratings_matrix):
        """
        Fit the model on user-item ratings matrix
        
        Args:
            ratings_matrix: numpy array of shape (n_users, n_items)
        """
        self.ratings_matrix = ratings_matrix
        
        # Calculate global mean (only non-zero ratings)
        mask = ratings_matrix > 0
        self.global_mean = np.mean(ratings_matrix[mask])
        
        # Calculate user means
        self.user_means = np.zeros(ratings_matrix.shape[0])
        for i in range(ratings_matrix.shape[0]):
            user_ratings = ratings_matrix[i, ratings_matrix[i] > 0]
            if len(user_ratings) > 0:
                self.user_means[i] = np.mean(user_ratings)
            else:
                self.user_means[i] = self.global_mean
        
        # Calculate item means
        self.item_means = np.zeros(ratings_matrix.shape[1])
        for j in range(ratings_matrix.shape[1]):
            item_ratings = ratings_matrix[ratings_matrix[:, j] > 0, j]
            if len(item_ratings) > 0:
                self.item_means[j] = np.mean(item_ratings)
            else:
                self.item_means[j] = self.global_mean
        
        return self
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings using baseline approach
        
        Args:
            user_ids: array of user indices
            item_ids: array of item indices
            
        Returns:
            Predicted ratings
        """
        # Ensure indices are integers
        user_ids = np.array(user_ids, dtype=int)
        item_ids = np.array(item_ids, dtype=int)
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Baseline: global_mean + user_bias + item_bias
            user_bias = self.user_means[user_id] - self.global_mean
            item_bias = self.item_means[item_id] - self.global_mean
            
            pred = self.global_mean + user_bias + item_bias
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save(self, path):
        """Save model to disk"""
        model_data = {
            'global_mean': self.global_mean,
            'user_means': self.user_means,
            'item_means': self.item_means,
            'ratings_matrix': self.ratings_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.global_mean = model_data['global_mean']
        model.user_means = model_data['user_means']
        model.item_means = model_data['item_means']
        model.ratings_matrix = model_data['ratings_matrix']
        
        return model


class CollaborativeFilteringKNN:
    """
    K-Nearest Neighbors Collaborative Filtering (DEPRECATED - TOO SLOW)
    Item-based recommendation using cosine similarity
    """
    
    def __init__(self, n_neighbors=20):
        """
        Args:
            n_neighbors: Number of similar items to consider
        """
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric='cosine',
            algorithm='brute'
        )
        self.ratings_matrix = None
        self.global_mean = None
    
    def fit(self, ratings_matrix):
        """
        Fit the model on user-item ratings matrix
        
        Args:
            ratings_matrix: numpy array of shape (n_users, n_items)
        """
        self.ratings_matrix = ratings_matrix
        self.global_mean = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Fit KNN on item vectors (transpose to get items as rows)
        self.knn.fit(ratings_matrix.T)
        
        return self
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings using KNN
        
        Args:
            user_ids: array of user indices
            item_ids: array of item indices
            
        Returns:
            Predicted ratings
        """
        # Ensure indices are integers
        user_ids = np.array(user_ids, dtype=int)
        item_ids = np.array(item_ids, dtype=int)
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Get user's ratings
            user_ratings = self.ratings_matrix[user_id]
            
            # Get the item vector (all user ratings for this item)
            item_vector = self.ratings_matrix[:, item_id].reshape(1, -1)
            
            # Find similar items
            distances, indices = self.knn.kneighbors(item_vector)
            
            # Calculate weighted average
            similar_items = indices[0]
            weights = 1 - distances[0]  # Convert distance to similarity
            
            # Get ratings for similar items
            rated_mask = user_ratings[similar_items] > 0
            
            if rated_mask.sum() > 0:
                weighted_sum = np.sum(
                    user_ratings[similar_items][rated_mask] * weights[rated_mask]
                )
                weight_sum = np.sum(weights[rated_mask])
                pred = weighted_sum / weight_sum if weight_sum > 0 else self.global_mean
            else:
                pred = self.global_mean
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save(self, path):
        """Save model to disk"""
        model_data = {
            'n_neighbors': self.n_neighbors,
            'knn': self.knn,
            'ratings_matrix': self.ratings_matrix,
            'global_mean': self.global_mean
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n_neighbors=model_data['n_neighbors'])
        model.knn = model_data['knn']
        model.ratings_matrix = model_data['ratings_matrix']
        model.global_mean = model_data['global_mean']
        
        return model


def get_model(model_type='svd', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'svd' or 'baseline'
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    if model_type == 'svd':
        return MatrixFactorizationSVD(
            n_factors=kwargs.get('n_factors', 50)
        )
    elif model_type == 'baseline':
        return CollaborativeFilteringBaseline()
    elif model_type == 'knn':
        return CollaborativeFilteringKNN(
            n_neighbors=kwargs.get('n_neighbors', 20)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
