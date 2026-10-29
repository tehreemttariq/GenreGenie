"""
GenreGenie Recommendation Models
Simple implementations of Matrix Factorization and Neural Collaborative Filtering
"""

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Simple Matrix Factorization for movie recommendations
    Learns user and movie embeddings
    """
    
    def __init__(self, n_users, n_movies, n_factors=50, dropout=0.2):
        super(MatrixFactorization, self).__init__()
        
        # User and movie embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Apply dropout
        user_emb = self.dropout(user_emb)
        movie_emb = self.dropout(movie_emb)
        
        # Calculate prediction
        dot_product = (user_emb * movie_emb).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        movie_b = self.movie_bias(movie_ids).squeeze()
        
        prediction = self.global_bias + user_b + movie_b + dot_product
        
        return prediction


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF)
    Combines embeddings with neural network layers
    """
    
    def __init__(self, n_users, n_movies, n_factors=50, layers=[64, 32], dropout=0.2):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        
        # Build MLP layers
        mlp_layers = []
        input_size = n_factors * 2
        
        for hidden_size in layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        # Add final output layer
        mlp_layers.append(nn.Linear(input_size, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, movie_emb], dim=1)
        
        # Pass through MLP
        prediction = self.mlp(x).squeeze()
        
        return prediction
