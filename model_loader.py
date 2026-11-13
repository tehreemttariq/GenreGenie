"""Load trained recommendation model"""
import pickle
import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")

def load_model():
    """Load the trained SVD model and mappings"""
    try:
        # Load SVD model (dict with embeddings)
        with open(MODEL_DIR / "svd_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Load mappings
        with open(MODEL_DIR / "mappings.json", "r") as f:
            mappings = json.load(f)
        
        print("✓ Model loaded successfully")
        return model_data, mappings
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_recommendations(model_data, mappings, user_id, n_recs=10):
    """Get top N recommendations for a user using dot product"""
    if model_data is None:
        return []
    
    try:
        user_idx = mappings.get("user_to_idx", {}).get(str(user_id))
        if user_idx is None:
            print(f"User {user_id} not found")
            return []
        
        # Get user embedding
        user_factors = model_data.get("user_factors")
        item_factors = model_data.get("item_factors")
        
        if user_factors is None or item_factors is None:
            return []
        
        user_embedding = user_factors[user_idx]
        
        # Compute scores with all items (dot product)
        scores = np.dot(item_factors, user_embedding)
        
        # Sort and get top N
        top_indices = np.argsort(-scores)[:n_recs]
        
        predictions = [(idx, scores[idx]) for idx in top_indices]
        return predictions
    
    except Exception as e:
        print(f"Error: {e}")
        return []
