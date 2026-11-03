
#Utility functions for training and saving models

import random
import numpy as np
import json
import os
import pickle


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def save_model(model, path):
    #save model to disk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✓ Model saved to {path}")


def load_model(path):
    #load model from disk
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {path}")
    return model


def save_training_history(results, path='results/training_results.json'):
    #save to json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {path}")


def load_training_history(path='results/training_results.json'):
    #load results from json
    with open(path, 'r') as f:
        results = json.load(f)
    return results
