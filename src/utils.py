"""
Utility functions for training and saving models
"""

import torch
import random
import numpy as np
import json
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_model(model, path, epoch, optimizer, loss):
    """
    Save model checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def save_training_history(history, path='results/training_history.json'):
    """
    Save training history to JSON
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to {path}")


def get_device():
    """
    Get device (GPU if available, else CPU)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def count_parameters(model):
    """
    Count trainable parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
