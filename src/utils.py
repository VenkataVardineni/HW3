"""
Utility functions for training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IMDBDataset(Dataset):
    """
    Dataset class for IMDb sentiment analysis.
    """
    
    def __init__(self, X, y):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (numpy array)
            y: Labels (numpy array)
        """
        self.X = torch.LongTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Create data loaders for training and testing.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_test: Test sequences
        y_test: Test labels
        batch_size: Batch size (default: 32)
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    train_dataset = IMDBDataset(X_train, y_train)
    test_dataset = IMDBDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def load_processed_data(seq_length):
    """
    Load preprocessed data for a specific sequence length.
    
    Args:
        seq_length: Sequence length (25, 50, or 100)
        
    Returns:
        Dictionary containing processed data
    """
    import pickle
    import os
    
    data_path = f'data/processed/imdb_processed_seqlen_{seq_length}.pkl'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. "
            "Please run preprocessing first: python -m src.preprocess"
        )
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_model(model, path):
    """
    Save model to file.
    
    Args:
        model: PyTorch model
        path: Path to save model
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    """
    Load model from file.
    
    Args:
        model: PyTorch model instance
        path: Path to load model from
        
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

