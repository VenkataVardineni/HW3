"""
Evaluation script for LSTM sentiment analysis model.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .models import create_model
from .utils import load_processed_data, create_data_loaders, load_model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary containing metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.numpy()
            
            # Forward pass
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_predictions.extend(predictions.flatten())
            all_labels.extend(y_batch)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate LSTM sentiment analysis model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file')
    parser.add_argument('--seq_length', type=int, default=50, choices=[25, 50, 100],
                        help='Sequence length (default: 50)')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Embedding dimension (default: 100)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (default: 128)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional LSTM')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    print(f"Loading processed data for sequence length {args.seq_length}...")
    data = load_processed_data(args.seq_length)
    
    X_test = data['X_test']
    y_test = data['y_test']
    vocab_size = data['vocab_size']
    
    print(f"Test samples: {len(X_test)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create test data loader
    _, test_loader = create_data_loaders(
        X_test, y_test, X_test, y_test, batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    model = model.to(device)
    
    # Load model weights
    print(f"Loading model from {args.model_path}...")
    model = load_model(model, args.model_path)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*50)


if __name__ == '__main__':
    main()

