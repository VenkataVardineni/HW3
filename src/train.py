"""
Training script for LSTM sentiment analysis model.
Runs experiments varying one factor at a time.
"""

import os
import csv
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .models import create_model
from .utils import load_processed_data, create_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None, verbose=False):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping value (None if no clipping)
        verbose: Whether to show progress bar
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    loader = tqdm(train_loader, desc="Training", leave=False) if verbose else train_loader
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, test_loader, criterion, device, verbose=False):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with metrics: loss, accuracy, precision, recall, f1
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    loader = tqdm(test_loader, desc="Evaluating", leave=False) if verbose else test_loader
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Get predictions
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            labels = y_batch.cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / num_batches,
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0)
    }
    
    return metrics


def run_experiment(experiment_name, seq_length, embedding_dim, hidden_dim, num_layers,
                   dropout, bidirectional, batch_size, epochs, learning_rate, optimizer_name,
                   grad_clip, weight_decay=0.0, early_stopping=False, early_stop_patience=3,
                   early_stop_min_delta=0.001, device=None, results_dir='results', plots_dir='results/plots',
                   architecture='lstm', activation='tanh'):
    """
    Run a single experiment.
    
    Args:
        experiment_name: Name of the experiment
        seq_length: Sequence length
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        optimizer_name: Optimizer name ('adam' or 'sgd')
        grad_clip: Gradient clipping value (None if no clipping)
        weight_decay: Weight decay for optimizer (default: 0.0)
        early_stopping: Whether to use early stopping (default: False)
        early_stop_patience: Patience for early stopping (default: 3)
        early_stop_min_delta: Minimum delta for early stopping (default: 0.001)
        device: Device to train on
        results_dir: Directory to save results
        plots_dir: Directory to save loss plots
        
    Returns:
        Dictionary with final metrics including epoch_time
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Load processed data
    data = load_processed_data(seq_length)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    vocab_size = data['vocab_size']
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=batch_size
    )
    
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        architecture=architecture,
        activation=activation
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training history
    train_losses = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    
    # Early stopping variables
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    stopped_early = False
    best_model_state = None
    
    # Training loop
    best_test_acc = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip, verbose=True)
        train_losses.append(train_loss)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device, verbose=True)
        test_losses.append(test_metrics['loss'])
        test_accuracies.append(test_metrics['accuracy'])
        
        # Track epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
        
        # Early stopping check - monitor test loss (validation loss)
        if early_stopping:
            # Check if this is an improvement
            improved = (best_test_loss - test_metrics['loss']) > early_stop_min_delta
            
            if improved:
                best_test_loss = test_metrics['loss']
                epochs_without_improvement = 0
                # Save best model state with proper deep copy
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  → New best test loss: {best_test_loss:.4f} (saving model state)")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= early_stop_patience:
                stopped_early = True
                print(f"\nEarly stopping triggered at epoch {epoch + 1} (patience: {early_stop_patience})")
                break
        else:
            # If not using early stopping, still track best for consistency
            if test_metrics['loss'] < best_test_loss:
                best_test_loss = test_metrics['loss']
                # Save best model state even without early stopping
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f} | "
              f"Best Acc: {best_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    # Always restore the best model state (whether early stopping triggered or not)
    if best_model_state is not None:
        print(f"\nRestoring best model state (best test loss: {best_test_loss:.4f})...")
        model.load_state_dict(best_model_state)
    
    # Final evaluation with best weights
    print("\nRunning final evaluation with best model weights...")
    final_metrics = evaluate(model, test_loader, criterion, device, verbose=True)
    final_train_loss = train_losses[-1]
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0.0
    
    # Save loss log
    os.makedirs(plots_dir, exist_ok=True)
    log_file = os.path.join(plots_dir, f"{experiment_name}_loss_log.txt")
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Test_Loss,Test_Accuracy\n")
        # Only write epochs that were actually run (in case of early stopping)
        num_epochs_run = len(train_losses)
        for epoch in range(num_epochs_run):
            f.write(f"{epoch+1},{train_losses[epoch]:.6f},{test_losses[epoch]:.6f},{test_accuracies[epoch]:.6f}\n")
    
    print(f"Final - Train Loss: {final_train_loss:.4f}, Test Loss: {final_metrics['loss']:.4f}, "
          f"Test Acc: {final_metrics['accuracy']:.4f}, Test F1: {final_metrics['f1']:.4f}")
    
    # Return metrics for CSV
    return {
        'experiment': experiment_name,
        'architecture': architecture,
        'activation': activation if architecture == 'rnn' else 'N/A',
        'seq_length': seq_length,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'bidirectional': bidirectional,
        'batch_size': batch_size,
        'epochs': len(epoch_times),  # Actual epochs run (may be less if early stopping)
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'grad_clip': grad_clip if grad_clip is not None else 'None',
        'weight_decay': weight_decay,
        'early_stopping': early_stopping,
        'stopped_early': stopped_early,
        'train_loss': final_train_loss,
        'test_loss': final_metrics['loss'],
        'test_accuracy': final_metrics['accuracy'],
        'test_precision': final_metrics['precision'],
        'test_recall': final_metrics['recall'],
        'test_f1': final_metrics['f1'],
        'epoch_time_avg': avg_epoch_time
    }


def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(description='Train LSTM sentiment analysis models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per experiment (default: 10)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--plots_dir', type=str, default='results/plots',
                        help='Directory to save loss plots (default: results/plots)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Baseline configuration
    baseline = {
        'seq_length': 50,
        'embedding_dim': 100,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'bidirectional': False,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'grad_clip': None
    }
    
    # Experiment configurations (varying one factor at a time)
    experiments = []
    
    # 1. Baseline
    experiments.append({
        'name': 'baseline',
        **baseline
    })
    
    # 2. Vary architecture: bidirectional
    experiments.append({
        'name': 'bidirectional',
        **baseline,
        'bidirectional': True
    })
    
    # 3. Vary optimizer: SGD
    experiments.append({
        'name': 'optimizer_sgd',
        **baseline,
        'optimizer': 'sgd'
    })
    
    # 4. Vary sequence length: 25
    experiments.append({
        'name': 'seq_length_25',
        **baseline,
        'seq_length': 25
    })
    
    # 5. Vary sequence length: 100
    experiments.append({
        'name': 'seq_length_100',
        **baseline,
        'seq_length': 100
    })
    
    # 6. Vary gradient clipping: 1.0
    experiments.append({
        'name': 'grad_clip_1.0',
        **baseline,
        'grad_clip': 1.0
    })
    
    # 7. Vary gradient clipping: 5.0
    experiments.append({
        'name': 'grad_clip_5.0',
        **baseline,
        'grad_clip': 5.0
    })
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # CSV file for metrics
    metrics_file = os.path.join(args.results_dir, 'metrics.csv')
    
    # Run experiments
    all_metrics = []
    total_experiments = len(experiments)
    
    print(f"\n{'='*60}")
    print(f"Starting {total_experiments} experiments")
    print(f"{'='*60}\n")
    
    exp_pbar = tqdm(enumerate(experiments), total=total_experiments, desc="Experiments", unit="exp")
    
    for exp_idx, exp_config in exp_pbar:
        exp_pbar.set_description(f"Experiment {exp_idx+1}/{total_experiments}: {exp_config['name']}")
        
        metrics = run_experiment(
            experiment_name=exp_config['name'],
            seq_length=exp_config['seq_length'],
            embedding_dim=exp_config['embedding_dim'],
            hidden_dim=exp_config['hidden_dim'],
            num_layers=exp_config['num_layers'],
            dropout=exp_config['dropout'],
            bidirectional=exp_config['bidirectional'],
            batch_size=exp_config['batch_size'],
            epochs=args.epochs,
            learning_rate=exp_config['learning_rate'],
            optimizer_name=exp_config['optimizer'],
            grad_clip=exp_config['grad_clip'],
            device=device,
            results_dir=args.results_dir,
            plots_dir=args.plots_dir
        )
        all_metrics.append(metrics)
        
        # Update experiment progress bar
        exp_pbar.set_postfix({
            'Acc': f"{metrics['test_accuracy']:.4f}",
            'F1': f"{metrics['test_f1']:.4f}"
        })
    
    exp_pbar.close()
    
    # Save metrics to CSV
    fieldnames = ['experiment', 'seq_length', 'embedding_dim', 'hidden_dim', 'num_layers',
                  'dropout', 'bidirectional', 'batch_size', 'epochs', 'learning_rate',
                  'optimizer', 'grad_clip', 'weight_decay', 'early_stopping', 'stopped_early',
                  'train_loss', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall',
                  'test_f1', 'epoch_time_avg']
    
    file_exists = os.path.exists(metrics_file)
    with open(metrics_file, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Loss logs saved to: {args.plots_dir}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nExperiment Summary:")
    print("-" * 60)
    for metrics in all_metrics:
        print(f"{metrics['experiment']:20s} | Acc: {metrics['test_accuracy']:.4f} | "
              f"F1: {metrics['test_f1']:.4f} | Loss: {metrics['test_loss']:.4f}")


if __name__ == '__main__':
    main()
