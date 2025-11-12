"""
Run the 14 OFAT compliance matrix experiments.
"""

import os
import csv
import torch
import random
import numpy as np
from .train import run_experiment


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_to_csv(metrics_list, csv_file):
    """Append metrics to CSV file."""
    fieldnames = ['experiment', 'architecture', 'activation', 'seq_length', 'embedding_dim', 
                  'hidden_dim', 'num_layers', 'dropout', 'bidirectional', 'batch_size', 'epochs',
                  'learning_rate', 'optimizer', 'grad_clip', 'weight_decay', 'early_stopping',
                  'stopped_early', 'train_loss', 'test_loss', 'test_accuracy', 'test_precision',
                  'test_recall', 'test_f1', 'epoch_time_avg']
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metrics_list)


def main():
    """Run compliance matrix experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run compliance matrix experiments')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per experiment (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    metrics_file = 'results/metrics.csv'
    
    # Fixed constants for all experiments
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    EARLY_STOPPING = False
    
    # Default baseline configuration
    DEFAULT_SEQ_LENGTH = 50
    DEFAULT_OPTIMIZER = 'adam'
    DEFAULT_GRAD_CLIP = None
    
    print("\n" + "="*60)
    print("COMPLIANCE MATRIX EXPERIMENTS (14 runs)")
    print("="*60)
    print(f"Fixed constants: embedding={EMBEDDING_DIM}, hidden={HIDDEN_DIM}, "
          f"layers={NUM_LAYERS}, dropout={DROPOUT}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"Baseline: LSTM, L={DEFAULT_SEQ_LENGTH}, Adam, no clip")
    print("="*60)
    
    # Define all 14 experiments
    experiments = [
        # A. Baseline
        {
            'name': 'BASE',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        
        # B. Architecture (change model only)
        {
            'name': 'ARCH_RNN_TANH',
            'architecture': 'rnn',
            'activation': 'tanh',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        {
            'name': 'ARCH_BILSTM',
            'architecture': 'bilstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        
        # C. Activation sweep (RNN only; change activation only)
        {
            'name': 'RNN_RELU',
            'architecture': 'rnn',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        {
            'name': 'RNN_SIGMOID',
            'architecture': 'rnn',
            'activation': 'sigmoid',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        
        # D. Optimizer (change optimizer only)
        {
            'name': 'OPT_SGD',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': 'sgd',
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        {
            'name': 'OPT_RMSPROP',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': 'rmsprop',
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        
        # E. Sequence length (change L only)
        {
            'name': 'SEQ_25',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': 25,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        {
            'name': 'SEQ_100',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': 100,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': DEFAULT_GRAD_CLIP,
        },
        
        # F. Stability strategy (toggle clipping only)
        {
            'name': 'CLIP_ON',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': 1.0,
        },
        
        # G. Additional clipping variants (pairing with specific architectures/optimizers)
        {
            'name': 'ARCH_RNN_TANH_CLIP',
            'architecture': 'rnn',
            'activation': 'tanh',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': 1.0,
        },
        {
            'name': 'ARCH_BILSTM_CLIP',
            'architecture': 'bilstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': DEFAULT_OPTIMIZER,
            'grad_clip': 1.0,
        },
        {
            'name': 'OPT_SGD_CLIP',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': 'sgd',
            'grad_clip': 1.0,
        },
        {
            'name': 'OPT_RMSPROP_CLIP',
            'architecture': 'lstm',
            'activation': 'relu',
            'seq_length': DEFAULT_SEQ_LENGTH,
            'optimizer': 'rmsprop',
            'grad_clip': 1.0,
        },
    ]
    
    all_metrics = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/14: {exp_config['name']}")
        print(f"{'='*60}")
        
        # Determine bidirectional flag
        bidirectional = (exp_config['architecture'] == 'bilstm')
        
        metrics = run_experiment(
            experiment_name=exp_config['name'],
            seq_length=exp_config['seq_length'],
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=bidirectional,
            batch_size=BATCH_SIZE,
            epochs=args.epochs,
            learning_rate=LEARNING_RATE,
            optimizer_name=exp_config['optimizer'],
            grad_clip=exp_config['grad_clip'],
            weight_decay=WEIGHT_DECAY,
            early_stopping=EARLY_STOPPING,
            device=device,
            results_dir='results',
            plots_dir='results/plots',
            architecture=exp_config['architecture'],
            activation=exp_config['activation']
        )
        
        all_metrics.append(metrics)
        
        print(f"\n✅ {exp_config['name']} complete!")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   Test F1: {metrics['test_f1']:.4f}")
        print(f"   Test Loss: {metrics['test_loss']:.4f}")
    
    # Save all metrics to CSV
    append_to_csv(all_metrics, metrics_file)
    
    print(f"\n{'='*60}")
    print("ALL COMPLIANCE MATRIX EXPERIMENTS COMPLETE!")
    print(f"{'='*60}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Loss logs saved to: results/plots")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for metrics in all_metrics:
        print(f"{metrics['experiment']:20s} | Acc: {metrics['test_accuracy']:.4f} | "
              f"F1: {metrics['test_f1']:.4f} | Loss: {metrics['test_loss']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

