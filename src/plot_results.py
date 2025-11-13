"""
Generate plots from experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_variant_comparison(df, output_dir='results/plots'):
    """
    Create variant comparison bar chart.
    Shows: BASE, RNN {tanh/relu/sigmoid}, BiLSTM, OPT {SGD/RMSProp}, CLIP_ON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select the variants to compare
    variants = {
        'BASE': 'BASE',
        'RNN (tanh)': 'ARCH_RNN_TANH',
        'RNN (relu)': 'RNN_RELU',
        'RNN (sigmoid)': 'RNN_SIGMOID',
        'BiLSTM': 'ARCH_BILSTM',
        'SGD': 'OPT_SGD',
        'RMSProp': 'OPT_RMSPROP',
        'CLIP_ON': 'CLIP_ON'
    }
    
    # Extract data for these variants
    variant_data = []
    for label, exp_name in variants.items():
        row = df[df['experiment'] == exp_name]
        if not row.empty:
            variant_data.append({
                'Variant': label,
                'Accuracy': row['test_accuracy'].values[0],
                'F1': row['test_f1'].values[0]
            })
    
    variant_df = pd.DataFrame(variant_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(variant_df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], variant_df['Accuracy'], width, 
                   label='Accuracy', alpha=0.8, color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], variant_df['F1'], width, 
                   label='F1 Score', alpha=0.8, color='coral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Variant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Variant Comparison: Accuracy and F1 Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_df['Variant'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(variant_df[['Accuracy', 'F1']].max()) * 1.15])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'variant_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved variant comparison plot to: {output_path}")
    plt.close()


def plot_metrics_vs_seq_len(df, output_dir='results/plots'):
    """Plot multiple metrics vs sequence length in separate subplots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for sequence length experiments
    seq_exps = df[df['experiment'].isin(['SEQ_25', 'BASE', 'SEQ_100'])]
    seq_exps = seq_exps.sort_values('seq_length')
    
    # Create a comprehensive plot with all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metrics vs Sequence Length', fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Accuracy vs Sequence Length
    ax = axes[0, 0]
    ax.plot(seq_exps['seq_length'], seq_exps['test_accuracy'], 
            marker='o', linewidth=2.5, markersize=10, color='steelblue', label='Accuracy')
    for idx, row in seq_exps.iterrows():
        ax.annotate(f'{row["test_accuracy"]:.3f}', 
                   (row['seq_length'], row['test_accuracy']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy vs Sequence Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    ax.set_ylim([0.7, 0.85])
    
    # 2. F1 Score vs Sequence Length
    ax = axes[0, 1]
    ax.plot(seq_exps['seq_length'], seq_exps['test_f1'], 
            marker='s', linewidth=2.5, markersize=10, color='coral', label='F1 Score')
    for idx, row in seq_exps.iterrows():
        ax.annotate(f'{row["test_f1"]:.3f}', 
                   (row['seq_length'], row['test_f1']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('F1 Score vs Sequence Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    ax.set_ylim([0.7, 0.85])
    
    # 3. Test Loss vs Sequence Length
    ax = axes[0, 2]
    ax.plot(seq_exps['seq_length'], seq_exps['test_loss'], 
            marker='^', linewidth=2.5, markersize=10, color='darkred', label='Test Loss')
    for idx, row in seq_exps.iterrows():
        ax.annotate(f'{row["test_loss"]:.3f}', 
                   (row['seq_length'], row['test_loss']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Loss', fontsize=11, fontweight='bold')
    ax.set_title('Test Loss vs Sequence Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    
    # 4. Precision vs Sequence Length
    ax = axes[1, 0]
    ax.plot(seq_exps['seq_length'], seq_exps['test_precision'], 
            marker='D', linewidth=2.5, markersize=10, color='green', label='Precision')
    for idx, row in seq_exps.iterrows():
        ax.annotate(f'{row["test_precision"]:.3f}', 
                   (row['seq_length'], row['test_precision']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Precision vs Sequence Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    ax.set_ylim([0.7, 0.85])
    
    # 5. Recall vs Sequence Length
    ax = axes[1, 1]
    ax.plot(seq_exps['seq_length'], seq_exps['test_recall'], 
            marker='v', linewidth=2.5, markersize=10, color='purple', label='Recall')
    for idx, row in seq_exps.iterrows():
        ax.annotate(f'{row["test_recall"]:.3f}', 
                   (row['seq_length'], row['test_recall']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax.set_title('Recall vs Sequence Length', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    ax.set_ylim([0.7, 0.85])
    
    # 6. Combined: Accuracy and F1 together
    ax = axes[1, 2]
    ax.plot(seq_exps['seq_length'], seq_exps['test_accuracy'], 
            marker='o', linewidth=2.5, markersize=10, label='Accuracy', color='steelblue')
    ax.plot(seq_exps['seq_length'], seq_exps['test_f1'], 
            marker='s', linewidth=2.5, markersize=10, label='F1 Score', color='coral')
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy & F1 Score vs Sequence Length', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(seq_exps['seq_length'].unique())
    ax.set_ylim([0.7, 0.85])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_vs_seq_len.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive metrics vs seq length plot to: {output_path}")
    plt.close()
    
    # Also create individual plots for each metric
    metrics_to_plot = [
        ('test_accuracy', 'Accuracy', 'steelblue', 'o'),
        ('test_f1', 'F1 Score', 'coral', 's'),
        ('test_loss', 'Test Loss', 'darkred', '^'),
        ('test_precision', 'Precision', 'green', 'D'),
        ('test_recall', 'Recall', 'purple', 'v')
    ]
    
    for metric_col, metric_name, color, marker in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(seq_exps['seq_length'], seq_exps[metric_col], 
                marker=marker, linewidth=3, markersize=12, color=color, label=metric_name)
        
        # Add value annotations
        for idx, row in seq_exps.iterrows():
            ax.annotate(f'{row[metric_col]:.3f}', 
                       (row['seq_length'], row[metric_col]),
                       textcoords="offset points", xytext=(0,15), ha='center', 
                       fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Sequence Length', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_name} vs Sequence Length', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_xticks(seq_exps['seq_length'].unique())
        
        if metric_col != 'test_loss':
            ax.set_ylim([0.65, 0.85])
        
        plt.tight_layout()
        filename = f'{metric_col}_vs_seq_len.png'
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_name} vs seq length plot to: {output_path}")
        plt.close()


def get_model_display_name(model_row):
    """
    Generate a descriptive model name from model parameters.
    
    Args:
        model_row: Series or dict with model parameters
        
    Returns:
        str: Descriptive model name
    """
    arch = model_row['architecture'].lower()
    seq_len = int(model_row['seq_length'])
    opt = model_row['optimizer'].upper()
    activation = model_row.get('activation', 'N/A')
    grad_clip = model_row.get('grad_clip', None)
    
    # Build architecture name
    if arch == 'lstm':
        arch_name = 'LSTM'
    elif arch == 'bilstm':
        arch_name = 'BiLSTM'
    elif arch == 'rnn':
        if activation and activation != 'N/A' and activation.lower() != 'none':
            arch_name = f'RNN-{activation.capitalize()}'
        else:
            arch_name = 'RNN'
    else:
        arch_name = arch.upper()
    
    # Build full name
    name_parts = [arch_name, f'L={seq_len}', opt]
    
    # Add gradient clipping if present
    if grad_clip and not pd.isna(grad_clip) and grad_clip != 'None':
        name_parts.append(f'Clip={grad_clip}')
    
    return ' '.join(name_parts)


def plot_training_loss_curves(df, plots_dir='results/plots', output_dir='results/plots'):
    """
    Plot training loss vs epochs for the best and worst models.
    
    Args:
        df: DataFrame with experiment metrics
        plots_dir: Directory containing loss log files
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify best and worst models based on test_accuracy
    best_model = df.loc[df['test_accuracy'].idxmax()]
    worst_model = df.loc[df['test_accuracy'].idxmin()]
    
    best_exp_name = best_model['experiment']
    worst_exp_name = worst_model['experiment']
    
    # Get descriptive model names
    best_model_name = get_model_display_name(best_model)
    worst_model_name = get_model_display_name(worst_model)
    
    print(f"\nBest model: {best_model_name} (Accuracy: {best_model['test_accuracy']:.4f}, F1: {best_model['test_f1']:.4f})")
    print(f"Worst model: {worst_model_name} (Accuracy: {worst_model['test_accuracy']:.4f}, F1: {worst_model['test_f1']:.4f})")
    
    # Read loss log files
    best_log_path = os.path.join(plots_dir, f'{best_exp_name}_loss_log.txt')
    worst_log_path = os.path.join(plots_dir, f'{worst_exp_name}_loss_log.txt')
    
    if not os.path.exists(best_log_path):
        print(f"Warning: Loss log not found for best model: {best_log_path}")
        return
    
    if not os.path.exists(worst_log_path):
        print(f"Warning: Loss log not found for worst model: {worst_log_path}")
        return
    
    # Load loss logs
    best_log = pd.read_csv(best_log_path)
    worst_log = pd.read_csv(worst_log_path)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training loss curves
    epochs_best = best_log['Epoch'].values
    train_loss_best = best_log['Train_Loss'].values
    
    epochs_worst = worst_log['Epoch'].values
    train_loss_worst = worst_log['Train_Loss'].values
    
    # Plot with different styles
    ax.plot(epochs_best, train_loss_best, 
            marker='o', linewidth=2.5, markersize=8, 
            label=f'Best: {best_model_name} (Acc: {best_model["test_accuracy"]:.3f})',
            color='steelblue', alpha=0.8)
    
    ax.plot(epochs_worst, train_loss_worst, 
            marker='s', linewidth=2.5, markersize=8, 
            label=f'Worst: {worst_model_name} (Acc: {worst_model["test_accuracy"]:.3f})',
            color='coral', alpha=0.8)
    
    # Add value annotations only for first and last epochs to avoid clutter
    # Best model: annotate first (top) and last (bottom)
    idx_first_best = 0
    idx_last_best = len(epochs_best) - 1
    
    ax.annotate(f'Epoch {epochs_best[idx_first_best]}: {train_loss_best[idx_first_best]:.3f}',
               (epochs_best[idx_first_best], train_loss_best[idx_first_best]),
               textcoords="offset points", xytext=(15, 20), ha='left',
               fontsize=9, color='steelblue', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax.annotate(f'Epoch {epochs_best[idx_last_best]}: {train_loss_best[idx_last_best]:.3f}',
               (epochs_best[idx_last_best], train_loss_best[idx_last_best]),
               textcoords="offset points", xytext=(15, -25), ha='left',
               fontsize=9, color='steelblue', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Worst model: annotate first and last (both at bottom since loss is flat)
    idx_first_worst = 0
    idx_last_worst = len(epochs_worst) - 1
    
    ax.annotate(f'Epoch {epochs_worst[idx_first_worst]}: {train_loss_worst[idx_first_worst]:.3f}',
               (epochs_worst[idx_first_worst], train_loss_worst[idx_first_worst]),
               textcoords="offset points", xytext=(-15, -30), ha='right',
               fontsize=9, color='coral', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax.annotate(f'Epoch {epochs_worst[idx_last_worst]}: {train_loss_worst[idx_last_worst]:.3f}',
               (epochs_worst[idx_last_worst], train_loss_worst[idx_last_worst]),
               textcoords="offset points", xytext=(-15, -15), ha='right',
               fontsize=9, color='coral', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss vs. Epochs (Best vs Worst Models)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=1.5)
    ax.set_xticks(range(1, max(len(epochs_best), len(epochs_worst)) + 1))
    
    # Adjust y-axis to give more room for annotations
    y_min = min(train_loss_best.min(), train_loss_worst.min())
    y_max = max(train_loss_best.max(), train_loss_worst.max())
    y_range = y_max - y_min
    ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.15 * y_range])
    
    # Add model details as text box in bottom left to avoid overlap
    textstr = f'Best Model: {best_model_name}\nWorst Model: {worst_model_name}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray', linewidth=1)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training loss curves plot to: {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('--metrics_file', type=str, default='results/metrics.csv',
                        help='Path to metrics CSV file (default: results/metrics.csv)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        print(f"Error: Metrics file not found: {args.metrics_file}")
        return
    
    # Load metrics
    df = pd.read_csv(args.metrics_file)
    print(f"Loaded {len(df)} experiments from {args.metrics_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_variant_comparison(df)
    plot_metrics_vs_seq_len(df)
    plot_training_loss_curves(df)
    
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()

