# HW3: LSTM Sentiment Analysis on IMDb Dataset

This project implements and evaluates LSTM, RNN, and BiLSTM models for IMDb sentiment analysis using a one-factor-at-a-time (OFAT) experimental design.

## 📋 Project Overview

We conducted a comprehensive experimental study to evaluate different architectures, optimizers, sequence lengths, and training strategies for sentiment classification on the IMDb dataset. The project follows a systematic OFAT approach, varying one factor at a time while keeping all other hyperparameters fixed.

## 🎯 What We've Done

### 1. **Data Preprocessing**
- **Dataset:** IMDb 50k reviews (25k train / 25k test split)
- **Preprocessing Steps:**
  - Lowercase conversion
  - Punctuation removal
  - Tokenization
  - Vocabulary building (top 10k words)
  - Sequence padding/truncation to lengths: 25, 50, 100

### 2. **Model Implementation**
- **LSTM Model:** Unidirectional LSTM with configurable layers
- **RNN Model:** Vanilla RNN with support for tanh, ReLU, and sigmoid activations
- **BiLSTM Model:** Bidirectional LSTM for enhanced context understanding
- **Common Architecture:**
  - Embedding dimension: 100
  - Hidden dimension: 64
  - Number of layers: 2
  - Dropout: 0.5
  - Batch size: 32
  - Output: Fully connected layer with sigmoid activation
  - Loss: Binary Cross-Entropy (BCE)

### 3. **14 OFAT Experiments Conducted**

We ran 14 experiments, varying one factor at a time:

| # | Experiment Name | Architecture | Activation | Optimizer | Seq Length | Grad Clip | Description |
|---|----------------|--------------|------------|-----------|------------|-----------|-------------|
| 1 | BASE | LSTM | ReLU | Adam | 50 | None | Baseline model |
| 2 | ARCH_RNN_TANH | RNN | tanh | Adam | 50 | None | RNN with tanh activation |
| 3 | ARCH_BILSTM | BiLSTM | ReLU | Adam | 50 | None | Bidirectional LSTM |
| 4 | RNN_RELU | RNN | relu | Adam | 50 | None | RNN with ReLU activation |
| 5 | RNN_SIGMOID | RNN | sigmoid | Adam | 50 | None | RNN with sigmoid activation |
| 6 | OPT_SGD | LSTM | ReLU | SGD | 50 | None | LSTM with SGD optimizer |
| 7 | OPT_RMSPROP | LSTM | ReLU | RMSProp | 50 | None | LSTM with RMSProp optimizer |
| 8 | SEQ_25 | LSTM | ReLU | Adam | 25 | None | Sequence length 25 |
| 9 | SEQ_100 | LSTM | ReLU | Adam | 100 | None | Sequence length 100 |
| 10 | CLIP_ON | LSTM | ReLU | Adam | 50 | 1.0 | Gradient clipping enabled |
| 11 | ARCH_RNN_TANH_CLIP | RNN | tanh | Adam | 50 | 1.0 | RNN (tanh) with clipping |
| 12 | ARCH_BILSTM_CLIP | BiLSTM | ReLU | Adam | 50 | 1.0 | BiLSTM with clipping |
| 13 | OPT_SGD_CLIP | LSTM | ReLU | SGD | 50 | 1.0 | SGD with clipping |
| 14 | OPT_RMSPROP_CLIP | LSTM | ReLU | RMSProp | 50 | 1.0 | RMSProp with clipping |

### 4. **Key Results Summary**

**Best Performing Models:**
- **Best Overall:** `SEQ_100` - Accuracy: 0.8122, F1: 0.8137, Loss: 0.4244
- **Best at L=50:** `BASE` - Accuracy: 0.7688, F1: 0.7635, Loss: 0.4904
- **Best Optimizer:** `OPT_RMSPROP` - Accuracy: 0.7664, F1: 0.7687, Loss: 0.5009
- **Best Architecture:** `ARCH_BILSTM` - Accuracy: 0.7530, F1: 0.7693, Loss: 0.5047

**Key Findings:**
- Longer sequences (L=100) significantly improve performance
- RMSProp optimizer performs comparably to Adam
- BiLSTM shows good F1 score but higher test loss
- RNN with sigmoid activation performs poorly
- Gradient clipping has mixed effects depending on architecture

## 🛠️ How We Did It

### Step 1: Project Setup

1. **Created project structure:**
   ```
   HW3/
   ├── data/
   │   ├── raw/           # Raw dataset
   │   └── processed/     # Preprocessed data
   ├── src/               # Source code
   ├── results/
   │   ├── metrics.csv   # Experiment results
   │   └── plots/        # Visualizations
   └── requirements.txt
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Downloaded dataset:**
   - Downloaded `IMDB Dataset.csv` from Kaggle
   - Placed in `data/raw/IMDB Dataset.csv`

### Step 2: Data Preprocessing

**File:** `src/preprocess.py`

**Process:**
1. Loaded raw CSV file (50k reviews)
2. Applied text cleaning:
   - Converted to lowercase
   - Removed punctuation
   - Tokenized text
3. Built vocabulary from top 10k most frequent words
4. Created sequences for three lengths: 25, 50, 100
5. Split data: 50/50 train/test (25k train / 25k test)
6. Saved processed data to `data/processed/`

**Command:**
```bash
python -m src.preprocess
```

### Step 3: Model Implementation

**File:** `src/models.py`

**Models Created:**
1. **LSTMSentimentModel:** Standard unidirectional LSTM
2. **RNNModel:** Vanilla RNN with configurable activation (tanh/relu/sigmoid)
3. **BiLSTM:** Bidirectional LSTM (via `bidirectional=True`)

**Architecture Details:**
- Embedding layer: `nn.Embedding(vocab_size, embedding_dim)`
- RNN/LSTM layer: `nn.LSTM` or `nn.RNN` with configurable layers
- Dropout: Applied between layers
- Output: `nn.Linear(hidden_dim, 1)` + `nn.Sigmoid()`

### Step 4: Training Infrastructure

**File:** `src/train.py`

**Key Features:**
- `run_experiment()`: Main function to run a single experiment
- Supports multiple architectures (LSTM, RNN, BiLSTM)
- Supports multiple optimizers (Adam, SGD, RMSProp)
- Gradient clipping support
- Early stopping with best model restoration
- Progress bars using tqdm
- Comprehensive metrics logging (accuracy, precision, recall, F1, loss)
- Loss logs saved per experiment

**Training Process:**
1. Load processed data for specified sequence length
2. Create data loaders (batch size 32)
3. Initialize model with specified architecture
4. Train for N epochs with progress tracking
5. Evaluate on test set after each epoch
6. Save best model state (if early stopping enabled)
7. Log all metrics to CSV

### Step 5: Running Experiments

**File:** `src/run_experiments.py`

**Process:**
1. Defined 14 experiment configurations
2. Set fixed constants (embedding=100, hidden=64, layers=2, dropout=0.5, batch=32, lr=1e-3)
3. Ran each experiment sequentially
4. Collected metrics for each run
5. Saved all results to `results/metrics.csv`
6. Generated loss logs in `results/plots/`

**Command:**
```bash
python -m src.run_experiments --epochs 10 --seed 42
```

### Step 6: Visualization

**File:** `src/plot_results.py`

**Plots Generated:**

1. **Variant Comparison Bar Chart** (`variant_comparison.png`)
   - Compares: BASE, RNN (tanh/relu/sigmoid), BiLSTM, SGD, RMSProp, CLIP_ON
   - Shows Accuracy and F1 Score side-by-side

2. **Comprehensive Metrics vs Sequence Length** (`metrics_vs_seq_len.png`)
   - 6-panel overview showing:
     - Accuracy vs Sequence Length
     - F1 Score vs Sequence Length
     - Test Loss vs Sequence Length
     - Precision vs Sequence Length
     - Recall vs Sequence Length
     - Combined Accuracy & F1

3. **Individual Metric Plots:**
   - `test_accuracy_vs_seq_len.png`
   - `test_f1_vs_seq_len.png`
   - `test_loss_vs_seq_len.png`
   - `test_precision_vs_seq_len.png`
   - `test_recall_vs_seq_len.png`

**Command:**
```bash
python -m src.plot_results
```

## 📁 Project Structure

```
HW3/
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv          # Raw dataset (not in repo)
│   └── processed/
│       ├── imdb_processed_seqlen_25.pkl
│       ├── imdb_processed_seqlen_50.pkl
│       ├── imdb_processed_seqlen_100.pkl
│       └── vocabulary.pkl
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Data preprocessing
│   ├── models.py                     # Model definitions (LSTM, RNN, BiLSTM)
│   ├── train.py                      # Training script with run_experiment()
│   ├── evaluate.py                   # Model evaluation utilities
│   ├── utils.py                      # Dataset and data loader utilities
│   ├── run_experiments.py            # Script to run all 14 OFAT experiments
│   └── plot_results.py               # Visualization generation
├── results/
│   ├── metrics.csv                   # All experiment results (14 rows)
│   └── plots/
│       ├── variant_comparison.png
│       ├── metrics_vs_seq_len.png
│       ├── test_accuracy_vs_seq_len.png
│       ├── test_f1_vs_seq_len.png
│       ├── test_loss_vs_seq_len.png
│       ├── test_precision_vs_seq_len.png
│       ├── test_recall_vs_seq_len.png
│       └── *_loss_log.txt            # 14 loss log files (one per experiment)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🚀 Setup Instructions

### 1. Clone/Download the Repository

```bash
cd /path/to/your/workspace
# If using git:
git clone <repository-url>
cd HW3
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- torch>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- tqdm>=4.65.0
- matplotlib>=3.7.0

### 4. Download Dataset

1. Download `IMDB Dataset.csv` from Kaggle
2. Place it in `data/raw/IMDB Dataset.csv`

## 📊 Usage

### Preprocess Data

```bash
python -m src.preprocess
```

This will:
- Load the raw dataset
- Preprocess texts (lowercase, strip punctuation, tokenize)
- Build vocabulary (top 10k words)
- Create sequences of lengths 25, 50, and 100
- Split into 50/50 train/test (25k/25k)
- Save processed data to `data/processed/`

### Run All Experiments

```bash
python -m src.run_experiments --epochs 10 --seed 42
```

This will:
- Run all 14 OFAT experiments
- Save metrics to `results/metrics.csv`
- Generate loss logs in `results/plots/`
- Display progress bars and results

### Generate Plots

```bash
python -m src.plot_results
```

This will:
- Load metrics from `results/metrics.csv`
- Generate variant comparison plot
- Generate comprehensive metrics vs sequence length plots
- Save all plots to `results/plots/`

### Run Individual Experiment

You can also run individual experiments using `train.py`:

```bash
python -m src.train \
    --experiment_name "custom_experiment" \
    --seq_length 100 \
    --embedding_dim 100 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.5 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 0.001 \
    --optimizer adam \
    --architecture lstm \
    --activation relu
```

## 📈 Results Summary

### Performance by Category

**Sequence Length Impact:**
- L=25: Accuracy 0.7226, F1 0.7239
- L=50: Accuracy 0.7688, F1 0.7635 (BASE)
- L=100: Accuracy 0.8122, F1 0.8137 ⭐ **Best**

**Architecture Comparison (at L=50):**
- LSTM (BASE): Accuracy 0.7688, F1 0.7635
- BiLSTM: Accuracy 0.7530, F1 0.7693
- RNN (tanh): Accuracy 0.6194, F1 0.6494
- RNN (relu): Accuracy 0.7461, F1 0.7499
- RNN (sigmoid): Accuracy 0.5701, F1 0.6061

**Optimizer Comparison (at L=50):**
- Adam: Accuracy 0.7688, F1 0.7635
- RMSProp: Accuracy 0.7664, F1 0.7687
- SGD: Accuracy 0.5157, F1 0.4564 ❌

**Gradient Clipping Impact:**
- Generally helps with RNN architectures
- Mixed results with LSTM/BiLSTM
- Best improvement: RNN (tanh) with clipping

## 🔬 Experimental Design

### Fixed Parameters (All Experiments)
- Embedding dimension: 100
- Hidden dimension: 64
- Number of layers: 2
- Dropout: 0.5
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0
- Early stopping: False
- Epochs: 10

### Varied Parameters (One at a Time)
- **Architecture:** LSTM, RNN, BiLSTM
- **Activation:** tanh, relu, sigmoid (RNN only)
- **Optimizer:** Adam, SGD, RMSProp
- **Sequence Length:** 25, 50, 100
- **Gradient Clipping:** None, 1.0

### Design Principles
- **OFAT (One Factor At A Time):** Only one parameter varied per experiment
- **Reproducibility:** Fixed random seed (42)
- **Consistency:** Same preprocessing, same data splits
- **Comprehensive Logging:** All metrics recorded for analysis

## 📝 Code Files - Detailed Explanation

This section explains what each source code file does and its key functions.

### `src/__init__.py`
**Purpose:** Makes `src` a Python package, enabling relative imports.

**What it does:**
- Empty file that marks the `src` directory as a Python package
- Allows imports like `from .models import create_model` instead of absolute imports

---

### `src/preprocess.py`
**Purpose:** Preprocesses the raw IMDb dataset according to assignment specifications.

**Key Functions:**
- `preprocess_text(text)`: Converts text to lowercase, removes punctuation, and tokenizes
- `build_vocabulary(texts, max_words=10000)`: Builds vocabulary from top 10k most frequent words
- `texts_to_sequences(texts, word_to_idx, max_length)`: Converts texts to sequences of token IDs
- `main()`: Orchestrates the entire preprocessing pipeline

**What it does:**
1. Loads raw CSV file from `data/raw/IMDB Dataset.csv`
2. Applies text preprocessing (lowercase, punctuation removal, tokenization)
3. Builds vocabulary from top 10,000 most frequent words
4. Converts texts to sequences of token IDs
5. Pads/truncates sequences to lengths 25, 50, and 100
6. Splits data into 50/50 train/test (25k train / 25k test)
7. Saves processed data as pickle files in `data/processed/`

**Output Files:**
- `data/processed/vocabulary.pkl`: Word-to-index mapping
- `data/processed/imdb_processed_seqlen_25.pkl`: Sequences of length 25
- `data/processed/imdb_processed_seqlen_50.pkl`: Sequences of length 50
- `data/processed/imdb_processed_seqlen_100.pkl`: Sequences of length 100

**Usage:**
```bash
python -m src.preprocess
```

---

### `src/models.py`
**Purpose:** Defines all neural network model architectures (LSTM, RNN, BiLSTM).

**Key Classes:**

1. **`RNNModel(nn.Module)`**
   - Vanilla RNN model with configurable activation function
   - Supports tanh, ReLU, and sigmoid activations
   - For sigmoid, uses `nn.RNNCell` in a manual loop (PyTorch limitation)
   - **Architecture:** Embedding → RNN layers → Dropout → FC → Sigmoid

2. **`LSTMSentimentModel(nn.Module)`**
   - Standard LSTM model (unidirectional or bidirectional)
   - **Architecture:** Embedding → LSTM layers → Dropout → FC → Sigmoid
   - Supports bidirectional LSTM via `bidirectional` parameter

**Key Functions:**
- `create_model(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, bidirectional, architecture, activation)`: Factory function that creates the appropriate model based on parameters

**What it does:**
- Defines model architectures with configurable hyperparameters
- Handles different architectures (LSTM, RNN, BiLSTM) through a unified interface
- Implements proper forward pass for each architecture type
- Returns sigmoid-activated outputs for binary classification

**Key Parameters:**
- `vocab_size`: Size of vocabulary (10,000 in our case)
- `embedding_dim`: Word embedding dimension (100)
- `hidden_dim`: Hidden state dimension (64)
- `num_layers`: Number of RNN/LSTM layers (2)
- `dropout`: Dropout rate (0.5)
- `bidirectional`: Whether to use bidirectional LSTM
- `architecture`: 'lstm', 'rnn', or 'bilstm'
- `activation`: 'tanh', 'relu', or 'sigmoid' (for RNN only)

---

### `src/utils.py`
**Purpose:** Provides utility functions for dataset handling, data loading, and model I/O.

**Key Classes:**

1. **`IMDBDataset(Dataset)`**
   - PyTorch Dataset class for IMDb data
   - Converts numpy arrays to PyTorch tensors
   - Handles indexing and data retrieval

**Key Functions:**

1. **`create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)`**
   - Creates PyTorch DataLoader objects for training and testing
   - Handles batching and shuffling
   - Returns `train_loader` and `test_loader`

2. **`load_processed_data(seq_length)`**
   - Loads preprocessed data for a specific sequence length
   - Returns `X_train, y_train, X_test, y_test, vocab_size`
   - Used by training scripts to get data

3. **`save_model(model, filepath)`**
   - Saves model state dictionary to disk
   - Used for model checkpointing

4. **`load_model(model, filepath)`**
   - Loads model state dictionary from disk
   - Used for model evaluation or resuming training

**What it does:**
- Provides data loading infrastructure
- Handles conversion between numpy and PyTorch tensors
- Manages model persistence (save/load)
- Creates efficient data loaders with batching

---

### `src/train.py`
**Purpose:** Core training script that implements the training loop, evaluation, and experiment management.

**Key Functions:**

1. **`train_epoch(model, train_loader, criterion, optimizer, device, grad_clip, verbose)`**
   - Trains model for one epoch
   - Handles forward pass, backward pass, and optimization
   - Supports gradient clipping
   - Returns average training loss
   - Shows progress bar if `verbose=True`

2. **`evaluate(model, test_loader, criterion, device, verbose)`**
   - Evaluates model on test set
   - Calculates loss, accuracy, precision, recall, F1 score
   - Returns dictionary of all metrics
   - Shows progress bar if `verbose=True`

3. **`run_experiment(experiment_name, seq_length, embedding_dim, hidden_dim, num_layers, dropout, bidirectional, batch_size, epochs, learning_rate, optimizer_name, grad_clip, weight_decay, early_stopping, early_stop_patience, early_stop_min_delta, device, results_dir, plots_dir, architecture, activation)`**
   - **Main function** that runs a complete experiment
   - Loads data, creates model, trains, evaluates
   - Implements early stopping with best model restoration
   - Logs metrics to CSV and saves loss logs
   - Returns dictionary with all experiment metrics

**What it does:**
1. Loads preprocessed data for specified sequence length
2. Creates data loaders (batch size 32)
3. Initializes model with specified architecture and hyperparameters
4. Creates optimizer (Adam, SGD, or RMSProp) with optional weight decay
5. Trains model for N epochs:
   - Forward pass, loss calculation, backward pass
   - Gradient clipping (if specified)
   - Evaluation after each epoch
   - Early stopping (if enabled) with best model restoration
6. Calculates comprehensive metrics (accuracy, precision, recall, F1, loss)
7. Saves loss logs to text file
8. Returns all metrics for CSV logging

**Key Features:**
- Progress bars using tqdm
- Early stopping with patience and min_delta
- Best model state saving and restoration
- Comprehensive metrics calculation
- Loss logging per epoch
- Support for multiple architectures and optimizers

**Usage:**
```bash
# Called by run_experiments.py, or directly:
python -m src.train --experiment_name "test" --seq_length 50 ...
```

---

### `src/evaluate.py`
**Purpose:** Standalone evaluation script for loading and evaluating saved models.

**Key Functions:**

1. **`evaluate_model(model, test_loader, device)`**
   - Evaluates a trained model on test data
   - Calculates accuracy, precision, recall, F1, confusion matrix
   - Returns dictionary of metrics

2. **`main()`**
   - Command-line interface for model evaluation
   - Loads saved model from disk
   - Evaluates on test set
   - Prints results

**What it does:**
- Loads a saved model checkpoint
- Evaluates it on the test set
- Displays comprehensive performance metrics
- Useful for evaluating models trained earlier

**Usage:**
```bash
python -m src.evaluate --model_path models/model.pth --seq_length 50 ...
```

---

### `src/run_experiments.py`
**Purpose:** Orchestrates and runs all 14 OFAT (One Factor At A Time) experiments.

**Key Functions:**

1. **`set_seed(seed=42)`**
   - Sets random seeds for reproducibility
   - Ensures consistent results across runs

2. **`append_to_csv(metrics_list, csv_file)`**
   - Appends experiment metrics to CSV file
   - Handles header writing for new files

3. **`main()`**
   - Main orchestration function
   - Defines all 14 experiment configurations
   - Runs experiments sequentially
   - Collects and saves all results

**What it does:**
1. Sets random seed for reproducibility
2. Defines 14 experiment configurations:
   - BASE (baseline LSTM)
   - Architecture variants (RNN, BiLSTM)
   - Activation variants (tanh, relu, sigmoid for RNN)
   - Optimizer variants (SGD, RMSProp)
   - Sequence length variants (25, 100)
   - Gradient clipping variants
3. For each experiment:
   - Calls `run_experiment()` from `train.py`
   - Collects metrics
   - Displays progress
4. Saves all metrics to `results/metrics.csv`
5. Prints summary of all experiments

**Experiment Categories:**
- **Baseline:** BASE (LSTM, L=50, Adam, no clip)
- **Architecture:** RNN (tanh/relu/sigmoid), BiLSTM
- **Optimizer:** SGD, RMSProp
- **Sequence Length:** 25, 100
- **Stability:** Gradient clipping variants

**Usage:**
```bash
python -m src.run_experiments --epochs 10 --seed 42
```

---

### `src/plot_results.py`
**Purpose:** Generates all visualization plots from experiment results.

**Key Functions:**

1. **`plot_variant_comparison(df, output_dir)`**
   - Creates bar chart comparing different model variants
   - Shows: BASE, RNN (tanh/relu/sigmoid), BiLSTM, SGD, RMSProp, CLIP_ON
   - Displays Accuracy and F1 Score side-by-side
   - Saves to `variant_comparison.png`

2. **`plot_metrics_vs_seq_len(df, output_dir)`**
   - Creates comprehensive plots for metrics vs sequence length
   - Generates:
     - 6-panel overview plot (`metrics_vs_seq_len.png`)
     - Individual plots for each metric:
       - `test_accuracy_vs_seq_len.png`
       - `test_f1_vs_seq_len.png`
       - `test_loss_vs_seq_len.png`
       - `test_precision_vs_seq_len.png`
       - `test_recall_vs_seq_len.png`
   - Includes value annotations on data points
   - Uses different markers and colors for clarity

3. **`main()`**
   - Loads metrics from CSV
   - Calls plotting functions
   - Saves all plots to `results/plots/`

**What it does:**
1. Loads `results/metrics.csv` using pandas
2. Filters data for relevant experiments
3. Generates variant comparison bar chart
4. Generates comprehensive sequence length analysis:
   - 6-panel overview with all metrics
   - Individual plots for each metric
5. Saves all plots as high-resolution PNG files (300 DPI)

**Output Files:**
- `results/plots/variant_comparison.png`
- `results/plots/metrics_vs_seq_len.png` (6-panel overview)
- `results/plots/test_accuracy_vs_seq_len.png`
- `results/plots/test_f1_vs_seq_len.png`
- `results/plots/test_loss_vs_seq_len.png`
- `results/plots/test_precision_vs_seq_len.png`
- `results/plots/test_recall_vs_seq_len.png`

**Usage:**
```bash
python -m src.plot_results
```

---

## 📋 File Dependencies Flow

```
preprocess.py
    ↓ (creates processed data)
    ↓
run_experiments.py
    ↓ (uses)
    ├── train.py
    │   ├── models.py (creates models)
    │   └── utils.py (loads data, creates loaders)
    │
    ↓ (generates metrics.csv)
    ↓
plot_results.py
    ↓ (reads metrics.csv)
    ↓ (generates plots)
```

**Data Flow:**
1. `preprocess.py` → Creates processed data files
2. `run_experiments.py` → Orchestrates experiments
3. `train.py` → Uses `models.py` and `utils.py` to train models
4. Results saved to `metrics.csv`
5. `plot_results.py` → Reads `metrics.csv` and generates visualizations

## 🎓 Assignment Compliance

✅ **All Requirements Met:**
- Dataset: IMDb 50k with 50/50 split (25k train / 25k test)
- Preprocessing: Lowercase, strip punctuation, tokenize
- Vocabulary: Top 10k words
- Sequence lengths: 25, 50, 100
- Fixed constants: embedding=100, hidden=64, layers=2, dropout=0.5, batch=32
- Output: Sigmoid activation with BCE loss
- OFAT design: One factor varied at a time
- Comprehensive metrics: Accuracy, Precision, Recall, F1, Loss
- Visualizations: Variant comparison and metrics vs sequence length

## 📊 Output Files

### Metrics
- **`results/metrics.csv`**: Contains all 14 experiment results with complete metrics

### Plots
- **`results/plots/variant_comparison.png`**: Bar chart comparing all variants
- **`results/plots/metrics_vs_seq_len.png`**: 6-panel comprehensive view
- **`results/plots/test_accuracy_vs_seq_len.png`**: Accuracy only
- **`results/plots/test_f1_vs_seq_len.png`**: F1 Score only
- **`results/plots/test_loss_vs_seq_len.png`**: Test Loss only
- **`results/plots/test_precision_vs_seq_len.png`**: Precision only
- **`results/plots/test_recall_vs_seq_len.png`**: Recall only

### Loss Logs
- 14 individual loss log files (one per experiment) in `results/plots/`

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError:**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **FileNotFoundError for dataset:**
   - Ensure `IMDB Dataset.csv` is in `data/raw/`
   - Check file name matches exactly

3. **CUDA out of memory:**
   - Reduce batch size in training script
   - Use CPU instead (default)

4. **Preprocessing errors:**
   - Ensure dataset is properly formatted CSV
   - Check that dataset has 'review' and 'sentiment' columns

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- IMDb Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## 👤 Author

HW3 Assignment - LSTM Sentiment Analysis

## 📄 License

This project is for educational purposes.
