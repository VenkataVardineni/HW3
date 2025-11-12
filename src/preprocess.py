"""
Preprocessing script for IMDb dataset.
Processes the raw dataset according to assignment specifications:
- Lowercase, strip punctuation, tokenize
- Keep top 10k words
- Pad/truncate to lengths 25, 50, 100
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess_text(text):
    """
    Preprocess text: lowercase, strip punctuation, tokenize.
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if pd.isna(text):
        return []
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenize by splitting on whitespace
    tokens = text.split()
    
    return tokens


def build_vocabulary(texts, max_words=10000):
    """
    Build vocabulary from texts, keeping top max_words.
    
    Args:
        texts: List of tokenized texts
        max_words: Maximum vocabulary size
        
    Returns:
        word_to_idx: Dictionary mapping words to indices
        idx_to_word: Dictionary mapping indices to words
    """
    # Count word frequencies
    word_counts = Counter()
    for tokens in tqdm(texts, desc="Building vocabulary"):
        word_counts.update(tokens)
    
    # Get top max_words words (excluding index 0 which is reserved for padding)
    # We'll use index 0 for padding, 1 for unknown words
    top_words = word_counts.most_common(max_words - 2)  # -2 for padding and unknown
    
    # Create word to index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in top_words:
        word_to_idx[word] = len(word_to_idx)
    
    # Create index to word mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


def text_to_sequence(tokens, word_to_idx, max_length):
    """
    Convert tokens to sequence of indices.
    
    Args:
        tokens: List of tokens
        word_to_idx: Dictionary mapping words to indices
        max_length: Maximum sequence length (pad or truncate)
        
    Returns:
        List of indices
    """
    # Convert tokens to indices
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    
    # Truncate if too long
    if len(indices) > max_length:
        indices = indices[:max_length]
    # Pad if too short
    elif len(indices) < max_length:
        indices = indices + [word_to_idx['<PAD>']] * (max_length - len(indices))
    
    return indices


def main():
    """Main preprocessing function."""
    # Paths
    raw_data_path = 'data/raw/IMDB Dataset.csv'
    processed_data_dir = 'data/processed'
    
    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Load raw data
    print("Loading raw dataset...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Dataset not found at {raw_data_path}. "
            "Please download 'IMDB Dataset.csv' from Kaggle and place it in data/raw/"
        )
    
    df = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df)} samples")
    
    # Preprocess texts
    print("Preprocessing texts...")
    texts = []
    for text in tqdm(df['review'], desc="Preprocessing texts"):
        tokens = preprocess_text(text)
        texts.append(tokens)
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(texts, max_words=10000)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Encode labels (positive=1, negative=0)
    labels = (df['sentiment'] == 'positive').astype(int).values
    
    # Split data - 50/50 split as required by assignment (25k train / 25k test)
    print("Splitting data (50/50 split: 25k train / 25k test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Process sequences for different lengths
    sequence_lengths = [25, 50, 100]
    
    for seq_len in sequence_lengths:
        print(f"\nProcessing sequences of length {seq_len}...")
        
        # Convert texts to sequences
        X_train_seq = []
        for tokens in tqdm(X_train, desc=f"Processing train sequences (len={seq_len})"):
            seq = text_to_sequence(tokens, word_to_idx, seq_len)
            X_train_seq.append(seq)
        
        X_test_seq = []
        for tokens in tqdm(X_test, desc=f"Processing test sequences (len={seq_len})"):
            seq = text_to_sequence(tokens, word_to_idx, seq_len)
            X_test_seq.append(seq)
        
        # Convert to numpy arrays
        X_train_seq = np.array(X_train_seq)
        X_test_seq = np.array(X_test_seq)
        
        # Save processed data
        save_path = os.path.join(processed_data_dir, f'imdb_processed_seqlen_{seq_len}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'X_train': X_train_seq,
                'X_test': X_test_seq,
                'y_train': y_train,
                'y_test': y_test,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'vocab_size': len(word_to_idx),
                'seq_length': seq_len
            }, f)
        
        print(f"Saved processed data to {save_path}")
        print(f"  Train shape: {X_train_seq.shape}, Test shape: {X_test_seq.shape}")
    
    # Save vocabulary separately
    vocab_path = os.path.join(processed_data_dir, 'vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': len(word_to_idx)
        }, f)
    
    print(f"\nVocabulary saved to {vocab_path}")
    print("Preprocessing complete!")


if __name__ == '__main__':
    main()

