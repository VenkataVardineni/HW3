"""
LSTM/RNN model definitions for IMDb sentiment analysis.
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """
    RNN model for sentiment analysis with configurable activation.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (default: 100)
        hidden_dim: Dimension of RNN hidden state
        num_layers: Number of RNN layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        activation: Activation function ('tanh', 'relu', 'sigmoid')
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, 
                 num_layers=2, dropout=0.5, activation='tanh'):
        super(RNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer - PyTorch nn.RNN only supports 'tanh' and 'relu'
        # For 'sigmoid', we'll use RNNCell in a loop
        if activation == 'sigmoid':
            # Use RNNCell for sigmoid (manual implementation)
            self.rnn_cells = nn.ModuleList([
                nn.RNNCell(embedding_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            self.rnn = None
            self.use_cells = True
        else:
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                nonlinearity=activation  # 'tanh' or 'relu'
            )
            self.rnn_cells = None
            self.use_cells = False
        
        self.activation_fn = nn.Sigmoid() if activation == 'sigmoid' else None
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        if self.use_cells:
            # Manual RNN with sigmoid activation
            batch_size, seq_len, _ = embedded.shape
            hiddens = [torch.zeros(batch_size, self.hidden_dim, device=x.device) 
                      for _ in range(self.num_layers)]
            
            for t in range(seq_len):
                input_t = embedded[:, t, :]  # (batch_size, embedding_dim)
                layer_input = input_t
                for layer_idx in range(self.num_layers):
                    h = self.rnn_cells[layer_idx](layer_input, hiddens[layer_idx])
                    # Apply sigmoid activation
                    h = self.activation_fn(h)
                    hiddens[layer_idx] = h
                    # Apply dropout between layers (except last)
                    if layer_idx < self.num_layers - 1:
                        h = self.dropout_layer(h)
                    layer_input = h  # Next layer uses this layer's output
            
            hidden = hiddens[-1]  # (batch_size, hidden_dim)
        else:
            # Standard RNN
            rnn_out, hidden = self.rnn(embedded)
            hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout
        hidden = self.dropout_layer(hidden)
        
        # Fully connected layer
        output = self.fc(hidden)  # (batch_size, 1)
        
        # Sigmoid activation
        output = self.sigmoid(output)
        
        return output


class LSTMSentimentModel(nn.Module):
    """
    LSTM model for sentiment analysis.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (default: 100)
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        bidirectional: Whether to use bidirectional LSTM (default: False)
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=2, dropout=0.5, bidirectional=False):
        super(LSTMSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(lstm_output_dim, 1)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout
        hidden = self.dropout_layer(hidden)
        
        # Fully connected layer
        output = self.fc(hidden)  # (batch_size, 1)
        
        # Sigmoid activation
        output = self.sigmoid(output)
        
        return output


def create_model(vocab_size, embedding_dim=100, hidden_dim=64, 
                 num_layers=2, dropout=0.5, bidirectional=False,
                 architecture='lstm', activation='tanh'):
    """
    Create and return a new model instance.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (default: 100)
        hidden_dim: Dimension of hidden state (default: 64)
        num_layers: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        bidirectional: Whether to use bidirectional (default: False, only for LSTM)
        architecture: Model architecture ('lstm', 'rnn', 'bilstm')
        activation: Activation function for RNN ('tanh', 'relu', 'sigmoid')
        
    Returns:
        Model instance
    """
    if architecture.lower() == 'rnn':
        model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
    elif architecture.lower() == 'bilstm':
        model = LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )
    else:  # 'lstm' (default)
        model = LSTMSentimentModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
    return model

