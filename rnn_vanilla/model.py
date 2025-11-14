import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
    Basic RNN model implementation
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer to convert character indices to vector representations
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer with tanh activation function
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to map RNN output back to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward propagation process
        :param x: Input tensor with shape (batch_size, seq_length)
        :param hidden: Hidden state, default is None
        :return: Output tensor and new hidden state
        """
        # Pass input through embedding layer to convert to vector representation
        embedded = self.embedding(x)
        
        # Initialize hidden state to zero if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        # Pass through RNN layer
        rnn_out, hidden = self.rnn(embedded, hidden)
        
        # Pass through fully connected layer to get output at each time step
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden state
        :param batch_size: Batch size
        :return: Initialized hidden state
        """
        # Create zero tensor with shape (num_layers, batch_size, hidden_dim)
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


class LSTMModel(nn.Module):
    """
    LSTM model implementation (for comparison)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        Forward propagation process
        """
        embedded = self.embedding(x)
        
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize LSTM hidden state and cell state
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)