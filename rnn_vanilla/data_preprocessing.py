import torch
import numpy as np


class TextDataset(torch.utils.data.Dataset):
    """
    Text dataset class for processing text data and converting it to trainable format
    """
    def __init__(self, text, seq_length=50):
        # Get unique characters from the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings between characters and indices
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Convert text to index sequence
        self.data = [self.char_to_idx[ch] for ch in text]
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Input sequence is seq_length characters starting from current index
        x = self.data[idx:idx+self.seq_length]
        # Output is the next character after the input sequence
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_text_data(file_path):
    """
    Load text data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text