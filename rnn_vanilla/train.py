import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle

from data_preprocessing import TextDataset, load_text_data
from model import VanillaRNN


def train_model(file_path, model_save_path, epochs=100, batch_size=64, 
                embedding_dim=50, hidden_dim=128, seq_length=50, learning_rate=0.001):
    """
    Train RNN model
    :param file_path: Text file path
    :param model_save_path: Model save path
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    :param embedding_dim: Embedding dimension
    :param hidden_dim: Hidden layer dimension
    :param seq_length: Sequence length
    :param learning_rate: Learning rate
    """
    # Load text data
    text = load_text_data(file_path)
    print(f"Total text length: {len(text)} characters")
    
    # Create dataset
    dataset = TextDataset(text, seq_length)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"First 10 characters: {dataset.chars[:10]}")
    
    # Save character mappings for text generation
    with open(os.path.join(model_save_path, 'char_mappings.pkl'), 'wb') as f:
        pickle.dump({
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char,
            'vocab_size': dataset.vocab_size
        }, f)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = VanillaRNN(dataset.vocab_size, embedding_dim, hidden_dim)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Start training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            output, _ = model(data)
            
            # Calculate loss (need to reshape output and targets)
            loss = criterion(output.reshape(-1, dataset.vocab_size), targets.reshape(-1))
            
            # Backward propagation
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print training progress
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')
    
    # Save final model
    final_model_path = os.path.join(model_save_path, 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')


if __name__ == "__main__":
    # Create model save directory
    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Train model
    train_model(
        file_path="nietzsche.txt",
        model_save_path=model_save_dir,
        epochs=50,  # Reduce epochs to save time
        batch_size=64,
        embedding_dim=50,
        hidden_dim=128,
        seq_length=50,
        learning_rate=0.001
    )