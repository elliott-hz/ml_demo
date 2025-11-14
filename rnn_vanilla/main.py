"""
Main program entry: train and generate text
"""
import os

from train import train_model
from generate import load_model_and_generate


def main():
    # Hard-coded parameters
    mode = 'train'  # Options: 'train' or 'generate'
    text_file = 'nietzsche.txt'
    model_dir = 'saved_models'
    start_text = 'First Citizen:\n'
    gen_length = 200
    
    if mode == 'train':
        # Create model save directory
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Train model
        train_model(
            file_path=text_file,
            model_save_path=model_dir,
            epochs=50,
            batch_size=64,
            embedding_dim=50,
            hidden_dim=128,
            seq_length=50,
            learning_rate=0.001
        )
    elif mode == 'generate':
        # Check if model exists
        if not os.path.exists(model_dir):
            print("Model directory does not exist, please train the model first!")
            return
        
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model') and f.endswith('.pth')]
        if not model_files:
            print("No trained model found, please train the model first!")
            return
        
        # Select the latest model file
        model_files.sort()
        latest_model = model_files[-1]
        model_path = os.path.join(model_dir, latest_model)
        mappings_path = os.path.join(model_dir, 'char_mappings.pkl')
        
        # Generate text
        load_model_and_generate(
            model_path, 
            mappings_path, 
            start_text, 
            gen_length
        )


if __name__ == "__main__":
    main()