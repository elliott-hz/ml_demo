import torch
import pickle
import os

from model import VanillaRNN


def generate_text(model, char_to_idx, idx_to_char, start_text, length=200, temperature=1.0):
    """
    Generate text using trained model
    :param model: Trained model
    :param char_to_idx: Character to index mapping
    :param idx_to_char: Index to character mapping
    :param start_text: Starting text
    :param length: Length of generated text
    :param temperature: Temperature parameter to control randomness of text generation
    :return: Generated text
    """
    model.eval()
    with torch.no_grad():
        # Convert starting text to index sequence
        chars = [char_to_idx.get(ch, 0) for ch in start_text]
        
        # Initialize hidden state
        hidden = None
        
        # Forward propagate each character in the starting text to get the final hidden state
        for ch in chars:
            input_tensor = torch.tensor([[ch]], dtype=torch.long)
            output, hidden = model(input_tensor, hidden)
        
        # Start generating new characters
        result = start_text
        current_char = chars[-1]  # Start from the last character of the starting text
        
        for _ in range(length):
            # Feed current character to model
            input_tensor = torch.tensor([[current_char]], dtype=torch.long)
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature parameter to adjust output distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            
            # Randomly select next character based on output distribution
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add selected character to result
            predicted_char = idx_to_char[top_i.item()]
            result += predicted_char
            
            # Update current character
            current_char = top_i.item()
        
        return result


def load_model_and_generate(model_path, mappings_path, start_text, gen_length=200):
    """
    Load model and generate text
    :param model_path: Model file path
    :param mappings_path: Character mapping file path
    :param start_text: Starting text
    :param gen_length: Length of generated text
    """
    # Load character mappings
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
        char_to_idx = mappings['char_to_idx']
        idx_to_char = mappings['idx_to_char']
        vocab_size = mappings['vocab_size']
    
    # Create model instance
    model = VanillaRNN(vocab_size, 50, 128)  # Parameters must match training
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    
    # Generate text
    generated_text = generate_text(
        model, char_to_idx, idx_to_char, start_text, gen_length, temperature=0.8
    )
    
    print("Starting text:", start_text)
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    # Check if trained model exists
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print("Please run train.py to train the model first!")
        exit()
    
    # Find the latest model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith('model') and f.endswith('.pth')]
    if not model_files:
        print("No trained model found, please run train.py first!")
        exit()
    
    # Select the latest model file
    model_files.sort()
    latest_model = model_files[-1]
    model_path = os.path.join(model_dir, latest_model)
    mappings_path = os.path.join(model_dir, 'char_mappings.pkl')
    
    # Generate text
    start_text = "First Citizen:\n"
    load_model_and_generate(model_path, mappings_path, start_text, gen_length=300)