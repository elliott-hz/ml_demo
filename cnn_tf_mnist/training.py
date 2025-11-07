import matplotlib.pyplot as plt
import os
from tensorflow.keras import models


def train_and_save_model(model, train_data, test_data, model_path='mnist_cnn_model.h5'):
    """
    Train model and save to local file
    
    Parameters:
        model: Keras model instance
        train_data: Training data
        test_data: Test data
        model_path (str): Model save path
        
    Returns:
        model: Trained model
        history: Training history
    """
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # Train the model
    history = model.fit(train_images, train_labels, epochs=5, batch_size=64, 
                        validation_data=(test_images, test_labels))
    
    # Create directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Save model to local file
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history


def plot_training_history(history, save_to_file=False, filename='training_curves.png'):
    """
    Plot training & validation accuracy and loss curves
    
    Parameters:
        history: Training history object returned by model.fit()
        save_to_file (bool): Whether to save the plot to a file instead of displaying it
        filename (str): Filename to save the plot if save_to_file is True
    """
    # Check if running in an environment that supports plotting
    if 'DISPLAY' not in os.environ and not save_to_file:
        print("Plotting disabled: No display environment detected. "
              "To save plots to files, call this function with save_to_file=True")
        return
    
    # Create directory if it doesn't exist
    if save_to_file:
        plot_dir = os.path.dirname(filename)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig(filename)
        print(f"Training curves saved to {filename}")
        plt.close()
    else:
        plt.show()