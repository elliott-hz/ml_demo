import matplotlib.pyplot as plt
import os
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_and_save_model(model, train_data, model_path='mnist_cnn_model.h5', validation_split=0.2, 
                         early_stopping=True, patience=5, min_delta=1e-4,
                         reduce_lr=True, lr_factor=0.5, lr_patience=3, lr_min=1e-6, lr_monitor='val_loss'):
    """
    Train model and save to local file
    
    Parameters:
        model: Keras model instance
        train_data: Training data (images, labels)
        model_path (str): Model save path
        validation_split (float): Fraction of training data to use for validation
        early_stopping (bool): Whether to use early stopping
        patience (int): Number of epochs with no improvement after which training will be stopped
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
        reduce_lr (bool): Whether to use ReduceLROnPlateau to reduce learning rate on plateau
        lr_factor (float): Factor by which to reduce the learning rate. new_lr = lr * factor
        lr_patience (int): Number of epochs with no improvement before reducing learning rate
        lr_min (float): Lower bound on the learning rate
        lr_monitor (str): Quantity to be monitored for ReduceLROnPlateau

    Returns:
        model: Trained model
        history: Training history
    """
    train_images, train_labels = train_data
    
    # Split training data to create validation set
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=validation_split, random_state=42)
    
    # Define callbacks
    callbacks = []
    if early_stopping:
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, 
                                   min_delta=min_delta, restore_best_weights=True)
        callbacks.append(early_stop)

    # Add ReduceLROnPlateau if requested
    if reduce_lr:
        reduce_lr_cb = ReduceLROnPlateau(monitor=lr_monitor, factor=lr_factor, patience=lr_patience,
                                         min_lr=lr_min, verbose=1)
        callbacks.append(reduce_lr_cb)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=25, batch_size=256,
                        validation_data=(val_images, val_labels), callbacks=callbacks)
    
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