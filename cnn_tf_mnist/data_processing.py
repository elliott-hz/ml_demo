import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_data(num_train_samples=1000, num_test_samples=1000):
    """
    Load and preprocess MNIST dataset
    
    Parameters:
        num_train_samples (int): Number of training samples, default 1000
        num_test_samples (int): Number of test samples, default 1000
        
    Returns:
        tuple: Preprocessed training and test data
    """
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Limit the amount of data
    train_images = train_images[:num_train_samples]
    train_labels = train_labels[:num_train_samples]
    test_images = test_images[:num_test_samples]
    test_labels = test_labels[:num_test_samples]
    
    # Data preprocessing
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return (train_images, train_labels), (test_images, test_labels)