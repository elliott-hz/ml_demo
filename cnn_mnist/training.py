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
    """
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    # Train the model
    model.fit(train_images, train_labels, epochs=5, batch_size=64, 
              validation_data=(test_images, test_labels))
    
    # Save model to local file
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model