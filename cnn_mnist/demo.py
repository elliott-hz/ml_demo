import os
import matplotlib.pyplot as plt
from tensorflow.keras import models
from cnn_mnist.data_processing import load_and_preprocess_data
from cnn_mnist.model import build_cnn_model
from cnn_mnist.training import train_and_save_model, plot_training_history
from cnn_mnist.evaluation import print_classification_report_func


# Main program flow
if __name__ == "__main__":
    model_path = 'mnist_cnn_model.h5'
    
    # Check if model file exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        # Load and preprocess data (only test data needed for evaluation)
        (_, _), (test_images, test_labels) = load_and_preprocess_data()
        
        # Load existing model
        trained_model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        history = None
    else:
        print("Training new model...")
        # Load and preprocess data
        (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
        
        # Build model
        model = build_cnn_model()
        
        # Train and save model
        trained_model, history = train_and_save_model(model, 
                                                     (train_images, train_labels), 
                                                     (test_images, test_labels))
    
    # Plot training history if available
    if history is not None:
        plot_training_history(history, save_to_file=True, filename='training_curves.png')
    
    # Evaluate model
    test_loss, test_acc = trained_model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc}')
    
    # Demonstrate loading model and predicting
    sample_image = test_images[:1]  # Take one example for prediction
    prediction = trained_model.predict(sample_image)
    print(f'Prediction result: {prediction.argmax()}')
    
    # Print classification report
    print_classification_report_func(trained_model, test_images, test_labels)