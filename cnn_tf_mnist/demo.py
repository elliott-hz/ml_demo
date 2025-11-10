import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to sys.path to enable importing from cnn_tf_mnist package
sys.path.insert(0, os.path.dirname(current_dir))

from cnn_tf_mnist.data_processing import load_and_preprocess_data
from cnn_tf_mnist.model import build_cnn_model
from cnn_tf_mnist.training import train_and_save_model, plot_training_history
from cnn_tf_mnist.evaluation import print_classification_report_func, plot_roc_curves, plot_confusion_matrix

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(current_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Main program flow
if __name__ == "__main__":
    model_path = os.path.join(RESULTS_DIR, 'mnist_cnn_model.h5')
    
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
        (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data(num_train_samples=2000)
        
        # Build model
        model = build_cnn_model()
        
        # Train and save model
        trained_model, history = train_and_save_model(model, 
                                                     (train_images, train_labels), 
                                                     model_path=model_path, validation_split=0.2)
    
    # Plot training history if available
    if history is not None:
        plot_training_history(history, save_to_file=True, filename=os.path.join(RESULTS_DIR, 'training_curves.png'))
    
    # Evaluate model
    test_loss, test_acc = trained_model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc}')
    
    # Plot evaluation graphs
    plot_roc_curves(trained_model, test_images, test_labels, save_dir=RESULTS_DIR)
    plot_confusion_matrix(trained_model, test_images, test_labels, save_dir=RESULTS_DIR)
    
    # Demonstrate loading model and predicting
    sample_image = test_images[:1]  # Take one example for prediction
    prediction = trained_model.predict(sample_image)
    print(f'Prediction result: {prediction.argmax()}')
    
    # Print classification report
    print_classification_report_func(trained_model, test_images, test_labels)