import numpy as np
from tensorflow.keras import models
from sklearn.metrics import classification_report


def load_model_and_predict(model_path, test_sample):
    """
    Load saved model and make predictions
    
    Parameters:
        model_path (str): Model file path
        test_sample: Data sample to predict
        
    Returns:
        prediction: Prediction result
    """
    # Load model
    loaded_model = models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Make predictions
    prediction = loaded_model.predict(test_sample)
    return prediction


def print_classification_report_func(model, test_images, test_labels):
    """
    Print classification report for the model on test dataset
    
    Parameters:
        model: Trained model
        test_images: Test images
        test_labels: Test labels (one-hot encoded)
    """
    # Get predictions
    test_predictions = model.predict(test_images)
    
    # Convert one-hot encoded labels back to class indices
    test_labels_classes = np.argmax(test_labels, axis=1)
    test_predictions_classes = np.argmax(test_predictions, axis=1)
    
    # Generate and print classification report
    report = classification_report(test_labels_classes, test_predictions_classes, 
                                 target_names=[f'Class {i}' for i in range(10)])
    print('\nClassification Report:')
    print(report)