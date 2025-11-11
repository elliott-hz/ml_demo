import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import seaborn as sns
import os


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


def plot_roc_curves(model, test_images, test_labels, save_dir='results'):
    """
    Plot ROC curves for each class
    
    Parameters:
        model: Trained model
        test_images: Test images
        test_labels: Test labels (one-hot encoded)
        save_dir (str): Directory to save the plot
    """
    # Get predictions
    test_predictions = model.predict(test_images)
    
    # Convert one-hot encoded labels back to class indices
    test_labels_classes = np.argmax(test_labels, axis=1)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Convert labels to binary format for ROC calculation
    test_labels_binary = test_labels
    
    n_classes = test_labels_binary.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_binary[:, i], test_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 
              'brown', 'pink', 'gray', 'olive']
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label='ROC curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    
    # Create directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(filename)
    plt.close()
    print(f"ROC curves saved to {filename}")


def plot_confusion_matrix(model, test_images, test_labels, save_dir='results'):
    """
    Plot confusion matrix
    
    Parameters:
        model: Trained model
        test_images: Test images
        test_labels: Test labels (one-hot encoded)
        save_dir (str): Directory to save the plot
    """
    # Get predictions
    test_predictions = model.predict(test_images)
    
    # Convert one-hot encoded labels back to class indices
    test_labels_classes = np.argmax(test_labels, axis=1)
    test_predictions_classes = np.argmax(test_predictions, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_classes, test_predictions_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(10)],
                yticklabels=[f'Class {i}' for i in range(10)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Create directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")