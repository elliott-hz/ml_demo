from cnn_mnist.data_processing import load_and_preprocess_data
from cnn_mnist.model import build_cnn_model
from cnn_mnist.training import train_and_save_model
from cnn_mnist.evaluation import load_model_and_predict, print_classification_report_func


# Main program flow
if __name__ == "__main__":
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Build model
    model = build_cnn_model()
    
    # Train and save model
    trained_model = train_and_save_model(model, 
                                        (train_images, train_labels), 
                                        (test_images, test_labels))
    
    # Evaluate model
    test_loss, test_acc = trained_model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
    
    # Demonstrate loading model and predicting
    sample_image = test_images[:1]  # Take one example for prediction
    prediction = load_model_and_predict('mnist_cnn_model.h5', sample_image)
    print(f'Prediction result: {prediction.argmax()}')
    
    # Print classification report
    print_classification_report_func(trained_model, test_images, test_labels)