import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize


class CNN_Evaluation:

    def __init__(self):
        pass

    # Evaluates the trained model on the test set, producing the model's predictions and comparing them with the true labels.
    def predict_on_test(self, model, x_test, y_test):
        print("\n\nStart Prediction on test dataset:")
        print("-" * 50)
        print(f"x_test shape: {x_test.shape}")
        print("-" * 50)

        # Performs forward propagation using the model.forward() method to get predictions on the test set
        test_logits = model.forward(x_test, training=False, verbose=False)
        test_preds = np.argmax(test_logits, axis=1)
        test_labels = np.argmax(y_test, axis=1)
        print("\nPrediction Completed")

        return test_logits, test_preds, test_labels

    # =========================
    # Plot loss & accuracy curves
    # =========================
    def loss_accuracy_curve(self, history):
        # Loss Curve: Plots both training and validation loss over epochs.
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy Curve: Plots both training and validation accuracy over epochs.
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label="Train Acc")
        plt.plot(history['val_acc'], label="Val Acc")
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    # =========================
    # Confusion matrix
    # =========================
    # The function show_confusion_matrix() is used to visualize the performance of the classification model by plotting a confusion matrix.
    def show_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # =========================
    # Classification report
    # =========================
    def show_classification_report(self, labels, preds):
        print("\nClassification Report:")
        print(classification_report(labels, preds))

    # =========================
    # ROC Curve + AUC
    # =========================
    def show_ROC_Curves(self, test_labels, y_test, test_logits):
        y_true_bin = label_binarize(test_labels, classes=np.arange(y_test.shape[1]))
        fpr, tpr, roc_auc = {}, {}, {}
        plt.figure(figsize=(6, 5))
        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], test_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve (One-vs-Rest)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()