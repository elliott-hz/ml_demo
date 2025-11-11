import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


def evaluate_and_report(model, x_test, y_test, results_dir='cnn_CIFAR10_AlexNet/results'):
    os.makedirs(results_dir, exist_ok=True)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc}')

    preds = model.predict(x_test)
    pred_classes = np.argmax(preds, axis=1)

    print('\nClassification report:')
    print(classification_report(y_test, pred_classes))

    cm = confusion_matrix(y_test, pred_classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

