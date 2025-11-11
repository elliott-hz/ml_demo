import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


def evaluate_and_report(model, test_ds, x_test=None, y_test=None, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    if x_test is not None and y_test is not None:
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        preds = model.predict(x_test)
        pred_classes = np.argmax(preds, axis=1)
        true = y_test
    else:
        # assume test_ds yields batches
        test_loss, test_acc = model.evaluate(test_ds)
        preds = model.predict(test_ds)
        pred_classes = np.argmax(preds, axis=1)
        # try to gather true labels
        true = []
        for _, y in test_ds:
            true.append(y.numpy())
        true = np.concatenate(true, axis=0)

    print(f'Test accuracy: {test_acc}')
    print('\nClassification report:')
    print(classification_report(true, pred_classes))

    cm = confusion_matrix(true, pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

