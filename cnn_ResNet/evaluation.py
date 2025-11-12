"""Evaluation utilities for cnn_ResNet.
Provides simple evaluation and reporting functions.
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, dataset):
    # Collect predictions and labels from dataset
    y_true = []
    y_pred = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.argmax(axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm

