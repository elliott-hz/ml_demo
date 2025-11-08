import numpy as np


# -----------------------------
# CrossEntropyLoss: Cross-Entropy Loss Function
# -----------------------------
class CrossEntropyLoss:
    """
    Cross-Entropy Loss: commonly used for multi-class classification.
    Combines softmax and cross-entropy calculation for better numerical stability.
    """
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        Forward propagation: compute cross-entropy loss.

        Args:
            y_pred: Model predictions, shape (N, C), after softmax.
            y_true: Ground truth labels, shape (N, C), one-hot encoded.
        Returns:
            Mean loss value.
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # Add a small epsilon to prevent log(0) numerical instability
        epsilon = 1e-10

        # Cross-entropy formula: L = -sum(y_true * log(y_pred)) / N
        return -np.sum(y_true * np.log(y_pred + epsilon)) / y_pred.shape[0]

    def backward(self):
        """
        Backward propagation: compute gradient of the loss w.r.t. model outputs.

        Returns:
            Gradient w.r.t. y_pred, same shape as y_pred.
        """
        # For softmax output with one-hot labels:
        # Gradient = (y_pred - y_true) / N
        # Shape remains (batch_size, num_classes)
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]
