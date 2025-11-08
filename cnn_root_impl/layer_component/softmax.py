import numpy as np

from cnn_root_impl.layer_component.layer import Layer


# -----------------------------
# Softmax: Softmax Activation Layer
# -----------------------------
class Softmax(Layer):
    """
    Softmax activation layer: converts input into a probability distribution.
    Typically used as the output layer for multi-class classification.
    """
    def __init__(self, axis=1, name=None):
        """
        Initialize the Softmax layer

        Args:
            axis: Axis along which to compute softmax, default is 1 (feature axis of each sample)
            name: Layer name
        """
        super().__init__(name)
        self.axis = axis                # Axis to compute softmax
        self.name = name or "Softmax"

    def forward(self, x, training=False):
        """
        Forward propagation: apply the Softmax function

        Steps:
            1. Input x shape is (N, 10), representing N samples with 10 class scores each.
            2. x_max shape is (N, 1), the maximum score of each sample across classes, kept for broadcasting.
            3. e_x shape is (N, 10), exponential of (score - max_score) for numerical stability.
            4. self.probs shape is (N, 10), normalized probabilities summing to 1 for each sample.

        Args:
            x: Input data, shape (N, C), C = number of classes
        Returns:
            Probability distribution, same shape as input
        """

        # Stable softmax: subtract max to prevent numerical overflow
        x_max = np.max(x, axis=self.axis, keepdims=True)

        # Exponentiate and subtract max
        e_x = np.exp(x - x_max)

        # Normalize to obtain probability distribution
        probs = e_x / np.sum(e_x, axis=self.axis, keepdims=True)

        if training:
            self.probs = probs      # Only save probs during training for use in backward
        else:
            self.probs = None       # for saving memory when testing

        return probs

    def backward(self, dout):
        """
        Backward propagation

        Args:
            dout: Gradient of the loss w.r.t the output
        Returns:
            Gradient of the loss with respect to the input

        Note:
            When Softmax is used together with CrossEntropyLoss,
            the gradient w.r.t logits is already computed in Loss.backward(),
            so I simply pass it through here.
        """
        return dout

