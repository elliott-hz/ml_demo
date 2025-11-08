import numpy as np

from cnn_root_impl.layer_component.layer import Layer


# -----------------------------
# ReLU: Activation Layer
# -----------------------------
class ReLU(Layer):
    """
    ReLU activation layer: y = max(0, x)
    Introduces non-linearity, alleviates vanishing gradients, and speeds up training.
    """
    def __init__(self, name=None):
        super().__init__(name)
        self.name = name or "ReLU"

    def forward(self, x, training=False):
        """
        Forward propagation: apply ReLU activation function

        Args:
            x: Input data
        Returns:
            Activated output
        """
        if training:
            self.mask = (x > 0)         # Save mask for backward propagation
            return x * self.mask        # Apply ReLU: output x if x > 0, otherwise 0
        else:
            return np.maximum(0, x)     # Directly return result

    def backward(self, dout):
        """
        Backward propagation: compute gradient w.r.t input
              r = f(x) = x * mask

        Args:
            dout: Gradient of the loss with respect to the layer output
        Returns:
            Gradient of the loss with respect to the layer input
              dL/dx = dL/dr * dr/dx = dout * dr/dx
        """
        # ReLU derivative: 1 if x > 0, otherwise 0
        return dout * self.mask
