# -----------------------------
# Dropout: Dropout Layer Class
# -----------------------------
import numpy as np

from cnn_root_impl.layer_component.layer import Layer


class Dropout(Layer):
    """
    Dropout layer: randomly drops some neurons during training to prevent overfitting.
    During testing, no neurons are dropped, keeping the output unchanged.
    """
    def __init__(self, p=0.5, name=None):
        """
        Initialize the Dropout layer

        Args:
            p: Drop probability, default is 0.5
            name: Layer name
        """
        super().__init__(name)
        self.p = p
        self.name = name or f"Dropout_{p}"

    def forward(self, x, training=False):
        """
        Forward propagation: randomly drop neurons during training

        Args:
            x: Input data
        Returns:
            Output after applying dropout
        """
        if training:
            # During training, randomly "turn off" some neurons (with probability p)
            # Scale the retained neurons by 1/(1-p) to keep the expected output unchanged
            # Generate mask: elements greater than p are kept (keep probability = 1-p)
            self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)

            # Scale retained elements to maintain expected output
            self.mask /= (1.0 - self.p)
            # t = f(x) = x * mask
            return x * self.mask
        else:
            # In test mode, no dropout is applied
            self.mask = None
            return x


    def backward(self, dout):
        """
        Backward propagation: pass gradients only through retained neurons
              t = f(x) = x * mask

        Args:
            dout: Gradient of the loss with respect to the layer output
        Returns:
            Gradient of the loss with respect to the layer input
              dL/dx = dL/dt * dt/dx = dout * dt/dx
        """

        # If no mask exists (test mode), pass gradient directly
        return dout * getattr(self, 'mask', 1.0)
