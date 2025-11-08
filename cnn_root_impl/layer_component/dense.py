# -----------------------------
# Dense: Fully Connected Layer
# -----------------------------
import numpy as np

from cnn_root_impl.layer_component.layer import Layer
from cnn_root_impl.layer_component.weight_initialization import he_initialization


class Dense(Layer):
    """
    Fully connected layer (dense layer): Each neuron is connected to all neurons in the previous layer.
    Used to perform nonlinear combination of extracted features for classification or regression.
    """
    def __init__(self, out_features, in_features=None, name=None):
        """
        Initialize the dense layer.

        Args:
            out_features: Number of output features (neurons)
            in_features: Number of input features, can be inferred automatically, default is None
            name: Layer name
        """
        super().__init__(name)
        self.out_features = out_features
        self.in_features = in_features                # Input feature size, can be inferred in build()
        self.name = name or f"Dense_{out_features}"
        self.W = None                                 # Weights
        self.b = None                                 # Bias
        self.params = []                              # List of parameters (W, b)
        self.grads = []                               # List of gradients

    def build(self, input_shape):
        """
        Initialize weights and bias of the dense layer.

        Args:
            input_shape: Shape of input data (N, in_features)
        Returns:
            Output shape
        """

        # Infer input feature size from input shape if not specified
        if self.in_features is None:
            self.in_features = input_shape[1]

        # Initialize weights using He initialization
        self.W = he_initialization((self.in_features, self.out_features))
        # Initialize bias to zeros
        self.b = np.zeros((self.out_features,), dtype=np.float32)

        # Save parameter and gradient references
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

        self.built = True
        return self.output_shape(input_shape)

    def output_shape(self, input_shape):
        """
        Compute output shape of the dense layer
        """
        return (input_shape[0], self.out_features)

    def forward(self, x, training=False):
        """
        Forward propagation: perform matrix multiplication and add bias

        Args:
            x: Input data, shape (N, in_features)
            training: I dont want to implement specific code for training, maybe in the future ......
        Returns:
            Output of the dense layer, shape (N, out_features)
        """

        if training: self.x = x           # Save input for backward propagation

        if training: pass                 # L2 (in the future) //TODO

        # Dense layer computation: y = x*W + b
        return x.dot(self.W) + self.b

    def backward(self, dout):
        """
        Backward propagation: compute gradients for weights, bias, and input

        Args:
            dout: Gradient of the loss with respect to the layer output, shape (N, out_features)
              z = f(x) = x.dot(self.W) + self.b
        Returns:
            Gradient of the loss with respect to the input, shape (N, in_features)
            Using the chain rule:
              dL/dx = dL/dz * dz/dx
              dL/dW = dL/dz * dz/dW
              dL/db = dL/dz * dz/db
        """
        # 1. Gradient w.r.t input x: output change depends on both input and weights
        # Shape: [N, out_features] * [out_features, in_features] -> [N, in_features]
        dx = dout.dot(self.W.T)

        # 2. Gradient w.r.t weights W: determined by input size and output gradient
        # Shape: [in_features, N] * [N, out_features] -> [in_features, out_features]
        dW = self.x.T.dot(dout)

        # 3. Gradient w.r.t bias b: bias is added directly to output, sum over all samples
        # Shape: [out_features]
        db = np.sum(dout, axis=0)

        # Save gradients to member variables for optimizer to update parameters ([...] to avoid creating a new array)
        self.grads[0][...] = dW
        self.grads[1][...] = db

        # Return dx to continue backpropagation
        return dx
