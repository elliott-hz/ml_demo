from cnn_root_impl.layer_component.layer import Layer


# -----------------------------
# Flatten Layer Class
# -----------------------------
class Flatten(Layer):
    """
    Flatten layer: Converts multi-dimensional feature maps into a 1D.
    Typically used as a transition between convolutional layers and fully connected (dense) layers.
    """
    def __init__(self, name=None):
        super().__init__(name)
        self.name = name or "Flatten"

    def build(self, input_shape):
        """
        Initialize the flatten layer (no parameters)
        """
        self.input_shape_ = input_shape                     # Save input shape
        self.built = True
        return self.output_shape(input_shape)

    def output_shape(self, input_shape):
        """
        Compute the output shape after flattening
        """
        N, h, w, c = input_shape
        return (N, h * w * c)                               # Flatten to (N, h*w*c)

    def forward(self, x, training=False):
        """
        Forward propagation: Flatten the input into a 1D vector

        Args:
            x: Input data, shape (N, h, w, c)
            training: no need to do specific things for training or prediction
        Returns:
            Flattened vector, shape (N, h*w*c)
        """

        # Save original shape for backward propagation
        self.orig_shape = x.shape
        N = x.shape[0]

        # Flatten to (N, -1), -1 means automatically infer the remaining dimension
        return x.reshape(N, -1)

    def backward(self, dout):
        """
        Backward propagation: Reshape gradient to the original input shape

        Args:
            dout: Gradient of the loss with respect to the layer output, shape (N, h*w*c)
        Returns:
            Gradient of the loss with respect to the layer input, same shape as original input
        """
        return dout.reshape(self.orig_shape)

