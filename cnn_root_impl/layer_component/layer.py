# ======================================================
# Step 2.2: Layer Abstract Base Class
# ======================================================
class Layer:
    """
    Abstract base class for neural network layers.

    This class defines the common interface that all neural network layers should implement.
    It provides the basic structure and essential methods that are common to all layer_component types.

    Subclasses must implement these core methods:
      1. forward() - Forward propagation through the layer_component
      2. backward() - Backward propagation (gradient computation)

    Attributes:
        built (bool): Flag indicating if the layer_component parameters have been initialized
        name (str): Layer name for debugging and identification
    """

    def __init__(self, name=None):
        """
        Initialize the Layer base class.

        Args:
            name (str, optional): Name of the layer_component for debugging purposes
        """
        # Flag indicating if the layer_component parameters have been initialized
        self.built = False
        # Layer name, used for debugging and printing summaries
        self.name = name

    def build(self, input_shape):
        """
        Initialize layer_component parameters (such as weights and biases).

        This method is called once before the first forward pass to initialize
        layer_component parameters based on the input shape. Subclasses should override
        this method to implement layer_component-specific initialization logic.

        Args:
            input_shape (tuple): Shape of input data (batch_size, ...)

        Returns:
            tuple: Output data shape
        """
        self.built = True
        # By default, return the input shape (identity transformation)
        return input_shape

    def forward(self, x, training=False):
        """
        Forward propagation: Compute the output of the layer_component.

        This method computes the forward pass through the layer_component, transforming
        the input data to produce the output. All subclasses must implement this method.

        Args:
            x (numpy.ndarray): Input data tensor
            training (bool): Boolean flag indicating whether in training mode.
                           Used for operations specific to training like dropout.

        Returns:
            numpy.ndarray: Output of the layer_component

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def backward(self, grad):
        """
        Backward propagation: Compute gradients for backpropagation.

        This method performs backward propagation through the layer_component, computing:
          1. Gradients w.r.t. layer_component parameters (weights and biases) for optimization
          2. Gradients w.r.t. layer_component input for backpropagation to previous layer_component

        All subclasses must implement this method.

        Args:
            grad (numpy.ndarray): Gradient of loss function w.r.t the current layer_component's output

        Returns:
            numpy.ndarray: Gradient of loss function w.r.t the current layer_component's input

        Note:
            This method should also store gradients w.r.t the layer_component's weights and biases
            as instance attributes for use in the optimization step.

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement the backward method")

    def output_shape(self, input_shape):
        """
        Calculate the output shape of the layer_component.

        This method computes the shape of the output tensor given the input shape.
        It is useful for building and validating network architectures.

        Args:
            input_shape (tuple): Shape of input data (batch_size, ...)

        Returns:
            tuple: Output data shape (batch_size, ...)
        """
        # By default, returns input shape, suitable for layers like ReLU, Softmax
        # that don't change the spatial dimensions or number of channels
        return input_shape