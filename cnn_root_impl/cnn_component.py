import numpy as np


# ======================================================
# Step 2.1: Weight Initialization Functions
# ======================================================


def compute_fan_in_and_fan_out(shape):
    """
    Compute the number of input units (fan_in) and output units (fan_out)
    for convolutional layers and fully connected (dense) layers.
    These values are used in various weight initialization methods.

    For convolutional layers:
        fan_in = in_channels * kernel_height * kernel_width
        fan_out = out_channels * kernel_height * kernel_width
    
    For fully connected layers:
        fan_in = input_features
        fan_out = output_features

    Args:
        shape: Shape of the weight tensor.
               For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
               For dense layers: (input_features, output_features)
    
    Returns:
        tuple: (fan_in, fan_out) - Number of input and output units
    
    Raises:
        TypeError: If shape length is not 2 or 4
    """
    # Handle convolutional layer weights: shape = (out_channels, in_channels, kernel_height, kernel_width)
    if len(shape) == 4:
        out_channels, in_channels, kernel_height, kernel_width = shape
        # fan_in = number of input channels × kernel receptive field area
        fan_in = in_channels * kernel_height * kernel_width
        # fan_out = number of output channels × kernel receptive field area
        fan_out = out_channels * kernel_height * kernel_width

    # Handle fully connected layer (dense) weights: shape = (in_features, out_features)
    elif len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    else:
        raise TypeError(
            "Weight tensor shape must have 2 or 4 dimensions, got shape with {} dimensions".format(len(shape)))
    return fan_in, fan_out


def he_initialization(shape):
    """
    He Initialization: Initialize weights according to He et al. (2015).
    Suitable for networks with ReLU activation, helping to mitigate the vanishing gradient problem.
    
    He initialization draws weights from a normal distribution with:
        mean = 0
        std = sqrt(2 / fan_in)
    
    This initialization is particularly effective for ReLU activation functions
    as it helps maintain the variance of activations and gradients across layers.

    Args:
        shape: Shape of the weight tensor.
               For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
               For dense layers: (input_features, output_features)
    
    Returns:
        numpy.ndarray: Weights initialized following 'He' distribution.
    
    Raises:
        TypeError: If shape is not a tuple, list, or ndarray
        ValueError: If fan_in is invalid (<= 0)
    """

    # Validate input shape type
    if not isinstance(shape, (tuple, list, np.ndarray)):
        raise TypeError("Weight tensor shape must be a tuple, list, or numpy array")

    # Compute fan_in for 'He' initialization
    fan_in, _ = compute_fan_in_and_fan_out(shape)

    # Validate fan_in value
    if fan_in <= 0:
        raise ValueError(
            f"Invalid number of input units for He initialization: {fan_in}. "
            f"Check the weight shape {shape}.")

    # He initialization uses standard deviation = sqrt(2 / fan_in)
    std = np.sqrt(2.0 / fan_in)

    # Generate samples from normal distribution N(0, std^2)
    return np.random.randn(*shape) * std


def xavier_initialization(shape):
    """
    Xavier (Glorot) Initialization: Initialize weights according to Glorot & Bengio (2010).
    Suitable for networks with Sigmoid or Tanh activation functions.
    
    Xavier's initialization draws weights from a normal distribution with:
        mean = 0
        std = sqrt(2 / (fan_in + fan_out))
    
    This initialization helps maintain the variance of activations and gradients 
    across layers, making it particularly effective for Sigmoid and Tanh activations.

    Args:
        shape: Shape of the weight tensor.
               For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
               For dense layers: (input_features, output_features)
    
    Returns:
        numpy.ndarray: Weights initialized following the Xavier distribution.
    
    Raises:
        TypeError: If shape is not a tuple, list, or ndarray
        ValueError: If fan_in or fan_out are invalid (<= 0)
    """
    # Validate input shape type
    if not isinstance(shape, (tuple, list, np.ndarray)):
        raise TypeError("Weight tensor shape must be a tuple, list, or numpy array")

    # Compute fan_in and fan_out for Xavier initialization
    fan_in, fan_out = compute_fan_in_and_fan_out(shape)

    # Validate fan_in and fan_out values
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(
            f"Invalid number of input or output units for Xavier initialization: "
            f"fan_in={fan_in}, fan_out={fan_out}. Check the weight shape {shape}.")

    # Xavier initialization uses standard deviation = sqrt(2 / (fan_in + fan_out))
    std = np.sqrt(2.0 / (fan_in + fan_out))

    # Generate samples from normal distribution N(0, std^2)
    return np.random.randn(*shape) * std


def norm_initialization(shape):
    """
    Normal (Gaussian) Initialization with small standard deviation.
    
    This initialization draws weights from a normal distribution with:
        mean = 0
        std = 0.01 (small fixed value)
    
    Suitable for cases where neither He nor Xavier initialization is appropriate.

    Args:
        shape: Shape of the weight tensor.
               For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
               For dense layers: (input_features, output_features)
    
    Returns:
        numpy.ndarray: Weights initialized from a normal distribution with small std.
    
    Raises:
        TypeError: If shape is not a tuple, list, or ndarray
        ValueError: If fan_in is invalid (<= 0)
    """
    # Validate input shape type
    if not isinstance(shape, (tuple, list, np.ndarray)):
        raise TypeError("Weight tensor shape must be a tuple, list, or numpy array")

    # Compute fan_in (for consistency check)
    fan_in, _ = compute_fan_in_and_fan_out(shape)

    # Validate fan_in value
    if fan_in <= 0:
        raise ValueError(
            f"Invalid number of input units for normal initialization: {fan_in}. "
            f"Check the weight shape {shape}.")

    # Use a small fixed standard deviation
    std = 0.01

    # Generate samples from normal distribution N(0, std^2)
    return np.random.randn(*shape) * std


def weight_initialization(shape, kernel_initializer='he'):
    """
    Weight initialization function that supports multiple initialization methods.
    
    This function serves as a factory method that selects the appropriate 
    weight initialization technique based on the provided kernel_initializer parameter.
    
    Supported initialization methods:
      'he': He initialization (best for ReLU activations)
      'xavier': Xavier/Glorot initialization (best for Sigmoid/Tanh activations)
      other: Normal initialization with small standard deviation

    Args:
        shape: Shape of the weight tensor.
               For conv layers: (out_channels, in_channels, kernel_h, kernel_w)
               For dense layers: (input_features, output_features)
        kernel_initializer: Initialization method ('he', 'xavier', or other for norm)
    
    Returns:
        numpy.ndarray: Initialized weights according to the selected method.
    """
    # Select initialization method based on kernel_initializer parameter
    if kernel_initializer == 'he':
        # He initialization - best for ReLU activations
        return he_initialization(shape)
    elif kernel_initializer == 'xavier':
        # Xavier's initialization - best for Sigmoid/Tanh activations
        return xavier_initialization(shape)
    else:
        # Normal initialization with small standard deviation - fallback option
        return norm_initialization(shape)


# ======================================================
# Step 2.2: Layer Abstract Base Class
# ======================================================
class Layer:
    """
    Abstract base class for neural network layers.
    
    This class defines the common interface that all neural network layers should implement.
    It provides the basic structure and essential methods that are common to all layer types.
    
    Subclasses must implement these core methods:
      1. forward() - Forward propagation through the layer
      2. backward() - Backward propagation (gradient computation)
    
    Attributes:
        built (bool): Flag indicating if the layer parameters have been initialized
        name (str): Layer name for debugging and identification
    """

    def __init__(self, name=None):
        """
        Initialize the Layer base class.
        
        Args:
            name (str, optional): Name of the layer for debugging purposes
        """
        # Flag indicating if the layer parameters have been initialized
        self.built = False
        # Layer name, used for debugging and printing summaries
        self.name = name

    def build(self, input_shape):
        """
        Initialize layer parameters (such as weights and biases).
        
        This method is called once before the first forward pass to initialize
        layer parameters based on the input shape. Subclasses should override
        this method to implement layer-specific initialization logic.

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
        Forward propagation: Compute the output of the layer.
        
        This method computes the forward pass through the layer, transforming
        the input data to produce the output. All subclasses must implement this method.

        Args:
            x (numpy.ndarray): Input data tensor
            training (bool): Boolean flag indicating whether in training mode.
                           Used for operations specific to training like dropout.
        
        Returns:
            numpy.ndarray: Output of the layer
        
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement the forward method")

    def backward(self, grad):
        """
        Backward propagation: Compute gradients for backpropagation.
        
        This method performs backward propagation through the layer, computing:
          1. Gradients w.r.t. layer parameters (weights and biases) for optimization
          2. Gradients w.r.t. layer input for backpropagation to previous layer
        
        All subclasses must implement this method.

        Args:
            grad (numpy.ndarray): Gradient of loss function w.r.t the current layer's output
        
        Returns:
            numpy.ndarray: Gradient of loss function w.r.t the current layer's input
        
        Note:
            This method should also store gradients w.r.t the layer's weights and biases
            as instance attributes for use in the optimization step.
        
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement the backward method")

    def output_shape(self, input_shape):
        """
        Calculate the output shape of the layer.
        
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
