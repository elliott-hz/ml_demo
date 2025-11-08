# Standard library imports
import numpy as np

# Local application imports
from cnn_root_impl.layer_component.layer import Layer
from cnn_root_impl.layer_component.weight_initialization import weight_initialization


# ==============================================
# Conv2D: 2D Convolutional Layer Implementation
# ==============================================
class Conv2D(Layer):
    """
    2D Convolutional Layer Implementation
    
    This layer performs 2D convolution operations using the im2col optimization technique.
    It supports various weight initialization methods and follows the standard layer interface.
    """

    def __init__(self, out_channels, kernel_size, stride=1, padding=0, in_channels=None, name=None, kernel_initializer='he'):
        """
        Initialize Conv2D layer with specified parameters.

        Args:
            out_channels (int): Number of output channels (number of convolution kernel groups)
                Each group contains convolution kernels equal to the number of input channels
            kernel_size (int or tuple): Dimensions of a single convolution kernel
                Can be an integer for square kernels or tuple (kernel_height, kernel_width)
                Actual complete kernel dimensions: (e.g. 3, 3, input_channels)
            stride (int or tuple): Convolution stride
                Can be an integer or tuple for height and width directions respectively
            padding (int): Padding size, used to control output feature map dimensions
            in_channels (int, optional): Number of input channels
                Can be automatically inferred during build
            name (str, optional): Layer name for identification
            kernel_initializer (str): Weight initialization method
                Options: 'he', 'norm', 'xavier'

        Examples:
            Conv2D(out_channels=32, kernel_size=(3,3), padding=0, in_channels=1, name='conv1')
            Conv2D(out_channels=64, kernel_size=(3,3), padding=0, name='conv2')  # Automatically infer input channels
        """

        super().__init__(name)
        self.db = None
        self.dW = None
        self.b = None
        self.W = None
        self.built = False

        # Ensure kernel_size is in tuple form (spatial dimensions)
        # Example: input 3 → converted to (3,3), meaning single kernel spatial dimensions are 3x3
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # Ensure stride is in tuple form for height and width directions respectively
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        # Input and output channels
        # in_channels can be automatically inferred during build
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Store trainable parameters: [weights W, biases b]
        #   1. initialized in [build]
        #   2. used in computing output in [forward]
        #   3. optimized and updated in optimizer (SGD)
        self.params = []

        # Store parameter gradients: [weight gradients dW, bias gradients db]
        #   1. initialized in [build]
        #   2. calculated and assigned in [backward]
        #   3. used in optimizer (SGD) to compute new parameters
        self.grads = []

        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """
        Initialize weights and biases for the convolutional layer.

        Args:
            input_shape (tuple): Input data shape (N, H, W, C_in) 
                Where N is batch size, H is height, W is width, C_in is input channels
                
        Returns:
            tuple: Output data shape (N, out_h, out_w, out_channels)
        """
        # Get input channels (C_in) from input shape
        in_channels = input_shape[-1]     # Input channels
        # Height and Width of Kernel dimensions
        kh, kw = self.kernel_size         # Kernel dimensions

        # Initialize weights using the specified initialization method
        # Complete weight shape: (out_channels, in_channels, kh, kw)
        # self.W = he_initialization((self.out_channels, in_channels, kh, kw))
        self.W = weight_initialization((self.out_channels, in_channels, kh, kw), self.kernel_initializer)

        # Bias shape: (out_channels,)
        # Each output channel corresponds to 1 bias, initialized to 0
        self.b = np.zeros(self.out_channels, dtype=np.float32)

        # Initialize gradient arrays (same shape as parameters)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Store parameters and gradients in lists
        self.params = [self.W, self.b]
        self.grads = [self.dW, self.db]

        self.built = True
        return self.output_shape(input_shape)

    def im2col(self, x, kh, kw, sh, sw, pad):
        """
        Convert the input tensor into a 2D matrix (columns) for efficient convolution computation.
        
        This implementation uses the im2col technique which reshapes the input tensor into
        a matrix where each row represents a flattened receptive field.

        Args:
            x (numpy.ndarray): Input tensor with shape (N, H, W, C)
                Where N is batch size, H is height, W is width, C is channels
            kh (int): Kernel height
            kw (int): Kernel width
            sh (int): Stride along height
            sw (int): Stride along width
            pad (int): Padding size (same on all sides)

        Returns:
            numpy.ndarray: 2D array with shape (N*out_h*out_w, kh*kw*C)
                Each row corresponds to a flattened receptive field
                (a local region of the input image)
        """
        # Extract dimensions from input tensor
        N, H, W, C = x.shape              # Batch, Height, Width, Channels
        out_h = (H + 2*pad - kh)//sh + 1  # Output height
        out_w = (W + 2*pad - kw)//sw + 1  # Output width

        # Add zero-padding to input (around height & width dimensions)
        x_padded = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode="constant")

        # Initialize a temporary array to collect local patches
        cols = np.zeros((N, out_h, out_w, kh, kw, C), dtype=x.dtype)

        # Extract each sliding window patch and store it
        for y in range(kh):
            y_max = y + sh*out_h
            for x_ in range(kw):
                x_max = x_ + sw*out_w
                # Each slice selects pixels spaced by the stride
                cols[:, :, :, y, x_, :] = x_padded[:, y:y_max:sh, x_:x_max:sw, :]

        # Reshape into (N*out_h*out_w, kh*kw*C)
        # Each row represents one local region flattened
        cols = cols.reshape(N*out_h*out_w, -1)
        return cols

    def col2im(self, cols, x_shape, kh, kw, sh, sw, pad, out_h, out_w):

        """
        Convert column matrix back to the original image tensor.
        
        This is the inverse operation of im2col, used in backward propagation
        to distribute gradients from flattened patches back to their
        original spatial positions.

        Args:
            cols (numpy.ndarray): 2D column matrix with shape (N*out_h*out_w, kh*kw*C)
            x_shape (tuple): Original input shape (N, H, W, C)
            kh (int): Kernel height
            kw (int): Kernel width
            sh (int): Stride along height
            sw (int): Stride along width
            pad (int): Padding size
            out_h (int): Output feature map height
            out_w (int): Output feature map width

        Returns:
            numpy.ndarray: Tensor of shape (N, H, W, C),
                reconstructed (and unpadded) from columns
        """

        # Extract dimensions from input shape
        N, H, W, C = x_shape              # Batch, Height, Width, Channels

        # Temporary padded buffer for accumulating gradients
        x_padded = np.zeros((N, H+2*pad, W+2*pad, C), dtype=cols.dtype)

        # Reshape column matrix back to 6D (for iteration)
        cols_reshaped = cols.reshape(N, out_h, out_w, kh, kw, C)

        # Slide patches back and add overlapping regions
        for y in range(kh):
            y_max = y + sh*out_h
            for x_ in range(kw):
                x_max = x_ + sw*out_w
                x_padded[:, y:y_max:sh, x_:x_max:sw, :] += cols_reshaped[:, :, :, y, x_, :]

        # Remove padding to get final reconstructed input
        if pad > 0:
            return x_padded[:, pad:-pad, pad:-pad, :]
        return x_padded

    def forward(self, x, training=False):
        """
        Forward propagation of convolution using im2col optimization.

        Process:
            1. Convert input into column matrix using im2col
            2. Reshape convolution kernels into rows
            3. Perform matrix multiplication between input patches and kernels
            4. Add bias and reshape result back to NHWC format

        Args:
            x (numpy.ndarray): Input tensor with shape (N, H, W, C)
                Where N is batch size, H is height, W is width, C is channels
            training (bool): Whether in training mode
                If True, cache variables for backward propagation

        Returns:
            numpy.ndarray: Output feature map with shape (N, out_h, out_w, out_channels)
        """

        # Build layer if not already built
        if not self.built:
            self.build(x.shape)

        # Extract dimensions for calculations
        N, H, W, C_in = x.shape      # Input dimensions
        kh, kw = self.kernel_size    # Kernel dimensions
        sh, sw = self.stride         # Stride dimensions

        # Compute output spatial dimensions
        out_h = (H + 2*self.padding - kh)//sh + 1
        out_w = (W + 2*self.padding - kw)//sw + 1

        # Step 1: im2col - flatten input patches
        cols = self.im2col(x, kh, kw, sh, sw, self.padding)     # Shape: (N*out_h*out_w, kh*kw*C_in)

        # Step 2: reshape weights into rows for matrix multiplication
        W_col = self.W.reshape(self.out_channels, -1)           # Shape: (C_out, kh*kw*C_in)

        # Step 3: perform efficient matrix multiplication
        # Each row in cols multiplied by each row in W_col^T
        out = cols @ W_col.T + self.b                           # Shape: (N*out_h*out_w, C_out)

        # Step 4: reshape back to (N, out_h, out_w, C_out)
        out = out.reshape(N, out_h, out_w, self.out_channels)

        # cache variables for backward propagation
        if training:
          self.x = x
          self.cols = cols
          self.out_shape_cache = (out_h, out_w)

        return out

    def backward(self, grad_out):
        """
        Backward propagation for convolution layer (compute gradients).

        Process:
            1. Compute gradient of weights (dW)
            2. Compute gradient of biases (db)
            3. Compute gradient of input (dx) using col2im
            4. Convert back to image shape

        Args:
            grad_out (numpy.ndarray): Gradient of loss with respect to layer output
                Shape (N, out_h, out_w, C_out)
                Where N is batch size, out_h/out_w are output dimensions, C_out is output channels

        Returns:
            numpy.ndarray: Gradient of loss with respect to input
                Shape (N, H, W, C_in)
                Where H/W are input dimensions, C_in is input channels
        """

        # Extract dimensions for calculations
        N, H, W, C_in = self.x.shape      # Input dimensions
        kh, kw = self.kernel_size         # Kernel dimensions
        sh, sw = self.stride              # Stride dimensions
        out_h, out_w = self.out_shape_cache  # Output spatial dimensions
        C_out = self.out_channels         # Output channels

        # Reshape output gradient to 2D matrix: (N*out_h*out_w, C_out)
        grad_out_reshaped = grad_out.reshape(N*out_h*out_w, C_out)

        # Step 1. Compute weight gradient (dW)
        # dW_col = grad_out^T @ input_cols
        # Each output error is multiplied by its corresponding input patch
        dW_col = grad_out_reshaped.T @ self.cols           # Shape: (C_out, kh*kw*C_in)
        self.dW = dW_col.reshape(self.W.shape)

        # Step 2. Compute bias gradient (db)
        # Sum all output gradients over batch and spatial dimensions
        self.db = grad_out_reshaped.sum(axis=0)

        # Step 3. Compute input gradient (dx)
        # dcols = grad_out × W_col
        W_col = self.W.reshape(C_out, -1)
        dcols = grad_out_reshaped @ W_col                  # Shape: (N*out_h*out_w, kh*kw*C_in)

        # Step 4. Convert back to image shape
        dx = self.col2im(dcols, self.x.shape, kh, kw, sh, sw, self.padding, out_h, out_w)

        return dx

    def output_shape(self, input_shape):
        """
        Calculate the output shape of the layer given an input shape.
        
        Args:
            input_shape (tuple): Input data shape (N, H, W, C_in)
                Where N is batch size, H is height, W is width, C_in is input channels
                
        Returns:
            tuple: Output data shape (N, out_h, out_w, out_channels)
        """
        N, H, W, _ = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        out_h = (H + 2*self.padding - kh)//sh + 1  # Calculate output height
        out_w = (W + 2*self.padding - kw)//sw + 1  # Calculate output width
        return (N, out_h, out_w, self.out_channels)

    def params_and_grads(self):
        """
        Get parameter and gradient pairs for optimization.
        
        Returns:
            list: List of tuples containing (parameter, gradient) pairs
                [(weights, weight_gradients), (biases, bias_gradients)]
        """
        return [(self.W, self.dW), (self.b, self.db)]
