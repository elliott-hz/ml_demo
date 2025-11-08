
import numpy as np

from cnn_root_impl.layer_component.layer import Layer


# ==============================================
# MaxPool2D: Max pooling layer
# ==============================================
class MaxPool2D(Layer):
    """
    Max pooling layer implementation that reduces spatial dimensions 
    of feature maps while preserving the most important features.
    
    This implementation optimizes pooling operations by extracting patch views
    for each (ph, pw) location and performing max/argmax operations across 
    the patch dimension. It loops over the pooling window dimensions (ph, pw)
    which are typically small, improving efficiency for common use cases.
    """
    def __init__(self, pool_size=2, stride=None, name=None):
        """
        Initialize the max pooling layer.

        Args:
            pool_size (int or tuple): Pooling window size
                Can be an integer for square windows or tuple (height, width)
            stride (int or tuple, optional): Stride for pooling operation
                Defaults to pool_size if not specified
            name (str, optional): Layer name for identification
        """
        super().__init__(name)
        # Ensure pool_size is in tuple form (height, width)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        # stride defaults to pool_size if not specified
        self.stride = stride or pool_size
        # Set layer name
        self.name = name or f"MaxPool_{self.pool_size}"

    def build(self, input_shape):
        """
        Build max pooling layer.
        
        This method marks the layer as built since max pooling layers
        do not have trainable parameters.
        
        Args:
            input_shape (tuple): Shape of input data (N, H, W, C)
                Where N is batch size, H is height, W is width, C is channels
                
        Returns:
            tuple: Output shape (N, out_H, out_W, C)
        """
        self.built = True
        return self.output_shape(input_shape)

    def forward(self, x, training=False):
        """
        Forward propagation: perform max pooling operation.

        Args:
            x (numpy.ndarray): Input data with shape (N, H, W, C)
                Where N is batch size, H is height, W is width, C is channels
            training (bool): Whether the layer is in training mode
                Affects whether indices are stored for backward pass
                
        Returns:
            numpy.ndarray: Pooled output with shape (N, out_H, out_W, C)
        """
        N, out_h, out_w, C = self.output_shape(x.shape)               # output shape: (N, out_h, out_w, C)
        ph, pw = self.pool_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        # Prepare container for patch windows: (N, out_h, out_w, ph, pw, C)
        patches = np.empty((N, out_h, out_w, ph, pw, C), dtype=x.dtype)

        # Fill patches by slicing the input (only loops over ph, pw which are typically small)
        for y in range(ph):
            y_max = y + sh * out_h
            for x_ in range(pw):
                x_max = x_ + sw * out_w
                # slice: x[:, y:y_max:sh, x_:x_max:sw, :] -> shape (N, out_h, out_w, C)
                patches[:, :, :, y, x_, :] = x[:, y:y_max:sh, x_:x_max:sw, :]

        # Flatten pool spatial dims (ph*pw) so we can do max/argmax along that axis
        patches_flat = patches.reshape(N, out_h, out_w, ph * pw, C)   # (N, out_h, out_w, ph*pw, C)

        # Compute max values (forward output)
        out = patches_flat.max(axis=3)                                # (N, out_h, out_w, C)

        if training:
            # Compute argmax indices (to record single max position per patch/channel)
            idx = patches_flat.argmax(axis=3)                         # (N, out_h, out_w, C)

            # Build boolean mask of shape (N, out_h, out_w, ph*pw, C) indicating which element in each patch is max.
            phpw = ph * pw
            # idx.flatten() -> length = N*out_h*out_w*C
            idx_flat = idx.reshape(-1)
            # One-hot along ph*pw dimension (dtype=bool) -> shape (N*out_h*out_w*C, ph*pw)
            mask_flat = np.eye(phpw, dtype=bool)[idx_flat]
            # reshape back to (N, out_h, out_w, ph*pw, C)
            mask = mask_flat.reshape(N, out_h, out_w, phpw, C)
            # reshape to (N, out_h, out_w, ph, pw, C)
            mask = mask.reshape(N, out_h, out_w, ph, pw, C)

            # Build full-size boolean index map self.max_idx with shape like input x
            self.max_idx = np.zeros_like(x, dtype=bool)
            for y in range(ph):
                y_max = y + sh * out_h
                for x_ in range(pw):
                    x_max = x_ + sw * out_w
                    # mask[:, :, :, y, x_, :] has shape (N, out_h, out_w, C)
                    self.max_idx[:, y:y_max:sh, x_:x_max:sw, :] = mask[:, :, :, y, x_, :]

            # Save input shape & dtype for backward pass
            self._input_shape = x.shape
            self._input_dtype = x.dtype
        else:
            self.max_idx = None   # not needed during inference
        return out

    def backward(self, dout):
        """
        Backward propagation: compute gradient with respect to the input.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to layer's output
                Shape (N, out_H, out_W, C_out)
                Where N is batch size, out_H/out_W are output dimensions, C_out is output channels
                
        Returns:
            numpy.ndarray: Gradient of the loss with respect to layer's input
                Same shape as input x (N, H, W, C)
        """
        # Extract dimensions for gradient computation
        N, out_h, out_w, C = dout.shape
        ph, pw = self.pool_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)

        # Initialize input gradient with original input shape & dtype
        dx = np.zeros(self._input_shape, dtype=self._input_dtype)

        # Scatter dout to the positions of maxima using the mask
        # Only loops over small pooling window dimensions (ph, pw)
        for y in range(ph):
            y_max = y + sh * out_h
            for xx in range(pw):
                x_max = xx + sw * out_w
                mask_patch = self.max_idx[:, y:y_max:sh, xx:x_max:sw, :]   # (N, out_h, out_w, C)
                # Add gradient only to the positions that were maxima
                dx[:, y:y_max:sh, xx:x_max:sw, :] += mask_patch * dout

        return dx

    def output_shape(self, input_shape):
        """
        Compute the output shape of the max pooling layer.
        
        Args:
            input_shape (tuple): Shape of input data (N, H, W, C)
                Where N is batch size, H is height, W is width, C is channels
                
        Returns:
            tuple: Output shape (N, out_H, out_W, C)
        """
        # Extract dimensions from input shape
        N, h, w, c_in = input_shape
        ph, pw = self.pool_size         # pooling window height & width
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        out_h = (h - ph) // sh + 1      # output height
        out_w = (w - pw) // sw + 1      # output width
        c_out = c_in                    # output channels = input channels
        return (N, out_h, out_w, c_out)
