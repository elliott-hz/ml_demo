
import time

import numpy as np


# -----------------------------
# Sequential: Neural Network Container
# -----------------------------
class Sequential:
    """
    Neural network container: organizes multiple layers in sequence,
    enabling end-to-end forward and backward propagation.
    """
    def __init__(self, layers_list=None):
        """
        Initialize the neural network

        Args:
            layers_list: List of layers in order
        """
        self.layers = layers_list or []

    def build(self, input_shape, show_summary=False):
        """
        Build the network: initialize all layers' parameters

        Args:
            input_shape: Shape of input data (batch_size, ...)
        """
        shape = input_shape
        for layer in self.layers:
            # Call build method of layer to initialize parameters
            if hasattr(layer, 'build'):
                shape = layer.build(shape)
            else:
                # If no build method, compute output shape directly
                shape = layer.output_shape(shape)

        # Save network output shape
        self._output_shape = shape

        if show_summary:
            print(f"\n\nContainer (Sequential) built completed: ")
            print("-" * 50)
            self._summary(input_shape)
            print("-" * 50)

    def forward(self, x, training=False, verbose=False):
        """
        Forward propagation: sequentially pass data through all layers with optional logging/training

        Args:
            x: Input data
            training: Whether in training mode
            verbose: Whether output logs
        Returns:
            Network output
        """
        out = x

        if verbose: print(f"\nInitial input shape: {out.shape}")

        for i, layer in enumerate(self.layers):
            start_time = time.time()                            # Record start time of current layer
            input_shape = out.shape                             # Save input shape for logging
            layer_out = layer.forward(out, training=training)   # Execute forward propagation of current layer
            elapsed = (time.time() - start_time) * 1000         # Compute elapsed time in milliseconds

            # Print log for current layer: include parameter count if applicable
            if verbose:
              p_count = sum(np.prod(p.shape) for p in layer.params) if hasattr(layer, 'params') and layer.params else 0
              print(f"Layer {i+1}: {layer.__class__.__name__}\t| Input Shape: {input_shape}\t|Output Shape: {layer_out.shape}\t|Time: {elapsed:.4f} ms\t|Params: {p_count}")

            # Update output for next layer
            out = layer_out

        if verbose: print(f"Forward propagation finished, output shape: {out.shape}")
        return out

    def backward(self, grad, verbose=False):
        """
        Backward propagation: propagate gradient from output to input with optional logging

        Args:
            grad: Gradient of loss w.r.t network output
        Returns:
            Gradient of loss w.r.t network input
        """
        g = grad
        if verbose: print(f"\nBackward propagation started, initial gradient shape: {g.shape}")

        # Traverse layers in reverse order
        for i, layer in enumerate(reversed(self.layers)):

            start_time = time.time()
            input_grad_shape = g.shape

            # Execute backward propagation of current layer
            g = layer.backward(g)
            elapsed = (time.time() - start_time) * 1000

            # Compute parameter count
            if hasattr(layer, 'params') and layer.params:
                p_count = sum(np.prod(p.shape) for p in layer.params)
            else:
                p_count = 0

            # Print log for current layer
            if verbose:
              print(f"Backward Layer {i+1}: {layer.__class__.__name__}\t| Input grad shape: {input_grad_shape}\t|Output grad shape: {g.shape}\t|Time: {elapsed:.4f} ms\t|Params: {p_count}")

        if verbose: print(f"Backward propagation finished, final gradient shape: {g.shape}")
        return g

    def params_and_grads(self):
        """
        Get all trainable parameters and their gradients

        Returns:
            List of tuples (layer, param, grad)
        """
        lst = []
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for p, g in zip(layer.params, layer.grads):
                    lst.append((layer, p, g))
        return lst

    def _summary(self, input_shape):
        """
        Print network summary: layer type, output shape, parameter count, etc.

        Args:
            input_shape: Shape of input data
        """
        # Build network to determine output shapes
        total_params = 0
        shape = input_shape
        print(f"{'Layer':<6} | {'Type':<20} | {'Input Shape':<20} | {'Output Shape':<20} | {'Params':<8} | Name")
        print("-"*100)

        for i, layer in enumerate(self.layers):
            input_shape_layer = shape
            out_shape = layer.output_shape(shape)             # Compute output shape of current layer
            params_count = 0                                  # Compute number of parameters
            if hasattr(layer, 'params') and layer.params:
                for p in layer.params:
                    params_count += np.prod(p.shape)          # Compute total parameters for the layer
            total_params += params_count

            # Print layer information
            print(f"{i+1:<6} | "
              f"{layer.__class__.__name__:20} | "
              f"{str(input_shape_layer):<20} | "
              f"{str(out_shape):<20} | "
              f"{params_count:<8} | "
              f"{getattr(layer, 'name', None)}")

            shape = out_shape                                 # Update shape for next layer
        print(f"\nTotal parameters: {total_params}")

