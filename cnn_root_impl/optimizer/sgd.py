from cnn_root_impl.train.cnn_container import Sequential


# -----------------------------
# SGD: Stochastic Gradient Descent Optimizer
# -----------------------------
class SGD:
    """
    Stochastic Gradient Descent optimizer: updates parameters using gradients.
    Update rule: param = param - learning_rate * grad
    """
    def __init__(self, lr=1e-3):
        """
        Initialize the optimizer.

        Args:
            lr: Learning rate, controls the step size for parameter updates.
        """
        print(f"\n\nStochastic Gradient Descent Optimizer initialized completed: ")
        print("-" * 50)
        print(f"learning rate: {lr}")
        print("-" * 50)

        self.lr = lr                                  # learning rate

    def step(self, model: Sequential):
        """
        Perform one optimization step (parameter update).

        Args:
            model: The model whose parameters will be updated.
        """
        for layer, p, g in model.params_and_grads():

            p[...] = p - self.lr * g                  # Update rule: p = p - lr * g
                                                      # Use [...] to update in-place and avoid creating new arrays

