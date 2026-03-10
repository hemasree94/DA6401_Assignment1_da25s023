import numpy as np

class Layer:
    """
    Fully Connected Layer
    Handles weight initialization, forward pass, and gradient computation
    """

    def __init__(self, input_size, output_size, weight_init="xavier"):

        if weight_init == "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:
            self.W = np.random.randn(input_size, output_size) * 0.01

        self.b = np.zeros((1, output_size))

        # Cache for backward
        self.X = None

        # Gradients (required by autograder)
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, d_out):
        batch_size = self.X.shape[0]

        self.grad_W = (self.X.T @ d_out) / batch_size
        self.grad_b = np.sum(d_out, axis=0, keepdims=True) / batch_size

        dX = d_out @ self.W.T
        return dX