import numpy as np

class Activations:
    """
    Activation Functions and Their Derivatives
    Implements: ReLU, Sigmoid, Tanh, Softmax
    """

    # ---------- ReLU ----------
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # ---------- Sigmoid ----------
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # ---------- Tanh ----------
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    # ---------- Softmax ----------
    def softmax(self, x):
        # stable softmax
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def softmax_derivative(self, x):
        """
        Returns Jacobian matrix for softmax (used rarely in full form).
        For most NN implementations, we combine softmax + cross-entropy,
        so this derivative is not explicitly used.
        """
        s = self.softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)