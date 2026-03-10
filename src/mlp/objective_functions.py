import numpy as np

class Losses:
    """
    Loss/Objective Functions and Their Derivatives
    Implements: Cross-Entropy, Mean Squared Error (MSE)
    """

    # ---------------- MSE ----------------
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        return (2 / y_true.shape[0]) * (y_pred - y_true)

    # ---------------- Cross Entropy ----------------
    def cross_entropy(self, y_true, y_pred):
        # Add epsilon for numerical stability
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def cross_entropy_derivative(self, y_true, y_pred):
        """
        When using Softmax in output layer,
        derivative simplifies to (y_pred - y_true) / batch_size
        """
        return (y_pred - y_true) / y_true.shape[0]