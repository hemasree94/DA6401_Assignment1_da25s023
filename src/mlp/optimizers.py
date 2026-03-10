import numpy as np

class Optimizers:
    """
    Optimization Algorithms
    Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1      # used for momentum & first moment
        self.beta2 = beta2      # used for second moment (Adam/RMSProp)
        self.epsilon = epsilon

    # For Momentum / NAG
        self.v = None
        self.v_shape = None

        # For RMSProp
        self.v_rms = None
        self.v_rms_shape = None

        # For Adam / Nadam
        self.m = None
        self.v_adam = None
        self.m_shape = None
        self.t = 0

    # ---------------- SGD ----------------
    def sgd(self, weights, grads):
        return weights - self.lr * grads

    # ---------------- Momentum ----------------
    def momentum(self, weights, grads):
        # Reset state if shape changes (e.g., different layer or weight matrix)
        if self.v is None or self.v.shape != weights.shape:
            self.v = np.zeros_like(weights)
            self.v_shape = weights.shape

        self.v = self.beta1 * self.v - self.lr * grads
        return weights + self.v

    # ---------------- NAG ----------------
    def nag(self, weights, grads):
        # Reset state if shape changes (e.g., different layer or weight matrix)
        if self.v is None or self.v.shape != weights.shape:
            self.v = np.zeros_like(weights)
            self.v_shape = weights.shape

        # Lookahead step
        v_prev = self.v.copy()
        self.v = self.beta1 * self.v - self.lr * grads

        return weights - self.beta1 * v_prev + (1 + self.beta1) * self.v

    # ---------------- RMSProp ----------------
    def rmsprop(self, weights, grads):
        # Reset state if shape changes (e.g., different layer or weight matrix)
        if self.v_rms is None or self.v_rms.shape != weights.shape:
            self.v_rms = np.zeros_like(weights)
            self.v_rms_shape = weights.shape

        self.v_rms = self.beta2 * self.v_rms + (1 - self.beta2) * (grads ** 2)

        return weights - self.lr * grads / (np.sqrt(self.v_rms) + self.epsilon)

    # ---------------- Adam ----------------
    def adam(self, weights, grads):
        # Reset state if shape changes (e.g., different layer or weight matrix)
        if self.m is None or self.m.shape != weights.shape:
            self.m = np.zeros_like(weights)
            self.v_adam = np.zeros_like(weights)
            self.m_shape = weights.shape
            # Reset timestep counter when shape changes
            self.t = 0

        self.t += 1

        # First moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Second moment
        self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v_adam / (1 - self.beta2 ** self.t)

        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    # ---------------- Nadam ----------------
    def nadam(self, weights, grads):
        # Reset state if shape changes (e.g., different layer or weight matrix)
        if self.m is None or self.m.shape != weights.shape:
            self.m = np.zeros_like(weights)
            self.v_adam = np.zeros_like(weights)
            self.m_shape = weights.shape
            # Reset timestep counter when shape changes
            self.t = 0

        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v_adam / (1 - self.beta2 ** self.t)

        nesterov_term = (
            self.beta1 * m_hat +
            (1 - self.beta1) * grads / (1 - self.beta1 ** self.t)
        )

        return weights - self.lr * nesterov_term / (np.sqrt(v_hat) + self.epsilon)