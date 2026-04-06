import numpy as np


class ReLU:
    """Rectified Linear Unit activation function."""

    def __init__(self):
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache = X
        return np.maximum(0, X)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self.cache
        # Gradient is 1 where X > 0, else 0
        dX = dout.copy()
        dX[X <= 0] = 0
        return dX


class Softmax:
    """
    Softmax activation function.
    Includes the max-subtraction trick for numerical stability to prevent np.exp overflow.
    """

    def __init__(self):
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        # Shift values to prevent overflow in exp
        shifted_X = X - np.max(X, axis=-1, keepdims=True)
        exps = np.exp(shifted_X)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        self.cache = probs
        return probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Usually, Softmax is combined with Cross-Entropy loss for stability.
        # If called independently, the Jacobian must be computed.
        probs = self.cache
        dX = np.empty_like(dout)
        for i, (p, dp) in enumerate(zip(probs, dout)):
            p = p.reshape(-1, 1)
            # Jacobian matrix of softmax
            jacobian = np.diagflat(p) - np.dot(p, p.T)
            dX[i] = np.dot(jacobian, dp)
        return dX
