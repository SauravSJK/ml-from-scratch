import numpy as np


class Linear:
    """Fully Connected (Dense) Layer."""

    def __init__(self, in_features: int, out_features: int):
        # He Initialization for better variance scaling with ReLU
        self.params = {
            "W": np.random.randn(in_features, out_features)
            * np.sqrt(2.0 / in_features),
            "b": np.zeros((1, out_features)),
        }
        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"]),
        }
        self.cache = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache = X
        return np.dot(X, self.params["W"]) + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        X = self.cache
        # Gradients for weights and bias
        self.grads["W"] = np.dot(X.T, dout)
        self.grads["b"] = np.sum(dout, axis=0, keepdims=True)
        # Gradient with respect to input
        dX = np.dot(dout, self.params["W"].T)
        return dX


class Dropout:
    """
    Inverted Dropout Layer.
    Randomly zeroes some of the elements of the input tensor with probability (1 - keep_prob).
    """

    def __init__(self, keep_prob: float = 0.5):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return X

        # Inverted dropout: scale during training so inference requires no scaling
        self.mask = (np.random.rand(*X.shape) < self.keep_prob) / self.keep_prob
        return X * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask
