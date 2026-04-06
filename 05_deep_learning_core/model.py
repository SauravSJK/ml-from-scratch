import numpy as np


class Sequential:
    """
    A Sequential container to stack layers and execute the training loop.
    """

    def __init__(self, layers: list, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Passes input sequentially through all layers."""
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, dout: np.ndarray) -> None:
        """Propagates the gradient backwards through all layers."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self) -> None:
        """Updates parameters for all applicable layers."""
        for layer in self.layers:
            if hasattr(layer, "params"):
                self.optimizer.update(layer)

    def compute_loss_and_gradient(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> tuple:
        """
        Computes Categorical Cross-Entropy Loss and its gradient.
        Assumes y_pred are probabilities (output of Softmax) and y_true are one-hot encoded.
        """
        n_samples = y_true.shape[0]

        # Clip probabilities to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Loss calculation
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples

        # Gradient calculation for Softmax + Cross Entropy
        dout = (y_pred_clipped - y_true) / n_samples

        return loss, dout
