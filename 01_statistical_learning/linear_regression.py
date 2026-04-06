import numpy as np


class LinearRegression:
    """
    Vectorized Linear Regression using Gradient Descent.

    Optimizes the Mean Squared Error (MSE) loss function:
    J(w, b) = (1 / 2m) * sum((X @ w + b - y)^2)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent loop
        for _ in range(self.n_iters):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute Gradients
            # dw = (1/m) * X.T @ (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        return np.dot(X, self.weights) + self.bias
