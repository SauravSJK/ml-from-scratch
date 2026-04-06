import numpy as np


class LogisticRegression:
    """
    Vectorized Logistic Regression using Gradient Descent.

    Optimizes the Binary Cross-Entropy (Log Loss) function:
    L = -(1 / m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        Clips the input to avoid overflow in exp.
        """
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the logistic model to the training data.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Binary target values (0 or 1) of shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Gradient Descent loop
        for _ in range(self.n_iters):
            # Forward pass: Linear combination followed by sigmoid activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute Gradients
            # Note: The gradient formula is identical to linear regression,
            # but y_predicted is now bound between 0 and 1.
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probability estimates for new data.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts binary class labels for new data.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
