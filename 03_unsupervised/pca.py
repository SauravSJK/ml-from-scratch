import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) for Dimensionality Reduction.
    Focuses on maximizing variance via Eigendecomposition of the Covariance Matrix.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X: np.ndarray) -> None:
        """
        Calculates the principal components of the dataset.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute the Covariance Matrix
        # rowvar=False expects data in (samples, features) format
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigendecomposition
        # np.linalg.eigh is used because the covariance matrix is symmetric.
        # It is faster and numerically more stable than np.linalg.eig.
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        # eigh returns eigenvalues in ascending order, so we reverse them
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        # 5. Store the top n_components
        # We transpose the eigenvectors so rows correspond to components
        self.components = eigenvectors[:, : self.n_components].T
        self.explained_variance = eigenvalues[: self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects the data onto the principal component space.
        """
        if self.components is None or self.mean is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        # Ensure new data is centered using the training mean
        X_centered = X - self.mean

        # Project data: (n_samples, n_features) @ (n_features, n_components)
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the model and immediately transforms the data."""
        self.fit(X)
        return self.transform(X)
