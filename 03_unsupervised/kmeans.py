import numpy as np


class KMeans:
    """
    Vectorized K-Means Clustering implementation.
    Optimizes the Within-Cluster Sum of Squares (WCSS).
    """

    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        """
        Computes the centroids using Lloyd's algorithm.

        Args:
            X: Training data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # 1. Randomly initialize centroids from existing data points
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            # 2. Assign clusters: Find the closest centroid for each point
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # 3. Update centroids: Calculate the mean of points in each cluster
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                # Handle empty clusters to prevent NaN errors
                if np.any(labels == i):
                    new_centroids[i] = X[labels == i].mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            # 4. Check for convergence
            # If the centroids haven't moved beyond the tolerance threshold, break.
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the closest cluster each sample in X belongs to.
        """
        if self.centroids is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Highly optimized distance calculation using NumPy broadcasting.
        X shape: (N, D) -> reshaped to (N, 1, D)
        Centroids shape: (K, D) -> reshaped to (1, K, D)
        Resulting broadcast shape: (N, K, D)
        """
        # Calculate Euclidean distance without a slow double for-loop
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
