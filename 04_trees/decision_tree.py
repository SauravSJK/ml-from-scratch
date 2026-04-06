import numpy as np


class DecisionNode:
    """
    A single node in the Decision Tree.
    If value is not None, it is a leaf node. Otherwise, it is a decision node.
    """

    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: "DecisionNode" = None,
        right: "DecisionNode" = None,
        *,
        value: int = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree:
    """
    ID3/C4.5 style Decision Tree Classifier.
    Builds the tree recursively by maximizing Information Gain (reducing Entropy).
    """

    def __init__(
        self, min_samples_split: int = 2, max_depth: int = 100, n_features: int = None
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Builds the decision tree classifier.
        """
        # Determine the maximum number of features to consider for splits
        self.n_features = (
            X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        )
        self.root = self._grow_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given input data.
        """
        if self.root is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # Randomly select features to consider (useful for expanding to Random Forests later)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # If no split improves Information Gain, create a leaf
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return
