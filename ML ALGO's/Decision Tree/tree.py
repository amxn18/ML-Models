import numpy as np


class DecisionTreeScratch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    # -----------------------------
    # Gini Impurity
    # -----------------------------
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    # -----------------------------
    # Split Dataset
    # -----------------------------
    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return (
            X[left_mask], y[left_mask],
            X[right_mask], y[right_mask]
        )

    # -----------------------------
    # Find Best Split
    # -----------------------------
    def _best_split(self, X, y):
        m, n = X.shape
        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        for feature in range(n):
            values = np.unique(X[:, feature])

            if len(values) == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                X_l, y_l, X_r, y_r = self._split(X, y, feature, threshold)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                weighted_gini = (
                    (len(y_l) / m) * self._gini(y_l)
                    + (len(y_r) / m) * self._gini(y_r)
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # -----------------------------
    # Build Tree (Recursive)
    # -----------------------------
    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return {"value": y[0]}

        if depth == self.max_depth:
            return {"value": np.bincount(y).argmax()}

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return {"value": np.bincount(y).argmax()}

        X_l, y_l, X_r, y_r = self._split(X, y, feature, threshold)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X_l, y_l, depth + 1),
            "right": self._build_tree(X_r, y_r, depth + 1),
        }

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    # -----------------------------
    # Predict One Sample
    # -----------------------------
    def _predict_one(self, node, x):
        if "value" in node:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(node["left"], x)
        else:
            return self._predict_one(node["right"], x)

    # -----------------------------
    # Predict Batch
    # -----------------------------
    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])
