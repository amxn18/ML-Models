import numpy as np


class DecisionStump:
    # Simple tree with one split
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        m, n = X.shape
        best_error = float("inf")

        for feature in range(n):
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_pred = y[left_mask].mean()
                right_pred = y[right_mask].mean()

                predictions = np.where(left_mask, left_pred, right_pred)
                error = np.mean((y - predictions) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature = feature
                    self.threshold = threshold
                    self.left_value = left_pred
                    self.right_value = right_pred

    def predict(self, X):
        left_mask = X[:, self.feature] <= self.threshold
        return np.where(left_mask, self.left_value, self.right_value)


class GradientBoostingScratch:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.init_value = None

    def fit(self, X, y):
        # Step 1: initial prediction
        self.init_value = y.mean()
        predictions = np.full_like(y, self.init_value, dtype=float)

        for _ in range(self.n_estimators):
            # Step 2: compute residuals
            residuals = y - predictions

            # Step 3: fit weak learner on residuals
            stump = DecisionStump()
            stump.fit(X, residuals)

            # Step 4: update predictions
            update = stump.predict(X)
            predictions += self.learning_rate * update

            self.models.append(stump)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_value, dtype=float)

        for model in self.models:
            predictions += self.learning_rate * model.predict(X)

        return predictions
