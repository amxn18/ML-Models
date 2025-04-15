import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class LWR:
    def __init__(self, tau=0.5):
        self.tau = tau  # Bandwidth (controls locality)
        self.X_train = None
        self.y_train = None

    def _weights(self, x_query):
        """
        Compute weights using Gaussian kernel
        """
        diff = self.X_train - x_query  # shape: (m, 1)
        weights = np.exp(- (diff ** 2) / (2 * self.tau ** 2))
        return np.diagflat(weights)  # shape: (m, m)

    def fit(self, X, y):
        """
        Store training data
        """
        self.X_train = X.reshape(-1, 1)
        self.y_train = y.reshape(-1, 1)

    def predict(self, X):
        """
        Predict for each x in X using LWR logic
        """
        X = X.reshape(-1, 1)
        m = self.X_train.shape[0]
        y_preds = []

        for x in X:
            W = self._weights(x)
            # θ = (XᵀWX)^-1 XᵀWy
            X_b = np.hstack([np.ones((m, 1)), self.X_train])  # Add bias
            theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ self.y_train
            x_b = np.array([1, x[0]]).reshape(1, 2)
            y_pred = x_b @ theta
            y_preds.append(y_pred[0][0])

        return np.array(y_preds)


# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, size=X.shape)

# Fit sklearn Linear Regression
X_sklearn = X.reshape(-1, 1)
sk_model = LinearRegression()
sk_model.fit(X_sklearn, y)
y_pred_sklearn = sk_model.predict(X_sklearn)


# Fit and predict using LWR
lwr = LWR(tau=0.5)
lwr.fit(X, y)
y_pred_lwr = lwr.predict(X)

# Plot both
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='lightblue', label='Data')
plt.plot(X, y_pred_lwr, color='red', label='Locally Weighted Regression')
plt.plot(X, y_pred_sklearn, color='green', linestyle='--', label='Linear Regression (sklearn)')
plt.legend()
plt.title('Locally Weighted Regression vs Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()
