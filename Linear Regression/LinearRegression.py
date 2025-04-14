import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linearRegression:
    def __init__(self, learningRate, iterations):
        self.learningRate = learningRate
        self.iterations = iterations

    def fit(self, x, y):
        # m --> No of data points  and n --> no of features 
        self.m, self.n = x.shape  #(30x1)
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        for _ in range(self.iterations):
            self.updateWeights()

    def updateWeights(self):
        yPrediction = self.predict(self.x)
        dw = - (2 * (self.x.T).dot(self.y - yPrediction)) / self.m
        db = - (2 * np.sum(self.y - yPrediction)) / self.m

        # Gradient Descent  --> w = w - learningRate * dw
        # Gradient Descent  --> b = b - learningRate * db
        self.w = self.w - self.learningRate * dw
        self.b = self.b - self.learningRate * db

    def predict(self, x):
        # x is a 2D array (m x n) and w is a 1D array (n x 1)
        # y = wx + b (this y gos to updateWeights function)
        return x.dot(self.w) + self.b

# Testing
df = pd.read_csv("Linear Regression/data.csv")
X = df[['YearsExperience']].values
y = df['Salary'].values

model = linearRegression(learningRate=0.01, iterations=1000)
model.fit(X, y)
y_pred_custom = model.predict(X)

print("Custom Model Predictions:", y_pred_custom)
print("Custom Model Weights:", model.w)
print("Custom Model Bias:", model.b)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_custom, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression using Custom Implementation')
plt.legend()
plt.show()


# Salary=9501×YearsExperience+25448

# Comparing With Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Sklearn Linear Regression Model
sk_model = LinearRegression()
sk_model.fit(X, y)
y_pred_sklearn = sk_model.predict(X)

print("\nSklearn Model Predictions:", y_pred_sklearn)
print("Sklearn Model Coefficients (Weight):", sk_model.coef_)
print("Sklearn Model Intercept (Bias):", sk_model.intercept_)


# Evaluation Metrics
print("\n--- Evaluation ---")
print("Custom Model R² Score:", r2_score(y, y_pred_custom))
print("Sklearn Model R² Score:", r2_score(y, y_pred_sklearn))

print("Custom Model MSE:", mean_squared_error(y, y_pred_custom))
print("Sklearn Model MSE:", mean_squared_error(y, y_pred_sklearn))


plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_custom, color='red', label='Custom Model')
plt.plot(X, y_pred_sklearn, color='green', linestyle='--', label='Sklearn Model')
plt.title("Linear Regression: Custom vs Sklearn")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

