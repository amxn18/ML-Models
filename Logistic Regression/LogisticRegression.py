import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logistic Regression from Scratch
class logisticRegression:
    def __init__(self, learningRate, iterations):
        self.learningRate = learningRate
        self.iterations = iterations
        self.loss_history = []

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        for i in range(self.iterations):
            self.updateWeights()

            # Compute loss for current iteration
            z = np.dot(self.x, self.w) + self.b
            y_pred = 1 / (1 + np.exp(-z))
            loss = -np.mean(self.y * np.log(y_pred + 1e-9) + (1 - self.y) * np.log(1 - y_pred + 1e-9))
            self.loss_history.append(loss)

    def updateWeights(self):
        z = np.dot(self.x, self.w) + self.b
        yCap = 1 / (1 + np.exp(-z))

        dw = np.dot(self.x.T, (yCap - self.y)) / self.m
        db = np.sum(yCap - self.y) / self.m

        self.w -= self.learningRate * dw
        self.b -= self.learningRate * db

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        yPrediction = 1 / (1 + np.exp(-z))
        return np.where(yPrediction > 0.5, 1, 0)

# Load Dataset
df = pd.read_csv("Logistic Regression/diabetes.csv")
x = df.drop(columns=['Outcome'])
y = df['Outcome']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Custom Model
model1 = logisticRegression(learningRate=0.1, iterations=10000)
model1.fit(x_train.values, y_train.values)

# Predict on Train and Test data
x_train_prediction = model1.predict(x_train.values)
x_test_prediction = model1.predict(x_test.values)

# Accuracy scores
print("Custom Logistic Regression:")
print("Training Accuracy:", accuracy_score(y_train, x_train_prediction))
print("Testing Accuracy :", accuracy_score(y_test, x_test_prediction))

# Plotting loss over iterations
plt.figure(figsize=(8, 5))
plt.plot(model1.loss_history, color='blue')
plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Sklearn Logistic Regression
model2 = LogisticRegression(max_iter=5000)
model2.fit(x_train, y_train)

x_train_prediction = model2.predict(x_train)
x_test_prediction = model2.predict(x_test)

print("\nSklearn Logistic Regression:")
print("Training Accuracy:", accuracy_score(y_train, x_train_prediction))
print("Testing Accuracy :", accuracy_score(y_test, x_test_prediction))
