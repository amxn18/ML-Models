import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class LassoRegression:
    def __init__(self, learningRate, iterations, lambdaParameter):
        self.learningRate = learningRate
        self.iterations = iterations
        self.lambdaParameter = lambdaParameter

    def fit(self, x, y):
        # m No of data pts(rows), n No of features(columns)
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        for i in range(self.iterations):
            self.updateWeights()

    def updateWeights(self):
        yPrediction = self.predict(self.x)
        # Initialize gradient vector
        self.dw = np.zeros(self.n)
        
        # Gradients for weights
        for i in range(self.n):
            if self.w[i] > 0:
                self.dw[i] = (-(2 * (self.x[:, i].dot(self.y - yPrediction))) + self.lambdaParameter) / self.m
            else:
                self.dw[i] = (-(2 * (self.x[:, i].dot(self.y - yPrediction))) - self.lambdaParameter) / self.m  
        
        # Gradient for bias
        self.db = -2 * np.sum(self.y - yPrediction) / self.m
        
        # Update weights and bias using learning rate
        self.w -= self.learningRate * self.dw
        self.b -= self.learningRate * self.db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# Loading Both Models
# Model 1: Custom Lasso Regression
# Model 2: Sklearn Lasso Regression
model1 = LassoRegression(learningRate=0.01, iterations=1000, lambdaParameter=0.1)
model2 = Lasso(alpha=0.1)

# Load dataset
df = pd.read_csv("Lasso Regression/data.csv")
print(df.head())

x = df["YearsExperience"].values.reshape(-1, 1)
y = df["Salary"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model1.fit(x_train, y_train)

# Evaluation on Training Data For Model 1
yTrainPredictionModel1 = model1.predict(x_train)
yTrainScoreModel1 = r2_score(y_train, yTrainPredictionModel1)
yTrainMSEModel1 = mean_squared_error(y_train, yTrainPredictionModel1)


# Evaluation on Testing Data For Model 1
yTestPredictionModel1 = model1.predict(x_test)
yTestScoreModel1 = r2_score(y_test, yTestPredictionModel1)
yTestMSEModel1 = mean_squared_error(y_test, yTestPredictionModel1)


model2.fit(x_train, y_train)
# Evaluation on Training Data For Model 2
yTrainPredictionModel2 = model2.predict(x_train)
yTrainScoreModel2 = r2_score(y_train, yTrainPredictionModel2)
yTrainMSEModel2 = mean_squared_error(y_train, yTrainPredictionModel2)


# Evaluation on Testing Data For Model 2
yTestPredictionModel2 = model2.predict(x_test)
yTestScoreModel2 = r2_score(y_test, yTestPredictionModel2)
yTestMSEModel2 = mean_squared_error(y_test, yTestPredictionModel2)


print("Lasso Regression Model 1 (Custom Model)")
print("R2 Score for Training Data (Custom Model):", yTrainScoreModel1)
print("MSE for Training Data (Custom Model):", yTrainMSEModel1)

print("Lasso Regression Model 1 (Custom Model)")
print("R2 Score for Testing Data (Custom Model):", yTestScoreModel1)
print("MSE for Testing Data (Custom Model):", yTestMSEModel1)

print("Lasso Regression Model 2 (Sklearn Model)")
print("R2 Score for Training Data (Sklearn Model):", yTrainScoreModel2)
print("MSE for Training Data (Sklearn Model):", yTrainMSEModel2)

print("Lasso Regression Model 2 (Sklearn Model)")
print("R2 Score for Testing Data (Sklearn Model):", yTestScoreModel2)
print("MSE for Testing Data (Sklearn Model):", yTestMSEModel2)

# 📊 Plotting Results

# Plotting Training Data and Predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_train, yTrainPredictionModel1, color='red', label='Custom Lasso Model')
plt.plot(x_train, yTrainPredictionModel2, color='green', linestyle='--', label='Sklearn Lasso Model')
plt.title("Training Data vs Predictions")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Plotting Testing Data and Predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='blue', label='Testing Data')
plt.plot(x_test, yTestPredictionModel1, color='red', label='Custom Lasso Model')
plt.plot(x_test, yTestPredictionModel2, color='green', linestyle='--', label='Sklearn Lasso Model')
plt.title("Testing Data vs Predictions")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
