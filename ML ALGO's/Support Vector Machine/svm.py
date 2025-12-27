import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learningRate, iterations, lambda_param, tolerance=1e-5):
        self.learningRate = learningRate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.tolerance = tolerance

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        prev_w = self.w.copy()
        
        for i in range(self.iterations):
            self.updateWeights()
            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                print(f"Convergence reached at iteration {i}")
                break
            prev_w = self.w.copy()

    def updateWeights(self):
        yLabel = np.where(self.y <= 0, -1, 1)
        
        for idx, xi in enumerate(self.x):
            condition = yLabel[idx] * (np.dot(xi, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_param * self.w
                db = 0
            else:
                dw = 2 * self.lambda_param * self.w - np.dot(xi, yLabel[idx])
                db = yLabel[idx]
            
            self.w -= self.learningRate * dw
            self.b -= self.learningRate * db

    def predict(self, x):
        output = np.dot(x, self.w) - self.b
        return np.sign(output)

# Load and prepare the dataset
df = pd.read_csv("Support Vector Machine/diabetes.csv")
x = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Splitting the data into training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialising models
model1 = SVM(learningRate=0.01, iterations=5000, lambda_param=0.001)
model2 = SVC(kernel='linear')

# Training and testing model1 (custom SVM)
model1.fit(x_train, y_train)
yPredictionModel1Training = model1.predict(x_train)
yPredictionModel1Testing = model1.predict(x_test)

model1trainingScore = accuracy_score(y_train, yPredictionModel1Training)
model1testingScore = accuracy_score(y_test, yPredictionModel1Testing)

# Training and testing model2 (sklearn SVM)
model2.fit(x_train, y_train)
yPredictionModel2Training = model2.predict(x_train)
yPredictionModel2Testing = model2.predict(x_test)

model2trainingScore = accuracy_score(y_train, yPredictionModel2Training)
model2testingScore = accuracy_score(y_test, yPredictionModel2Testing)

print(f"Model 1 Training Score: {model1trainingScore}")
print(f"Model 1 Testing Score: {model1testingScore}")
print(f"Model 2 Training Score: {model2trainingScore}")
print(f"Model 2 Testing Score: {model2testingScore}")
