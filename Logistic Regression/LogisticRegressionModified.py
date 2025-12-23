import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, threshold = 0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, x, y):
        np.random.seed(42)
        noOfSamples, numOfFeatures = x.shape
        self.weights = np.zeros(numOfFeatures)
        self.bias = 0.0

        for epoch in range(self.epochs):
            z = x @ self.weights + self.bias
            y_predicted = self.sigmoid(z)

            # Gradients 
            dw = (x.T @ (y_predicted - y))/ noOfSamples
            db = np.mean(y_predicted - y)

            # Update
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            # log loss
            loss = -np.mean(y*np.log(y_predicted) + (1-y) * np.log(1-y_predicted))
            self.loss_history.append(loss)
        return self
    
    def predict_probs(self, x):
        z = x@self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, x):
        return (self.predict_probs(x) >= self.threshold).astype(int)