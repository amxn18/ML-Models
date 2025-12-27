import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class KNNClassifier:
    def __init__(self, distanceMetric='euclidean'):
        self.distanceMetric = distanceMetric

    def fit(self, X_train, y_train):
        # Storing training data and labels
        self.X_train = X_train
        self.y_train = y_train

    def getDistance(self, trainingDataPoint, testingDataPoint):
        # Calculate distance based on chosen metric
        if self.distanceMetric == "euclidean":
            return np.sqrt(np.sum((trainingDataPoint - testingDataPoint) ** 2))
        elif self.distanceMetric == "manhattan":
            return np.sum(np.abs(trainingDataPoint - testingDataPoint))
        else:
            raise ValueError("Unsupported distance metric")

    def nearestNeighbors(self, test_data, k):
        # Compute distances from test point to all training points
        distanceList = []
        for i in range(len(self.X_train)):
            distance = self.getDistance(self.X_train[i], test_data)
            distanceList.append((self.y_train[i], distance))

        # Sort by distance and take top-k labels
        distanceList.sort(key=lambda x: x[1])
        neighbourLabels = [distanceList[i][0] for i in range(k)]
        return neighbourLabels

    def predict(self, X_test, k=3):
        # Predict class for each test point
        predictions = []
        for test_point in X_test:
            neighbours = self.nearestNeighbors(test_point, k)
            predictedClass = statistics.mode(neighbours)
            predictions.append(predictedClass)
        return predictions

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom KNN model (from scratch)
model1 = KNNClassifier(distanceMetric="euclidean")
model1.fit(X_train, y_train)
y_pred_custom = model1.predict(X_test, k=3)

# Sklearn KNN model
model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train, y_train)
y_pred_sklearn = model2.predict(X_test)

# Accuracy comparison
print("Custom KNN Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Sklearn KNN Accuracy:", accuracy_score(y_test, y_pred_sklearn))

# Confusion matrix for custom KNN
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - Custom KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Confusion matrix for sklearn KNN
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_sklearn), annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix - Sklearn KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
