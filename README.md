# 📊 Machine Learning Models

This repository contains custom implementations of various machine learning models built from scratch. These models are implemented to demonstrate fundamental concepts, such as **Gradient Descent**, **Optimization**, and **Prediction**. Alongside custom implementations, **Scikit-learn** equivalents are also compared to validate the correctness of the models.

## 🧑‍💻 Models Included:
- **Linear Regression**
- **Logistic Regression**
- **Lasso Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Locally Weighted Regression (LWR)**

---

## 📂 Dataset

The datasets used in these models are synthetic or real-world datasets such as the **Iris dataset** for classification problems and custom data for regression problems. All datasets used are available in the respective model's folder.

---

## 📈 Models Overview

### 1. **Linear Regression**
- A regression model that establishes a relationship between input features and continuous output. The model is trained using **Gradient Descent** to minimize the Mean Squared Error (MSE).
- **Scikit-learn Equivalent**: `LinearRegression`

### 2. **Logistic Regression**
- A classification model used for binary or multiclass classification problems. The model uses a **logistic function** (sigmoid) to output probabilities, which are then mapped to class labels.
- **Scikit-learn Equivalent**: `LogisticRegression`

### 3. **Lasso Regression**
- A variation of linear regression that uses **L1 regularization** to reduce the complexity of the model and perform feature selection. It prevents overfitting by adding a penalty term to the cost function.
- **Scikit-learn Equivalent**: `Lasso`

### 4. **K-Nearest Neighbors (KNN)**
- A non-parametric classification and regression algorithm that predicts the output based on the **majority class** of the **k-nearest data points**. It calculates the distance between data points using **Euclidean** or **Manhattan distance**.
- **Scikit-learn Equivalent**: `KNeighborsClassifier`

### 5. **Support Vector Machine (SVM)**
- A classification algorithm that finds the hyperplane that best separates classes by maximizing the margin between them. SVM is known for its ability to handle high-dimensional spaces.
- **Scikit-learn Equivalent**: `SVC` (Support Vector Classifier)

### 6. **Locally Weighted Regression (LWR)**
- A variation of linear regression where the model assigns weights to the data points based on their proximity to the prediction point. It’s used for **nonlinear regression** problems.
- **Scikit-learn Equivalent**: Not directly available, but can be implemented with weighted regression techniques.

---

# 📊 Evaluation Metrics
All models are evaluated on the following metrics:
- R² Score (Coefficient of Determination)
- Mean Squared Error (MSE) for regression models
- Accuracy for classification models

# 📉 Visualizations
Some models also include visualizations to display the results:
- **Linear and Logistic Regression**: The fitted line is plotted against the actual data points.
- **KNN**: Decision boundaries for classification problems are plotted.
- **SVM**: The separating hyperplane is visualized.
- **Lasso Regression**: Weights are plotted to observe feature importance.

# 📁 Files in this Repository
- `linear_regression.py` — Custom implementation of Linear Regression.
- `logistic_regression.py` — Custom implementation of Logistic Regression.
- `lasso_regression.py` — Custom implementation of Lasso Regression.
- `knn_classifier.py` — Custom implementation of KNN Classifier.
- `svm_classifier.py` — Custom implementation of Support Vector Machine.
- `locally_weighted_regression.py` — Custom implementation of Locally Weighted Regression.
- `requirements.txt` — Python dependencies.
- `data/` — Dataset files used for training and testing.

# 💡 Future Enhancements
- [ ] Add **Cross-Validation** and **GridSearchCV** for hyperparameter tuning.
- [ ] Extend models to handle **multi-class classification** and **multi-output regression**.
- [ ] Improve **visualization** with interactive plots for better model understanding.
- [ ] Implement models using **Stochastic Gradient Descent** (SGD).

# 👨‍💻 Author
Made with ❤️ as part of a **Machine Learning** and **Data Science** learning journey.

Feel free to contribute, improve the models, or suggest improvements!

EOF
