# 📈 Custom Linear Regression vs Scikit-learn

This project demonstrates a **custom implementation** of Linear Regression using **Gradient Descent**, and compares its performance with **Scikit-learn's LinearRegression**.

---

## 🗂️ Dataset

The dataset used is a simple CSV file: `data.csv`, which contains:

- `YearsExperience` (independent variable)
- `Salary` (dependent variable)

---

## 🛠️ Custom Linear Regression

### 🔁 Workflow

1. Initialize weights and bias to 0.
2. Perform **gradient descent** updates:
   - Compute predictions: `y_pred = X.dot(w) + b`
   - Calculate gradients:
     ```python
     dw = - (2/m) * X.T.dot(y - y_pred)
     db = - (2/m) * np.sum(y - y_pred)
     ```
   - Update parameters:
     ```python
     w -= learning_rate * dw
     b -= learning_rate * db
     ```

3. Repeat for given number of iterations.

---

### 📘 Formula

\[
\hat{y} = w \cdot x + b
\]

- Where:
  - \( w \) is the weight (slope)
  - \( b \) is the bias (intercept)

---

## ✅ Performance Evaluation

Both the **custom** and **Scikit-learn** models are evaluated on:

- **R² Score (Coefficient of Determination)**
- **Mean Squared Error (MSE)**

| Metric         | Custom Model           | Sklearn Model           |
|----------------|------------------------|--------------------------|
| **Weight (w)** | ≈ 9501                 | ≈ 9449                   |
| **Bias (b)**   | ≈ 25448                | ≈ 25729                  |
| **R² Score**   | ≈ 0.9569               | ≈ 0.9569                 |
| **MSE**        | ≈ 3.13 × 10⁶           | ≈ 3.12 × 10⁶             |

Both models perform nearly identically, validating the accuracy of the custom implementation.

---

## 📊 Visualization

Matplotlib is used to compare:

- 🔵 **Data Points**
- 🔴 **Custom Model Line**
- 🟢 **Sklearn Model Line (dashed)**

The plot clearly shows both regression lines overlapping, confirming similar behavior.

---

## 🔍 Files

- `LinearRegression.py` — Custom model class and training logic
- `data.csv` — Dataset (YearsExperience vs Salary)
- `README.md` — Project overview

---

## 💡 Future Enhancements

- [ ] Plot **loss vs. iterations**
- [ ] Add **train-test split** and validation
- [ ] Extend to **multivariable regression**
- [ ] Use **early stopping** or **momentum**

---

## 👨‍💻 Author

Made with ❤️ as part of a machine learning fundamentals project.

EOF
