# 🧠 Locally Weighted Regression (LWR)

A non-parametric regression technique that fits a linear model around each test point using a weighted version of least squares.

---

📌 OVERVIEW:
------------
- LWR (aka Locally Weighted Linear Regression / LWLR)
- Fits local models instead of one global line
- Uses Gaussian kernel to assign weights to nearby points
- Ideal for small to mid-size datasets with non-linear trends

---

📘 FORMULA:
-----------
Weight for i-th point:
    wᵢ = exp( - (xᵢ - x)² / (2 * τ²) )

Weighted normal equation:
    θ = (Xᵀ W X)⁻¹ Xᵀ W y

Prediction:
    y_pred = xᵀ θ

---

⚙️ WHAT'S INCLUDED:
-------------------
✅ LWR from scratch using NumPy  
✅ `.fit()` and `.predict()` methods  
✅ Gaussian kernel for weights  
✅ sklearn LinearRegression comparison  
✅ Matplotlib plots  

---

💡 INTUITION:
--------------
- Linear Regression fits one model globally
- LWR fits a separate model for each query point using weighted nearby data
- Bandwidth (τ) controls locality → small τ = more local

---

🧪 DEPENDENCIES:
-----------------
$ pip install numpy matplotlib scikit-learn

---


📊 OUTPUT:
-----------
- Blue points → Training data  
- 🔴 Red line → LWR predictions  
- 🟢 Green dashed → sklearn LinearRegression

---

⚖️ PROS vs CONS:
----------------
✅ Captures local patterns  
✅ No assumption about global data shape  
⚠️ Computationally heavy (O(n³) per test point)  
⚠️ Needs good τ tuning

---

📚 REFERENCES:
--------------
- Stanford CS229 Lecture Notes  
- Bishop’s PRML  
- https://en.wikipedia.org/wiki/Local_regression

