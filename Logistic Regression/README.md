# ⚙️ Logistic Regression from Scratch | Diabetes Dataset 🧠

📁 Project       : Logistic Regression (Custom + Sklearn)
📊 Dataset       : Pima Indians Diabetes Dataset (CSV)
🛠️ Language      : Python
📦 Libraries     : NumPy, Pandas, Matplotlib, Scikit-learn
🎯 Goal          : Implement logistic regression from scratch, plot training loss, and compare with sklearn

──────────────────────────────────────────────────────────────────────────────

📌 OVERVIEW:
This project demonstrates how Logistic Regression works under the hood.
It includes:
   ✔️ Custom implementation using NumPy (no ML libraries)
   ✔️ Real-time loss plotting during training
   ✔️ Accuracy evaluation on both train and test sets
   ✔️ Comparison with Scikit-learn’s LogisticRegression

──────────────────────────────────────────────────────────────────────────────

📂 FILE STRUCTURE:
│
├── diabetes.csv                 # Dataset
├── LogisticRegression.py        # Custom implementation
├── README.md                    # This file
└── loss_plot.png                # Saved loss curve (if saved)

──────────────────────────────────────────────────────────────────────────────

🚀 HOW TO RUN:

1. Install required libraries:
   pip install numpy pandas matplotlib scikit-learn

2. Run the model:
   python LogisticRegression.py

3. Output:
   • Training Accuracy (Custom & Sklearn)
   • Testing Accuracy (Custom & Sklearn)
   • Loss plot (for custom model)

──────────────────────────────────────────────────────────────────────────────

📈 PLOT: LOSS VS ITERATIONS

• Visualizes how the model converges
• Helps in tuning learning rate & iterations

──────────────────────────────────────────────────────────────────────────────

⚖️ ACCURACY COMPARISON

Custom Logistic Regression:
   • Training Accuracy: ~64%
   • Testing Accuracy : ~64%

Sklearn Logistic Regression:
   • Training Accuracy: ~77%
   • Testing Accuracy : ~74%

🔍 Note: Accuracy may vary slightly on each run due to train-test split randomness.

──────────────────────────────────────────────────────────────────────────────

🧠 MATH USED :

• Sigmoid Function: 1 / (1 + e^(-z))
• Binary Cross-Entropy Loss
• Gradient Descent:
   • dw = (1/m) * X.T · (ŷ - y)
   • db = (1/m) * sum(ŷ - y)

• Numerically stable sigmoid used to avoid overflow errors

──────────────────────────────────────────────────────────────────────────────

💡 NEXT STEPS / IDEAS:

• Add L2 Regularization
• Try on other binary classification datasets
• Implement mini-batch or stochastic gradient descent
• Add precision, recall, and confusion matrix

──────────────────────────────────────────────────────────────────────────────

👨‍💻 AUTHOR:
Aman Kukreja  
GitHub   : https://github.com/amxn18  
LinkedIn : https://linkedin.com/in/amankukreja18/

──────────────────────────────────────────────────────────────────────────────
