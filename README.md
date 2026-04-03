# CMOR 438 - Data Science and Machine Learning
**Author:** Karyn Fu

---

## Overview

This repository contains a collection of Jupyter notebooks demonstrating core machine learning concepts taught in CMOR 438, alongside a custom Python machine learning package built from scratch. The package implements fundamental supervised learning algorithms without relying on scikit-learn's model implementations, emphasizing a deep understanding of the underlying mathematics and logic.

---

## Repository Structure
```
CMOR438-Spring-2026/
├── mlpackage/
│   ├── supervised_learning/
│   │   ├── knn.py               # K-Nearest Neighbors classifier
│   │   └── linear_regression.py # Linear Regression via Normal Equation
│   ├── tests/
│   │   ├── test_knn.py
│   │   └── test_linear_regression.py
│   └── init.py
├── notebooks/
│   └── mlpackage_demo.ipynb               # Demo notebook for mlpackage
├── pytest.ini
├── requirements.txt
└── README.md
```
---

## ML Package

The `mlpackage` library provides from-scratch implementations of supervised learning algorithms. All models are implemented using only `numpy` and `pandas` — no scikit-learn models are used under the hood.

### Algorithms

#### K-Nearest Neighbors (`mlpackage.supervised_learning.knn`)

A lazy classification algorithm that predicts a label by finding the K closest training points and taking a majority vote.

- `fit(X, y)` — stores training data
- `predict(X)` — returns predicted class labels
- `accuracy(X, y)` — returns fraction of correct predictions
- `confusion_matrix(X, y)` — returns a DataFrame confusion matrix
- `draw_decision_boundary(X, y)` — plots decision boundary for 2D data

#### Linear Regression (`mlpackage.supervised_learning.linear_regression`)

A regression algorithm that fits a line to data using the Normal Equation (closed-form solution).

- `fit(X, y)` — computes optimal parameters using the pseudoinverse
- `predict(X)` — returns predicted continuous values
- `rmse(X, y)` — returns Root Mean Squared Error
- `R_squared(X, y)` — returns coefficient of determination (R²)

---

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/karynfu/CMOR438-Spring-2026.git
cd CMOR438-Spring-2026
pip install -r requirements.txt
```

---

## Usage
```python
from mlpackage.supervised_learning.knn import KNN
from mlpackage.supervised_learning.linear_regression import LinearRegression

# KNN example
model = KNN(k=3)
model.fit(X_train, y_train)
print(model.accuracy(X_test, y_test))

# Linear Regression example
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.R_squared(X_test, y_test))
```

---

## Running the Tests

All algorithms are tested using `pytest`. To run the full test suite:
```bash
pytest -v
```

All 10 tests should pass covering accuracy, prediction shape, edge cases, and error handling.

---

## Demo Notebook

The notebook `notebooks/mlpackage_demo.ipynb` demonstrates both algorithms on the Iris dataset, including:

- Data exploration and visualization
- Train/test splitting
- Model training and evaluation
- Confusion matrix for KNN
- Regression line visualization for Linear Regression
- Analysis of how K affects KNN accuracy

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (dataset loading and train/test split only)
- `pytest`