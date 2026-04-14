# CMOR 438 - Data Science and Machine Learning
**Author:** Karyn Fu

---

## Overview

This repository contains a collection of Jupyter notebooks demonstrating core machine learning concepts taught in CMOR 438, alongside a custom Python machine learning package built from scratch. The package implements fundamental supervised learning algorithms without relying on scikit-learn's model implementations, emphasizing a deep understanding of the underlying mathematics and logic.

---

## Repository Structure

```
CMOR438-Spring-2026/
├── mlpackage/                          # Custom ML package (from scratch)
│   ├── supervised_learning/
│   │   ├── knn.py                      # K-Nearest Neighbors classifier
│   │   └── linear_regression.py        # Linear Regression via Normal Equation
│   └── tests/
│       ├── test_knn.py
│       └── test_linear_regression.py
├── notebooks/
│   ├── mlpackage_demo.ipynb            # Package demo: KNN + Linear Regression on Iris
│   ├── Supervised_Learning/
│   │   ├── knn.ipynb
│   │   ├── linear_regression.ipynb
│   │   ├── logistic_regression.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── random_forests.ipynb
│   │   ├── svm.ipynb
│   │   ├── naive_bayes.ipynb
│   │   └── neural_networks.ipynb
│   ├── Unsupervised_Learning/
│   │   ├── kmeans.ipynb
│   │   ├── pca.ipynb
│   │   └── hierarchical_clustering.ipynb
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Jupyter Notebooks

All notebooks use preloaded sklearn datasets and include data exploration, preprocessing, modeling, evaluation, and visualizations.

### ML Package Demo

| Notebook | Algorithms | Dataset |
|----------|-----------|---------|
| `mlpackage_demo.ipynb` | KNN, Linear Regression (from `mlpackage`) | Iris |

### Supervised Learning

| Notebook | Algorithm | Dataset | Key Topics |
|----------|-----------|---------|------------|
| `knn.ipynb` | K-Nearest Neighbors | Iris | Euclidean distance, K tuning, decision boundaries, feature scaling |
| `linear_regression.ipynb` | Linear Regression | Diabetes | Normal Equation, residual analysis, Ridge, Lasso coefficient paths |
| `logistic_regression.ipynb` | Logistic Regression | Breast Cancer | Sigmoid, gradient descent, log loss, ROC curve |
| `decision_trees.ipynb` | Decision Trees (CART) | Iris | Gini impurity, tree depth, feature importance, decision boundary |
| `random_forests.ipynb` | Random Forests, Gradient Boosting | Wine | Bagging, feature randomness, ensemble comparison |
| `svm.ipynb` | Support Vector Machines | Iris | Margin maximization, kernels, GridSearchCV |
| `naive_bayes.ipynb` | Gaussian Naïve Bayes | Digits | Bayes' theorem, class-conditional means, NB variants |
| `neural_networks.ipynb` | MLP (feed-forward) | Digits | Backpropagation, ReLU, SGD, Adam, L2 regularization |

### Unsupervised Learning

| Notebook | Algorithm | Dataset | Key Topics |
|----------|-----------|---------|------------|
| `kmeans.ipynb` | K-Means | Synthetic blobs, Iris | Lloyd's algorithm, K-Means++, elbow method, silhouette score |
| `pca.ipynb` | PCA | Digits | Eigendecomposition, explained variance, image reconstruction |
| `hierarchical_clustering.ipynb` | Agglomerative Clustering | Iris | Dendrograms, linkage methods, choosing K |

---

## ML Package

The `mlpackage` library provides from-scratch implementations of two supervised learning algorithms using only `numpy` and `pandas`.

### K-Nearest Neighbors (`mlpackage.supervised_learning.knn`)

A lazy classification algorithm that predicts a label by finding the K closest training points and taking a majority vote.

- `fit(X, y)` — stores training data
- `predict(X)` — returns predicted class labels
- `accuracy(X, y)` — returns fraction of correct predictions
- `confusion_matrix(X, y)` — returns a DataFrame confusion matrix
- `draw_decision_boundary(X, y)` — plots decision boundary for 2D data

### Linear Regression (`mlpackage.supervised_learning.linear_regression`)

A regression algorithm that fits a line to data using the Normal Equation (closed-form solution).

$$\theta = (X^\top X)^+ X^\top y$$

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

10 tests covering accuracy, prediction shape, edge cases, and error handling.

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn` (dataset loading, preprocessing, and metrics)
- `scipy` (hierarchical clustering)
- `pytest`
