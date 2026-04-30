# CMOR 438 - Data Science and Machine Learning

**Author:** Karyn Fu

---

## Overview

This repository contains a custom Python machine learning package and a comprehensive set of Jupyter notebooks developed for CMOR 438. The package implements fundamental supervised learning algorithms from scratch using only NumPy and Pandas — no scikit-learn models — to build a deep understanding of the underlying mathematics. The notebooks cover the full range of algorithms taught in the course, from classical ML to neural networks, each with data exploration, preprocessing, modeling, evaluation, and visualizations.

---

## Repository Structure

```
CMOR438-Spring-2026/
├── mlpackage/                              # Custom ML package
│   ├── __init__.py                         # Public API
│   ├── metrics.py                          # Regression and classification metrics
│   ├── preprocessing.py                    # Data splitting and feature scaling
│   ├── supervised_learning/
│   │   ├── __init__.py
│   │   ├── knn.py                          # K-Nearest Neighbors classifier
│   │   └── linear_regression.py            # Linear Regression via Normal Equation
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_knn.py                     # 4 unit tests for KNN
│   │   ├── test_linear_regression.py       # 6 unit tests for Linear Regression
│   │   ├── test_metrics.py                 # 16 unit tests for metrics
│   │   └── test_preprocessing.py           # 16 unit tests for preprocessing
│   └── README.md                           # Package-level documentation
├── notebooks/
│   ├── mlpackage_demo.ipynb                # End-to-end demo of the custom package
│   ├── Supervised Learning/
│   │   ├── knn.ipynb
│   │   ├── linear_regression.ipynb
│   │   ├── logistic_regression.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── random_forests.ipynb
│   │   ├── svm.ipynb
│   │   ├── naive_bayes.ipynb
│   │   └── neural_networks.ipynb
│   └── Unsupervised Learning/
│       ├── kmeans.ipynb
│       ├── pca.ipynb
│       └── hierarchical_clustering.ipynb
├── pyproject.toml                          # Package build config (pip install -e .)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## ML Package

The `mlpackage` library provides from-scratch implementations of supervised learning algorithms, preprocessing utilities, and evaluation metrics using only `numpy` and `pandas`. See [`mlpackage/README.md`](mlpackage/README.md) for the full API reference.

### Supervised Learning

#### K-Nearest Neighbors (`mlpackage.supervised_learning.knn`)

A non-parametric, lazy learning classifier. To predict a label, it finds the K nearest training points by Euclidean distance and returns the majority class vote.

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Stores training data |
| `predict(X)` | Returns predicted class labels |
| `accuracy(X, y)` | Returns fraction of correct predictions |
| `confusion_matrix(X, y)` | Returns a labeled DataFrame confusion matrix |
| `draw_decision_boundary(X, y)` | Plots decision boundary for 2D data |

#### Linear Regression (`mlpackage.supervised_learning.linear_regression`)

A regression algorithm that finds optimal parameters using the **Normal Equation** — a closed-form solution requiring no iterative optimization:

$$\theta = (X^\top X)^+ X^\top y$$

The pseudoinverse $(X^\top X)^+$ is used for numerical stability on collinear features.

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Computes optimal weights via the Normal Equation |
| `predict(X)` | Returns predicted continuous values |
| `rmse(X, y)` | Returns Root Mean Squared Error |
| `R_squared(X, y)` | Returns coefficient of determination (R²) |

### Preprocessing (`mlpackage.preprocessing`)

Common data preparation utilities with a consistent `fit` / `transform` / `fit_transform` interface.

| Class / Function | Description |
|------------------|-------------|
| `train_test_split(X, y, ...)` | Shuffle and split into train/test subsets |
| `StandardScaler` | Zero-mean, unit-variance normalization |
| `MinMaxScaler` | Scale features to a fixed [min, max] range |
| `LabelEncoder` | Encode categorical labels as integers |

### Metrics (`mlpackage.metrics`)

Standalone evaluation functions for regression and classification.

| Function | Description |
|----------|-------------|
| `mean_squared_error(y_true, y_pred)` | MSE |
| `root_mean_squared_error(y_true, y_pred)` | RMSE |
| `mean_absolute_error(y_true, y_pred)` | MAE |
| `r_squared(y_true, y_pred)` | Coefficient of determination (R²) |
| `accuracy(y_true, y_pred)` | Classification accuracy |
| `confusion_matrix(y_true, y_pred)` | Multiclass confusion matrix (ndarray) |
| `precision(y_true, y_pred)` | Binary precision |
| `recall(y_true, y_pred)` | Binary recall / sensitivity |
| `f1_score(y_true, y_pred)` | Binary F1 score |

---

## Jupyter Notebooks

All notebooks follow a consistent workflow: data exploration → preprocessing → modeling → evaluation → visualizations → key takeaways. Each includes detailed Markdown explanations of the algorithm's mathematics and intuition.

### Package Demo

| Notebook | Description | Dataset |
|----------|-------------|---------|
| `mlpackage_demo.ipynb` | Demonstrates both `mlpackage` algorithms end-to-end, including decision boundary plots and regression line visualization | Iris |

### Supervised Learning

| Notebook | Algorithm | Dataset | Topics Covered |
|----------|-----------|---------|----------------|
| `knn.ipynb` | K-Nearest Neighbors | Iris | Euclidean distance, K sweep (1–30), decision boundaries, feature scaling comparison |
| `linear_regression.ipynb` | Linear Regression | Diabetes | Normal Equation, residual analysis, Ridge & Lasso, coefficient paths |
| `logistic_regression.ipynb` | Logistic Regression | Breast Cancer | Sigmoid function, gradient descent, log loss, ROC/AUC curve |
| `decision_trees.ipynb` | Decision Trees (CART) | Iris | Gini impurity, tree depth vs. accuracy, feature importance, decision boundary |
| `random_forests.ipynb` | Random Forests & Gradient Boosting | Wine | Bagging, feature randomness, ensemble comparison, number of trees |
| `svm.ipynb` | Support Vector Machines | Iris | Margin maximization, linear/poly/RBF kernels, effect of C, GridSearchCV |
| `naive_bayes.ipynb` | Naïve Bayes | Digits | Bayes' theorem, class-conditional means, GNB/BNB/MNB comparison |
| `neural_networks.ipynb` | Feed-Forward MLP | Digits | Backpropagation, ReLU, mini-batch SGD, Adam, L2 regularization, architecture comparison |

### Unsupervised Learning

| Notebook | Algorithm | Dataset | Topics Covered |
|----------|-----------|---------|----------------|
| `kmeans.ipynb` | K-Means Clustering | Synthetic blobs, Iris | Lloyd's algorithm, K-Means++ init, elbow method, silhouette score |
| `pca.ipynb` | Principal Component Analysis | Digits | Eigendecomposition, explained variance, image reconstruction, 2D projection |
| `hierarchical_clustering.ipynb` | Agglomerative Clustering | Iris | Dendrograms, linkage methods (single/complete/average/Ward), choosing K |

---

## Datasets

All datasets are loaded directly from `sklearn.datasets` — no external files needed.

| Dataset | Source | Samples | Features | Task | Used In |
|---------|--------|---------|----------|------|---------|
| **Iris** | `load_iris()` | 150 | 4 | Multiclass classification | KNN, Decision Trees, SVM, K-Means, Hierarchical Clustering, mlpackage demo |
| **Breast Cancer Wisconsin** | `load_breast_cancer()` | 569 | 30 | Binary classification | Logistic Regression |
| **Diabetes** | `load_diabetes()` | 442 | 10 | Regression | Linear Regression |
| **Digits** | `load_digits()` | 1,797 | 64 (8×8 pixels) | Multiclass classification | Naïve Bayes, Neural Networks, PCA |
| **Wine** | `load_wine()` | 178 | 13 | Multiclass classification | Random Forests |
| **Synthetic blobs** | `make_blobs()` | 300 | 2 | Clustering | K-Means |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/karynfu/CMOR438-Spring-2026.git
cd CMOR438-Spring-2026

# Install the package in editable mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Using the Package

```python
from mlpackage import KNN, LinearRegression
from mlpackage.preprocessing import StandardScaler, train_test_split
from mlpackage import metrics
from sklearn.datasets import load_iris

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for distance-based models)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# KNN classification
knn = KNN(k=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)
print(f"KNN Accuracy: {metrics.accuracy(y_test, y_pred):.4f}")
print(metrics.confusion_matrix(y_test, y_pred))

# Linear Regression
from mlpackage.supervised_learning import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np

X_reg, y_reg = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_tr, y_tr)
print(f"R²:   {metrics.r_squared(y_te, lr.predict(X_te)):.4f}")
print(f"RMSE: {metrics.root_mean_squared_error(y_te, lr.predict(X_te)):.4f}")
```

### Running the Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and open any notebook. Run all cells with **Cell → Run All**.

### Running the Tests

```bash
pytest -v
```

42 tests across 4 test files covering prediction accuracy, output shapes, metric correctness, scaler invertibility, and error handling.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation (core of all algorithm implementations) |
| `pandas` | Data manipulation and confusion matrix formatting |
| `matplotlib` | Plotting and visualizations |
| `scikit-learn` | Dataset loading, train/test splitting, and evaluation metrics only |
| `scipy` | Hierarchical clustering (dendrogram computation) |
| `pytest` | Unit testing framework |
