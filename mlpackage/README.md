# mlpackage

A from-scratch machine learning library built for CMOR 438 at Rice University.

All core algorithms are implemented using only **NumPy** and **Pandas**. The goal is to build a deep understanding of how machine learning algorithms work.

---

## Contents

```
mlpackage/
├── supervised_learning/
│   ├── knn.py                  # K-Nearest Neighbors classifier
│   └── linear_regression.py    # Linear Regression (Normal Equation)
├── preprocessing.py            # Data splitting and feature scaling
├── metrics.py                  # Evaluation metrics for regression and classification
├── tests/
│   ├── test_knn.py
│   ├── test_linear_regression.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
└── README.md
```

---

## Installation

Clone the repository and install the package in editable mode so that changes to the source are immediately reflected:

```bash
git clone https://github.com/karynfu/CMOR438-Spring-2026.git
cd CMOR438-Spring-2026
pip install -e .
```

**Requirements:** Python >= 3.9, NumPy, Pandas, Matplotlib

---

## Modules

### `supervised_learning`

#### `KNN` — K-Nearest Neighbors Classifier

A lazy, non-parametric classifier that stores all training data and classifies new points by majority vote among the K nearest neighbors (Euclidean distance).

```python
from mlpackage.supervised_learning import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNN(k=5)
model.fit(X_train, y_train)

print(model.accuracy(X_test, y_test))   # e.g. 0.9667
print(model.confusion_matrix(X_test, y_test))
model.draw_decision_boundary(X_test[:, :2], y_test)  # 2D only
```

| Method | Description |
|---|---|
| `fit(X, y)` | Store training data |
| `predict(X)` | Return predicted labels |
| `accuracy(X, y)` | Fraction of correct predictions |
| `confusion_matrix(X, y)` | Pandas DataFrame confusion matrix |
| `draw_decision_boundary(X, y)` | Plot 2D decision regions |

---

#### `LinearRegression` — Normal Equation

Fits a linear model analytically using the closed-form Normal Equation:

θ = (XᵀX)⁺ Xᵀy

Uses the Moore-Penrose pseudoinverse for numerical stability when XᵀX is singular.

```python
import numpy as np
from mlpackage.supervised_learning import LinearRegression

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

model = LinearRegression()
model.fit(X, y)

print(model.predict(np.array([[6]])))   # ≈ [12.0]
print(model.rmse(X, y))                # root mean squared error
print(model.R_squared(X, y))           # coefficient of determination
```

| Method | Description |
|---|---|
| `fit(X, y)` | Compute coefficients via Normal Equation |
| `predict(X)` | Return predicted values |
| `rmse(X, y)` | Root Mean Squared Error |
| `R_squared(X, y)` | Coefficient of determination (R²) |

---

### `preprocessing`

Common preprocessing utilities with a fit/transform interface.

```python
from mlpackage.preprocessing import train_test_split, StandardScaler, MinMaxScaler, LabelEncoder

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize (zero mean, unit variance)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Scale to [0, 1]
mm = MinMaxScaler(feature_range=(0, 1))
X_scaled = mm.fit_transform(X_train)

# Encode string labels
le = LabelEncoder()
y_encoded = le.fit_transform(['cat', 'dog', 'cat', 'bird'])
y_labels  = le.inverse_transform(y_encoded)
```

| Class / Function | Description |
|---|---|
| `train_test_split(X, y, ...)` | Shuffle and split into train/test |
| `StandardScaler` | Zero-mean, unit-variance normalization |
| `MinMaxScaler` | Scale to a fixed [min, max] range |
| `LabelEncoder` | Map categorical labels to integers |

---

### `metrics`

Standalone evaluation functions for both regression and classification.

```python
from mlpackage import metrics

# Regression
mse  = metrics.mean_squared_error(y_true, y_pred)
rmse = metrics.root_mean_squared_error(y_true, y_pred)
mae  = metrics.mean_absolute_error(y_true, y_pred)
r2   = metrics.r_squared(y_true, y_pred)

# Classification
acc  = metrics.accuracy(y_true, y_pred)
cm   = metrics.confusion_matrix(y_true, y_pred)   # np.ndarray
p    = metrics.precision(y_true, y_pred, pos_label=1)
r    = metrics.recall(y_true, y_pred, pos_label=1)
f1   = metrics.f1_score(y_true, y_pred, pos_label=1)
```

---

## Running Tests

Tests are written with [pytest](https://docs.pytest.org).

```bash
# From the repository root
pytest
```

All tests live in `mlpackage/tests/` and are auto-discovered by pytest.

---

## Example Notebook

See [`notebooks/mlpackage_demo.ipynb`](../notebooks/mlpackage_demo.ipynb) for an end-to-end walkthrough of both algorithms on the Iris dataset, including decision boundary plots and metric evaluation.

---

## Design Principles

- **From scratch**: only NumPy and Pandas; no scikit-learn for core model logic
- **Readable**: clean code with full docstrings (NumPy style) on every class and method
- **Tested**: every public method has at least one unit test
- **Consistent API**: all transformers use `fit` / `transform` / `fit_transform`
