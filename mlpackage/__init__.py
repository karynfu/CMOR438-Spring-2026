"""
mlpackage
=========
A from-scratch machine learning library built for CMOR 438 / INDE 577.

All core algorithms are implemented using only NumPy and Pandas — no
scikit-learn models are used internally. The package is structured to
mirror the course curriculum:

Modules
-------
supervised_learning
    - KNN               : K-Nearest Neighbors classifier
    - LinearRegression  : Normal-equation linear regression

preprocessing
    - train_test_split  : Shuffle and split datasets
    - StandardScaler    : Zero-mean, unit-variance normalization
    - MinMaxScaler      : Scale features to [min, max]
    - LabelEncoder      : Encode categorical labels as integers

metrics
    - mean_squared_error        : MSE for regression
    - root_mean_squared_error   : RMSE for regression
    - mean_absolute_error       : MAE for regression
    - r_squared                 : Coefficient of determination
    - accuracy                  : Classification accuracy
    - confusion_matrix          : Multiclass confusion matrix
    - precision                 : Binary precision
    - recall                    : Binary recall / sensitivity
    - f1_score                  : Binary F1 score

Quick start
-----------
>>> from mlpackage.supervised_learning import KNN, LinearRegression
>>> from mlpackage.preprocessing import StandardScaler, train_test_split
>>> from mlpackage import metrics
"""

from mlpackage.supervised_learning.knn import KNN
from mlpackage.supervised_learning.linear_regression import LinearRegression

from mlpackage.preprocessing import (
    train_test_split,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)

from mlpackage import metrics

__all__ = [
    # Supervised learning
    "KNN",
    "LinearRegression",
    # Preprocessing
    "train_test_split",
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    # Metrics namespace
    "metrics",
]

__version__ = "0.1.0"
__author__ = "Karyn Fu"
