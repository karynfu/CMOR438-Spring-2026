"""
supervised_learning
-------------------
From-scratch implementations of supervised machine learning algorithms.

Available classes
-----------------
KNN
    K-Nearest Neighbors classifier using Euclidean distance.
LinearRegression
    Closed-form linear regression via the Normal Equation.
"""

from mlpackage.supervised_learning.knn import KNN
from mlpackage.supervised_learning.linear_regression import LinearRegression

__all__ = ["KNN", "LinearRegression"]
