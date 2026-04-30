"""
metrics.py
----------
Evaluation metrics for machine learning models.

This module provides standalone functions for computing common regression
and classification metrics using only NumPy. All functions follow a
consistent signature: ``metric(y_true, y_pred)``.

Available metrics
-----------------
Regression:
    - mean_squared_error
    - root_mean_squared_error
    - mean_absolute_error
    - r_squared

Classification:
    - accuracy
    - precision
    - recall
    - f1_score
    - confusion_matrix
"""

import numpy as np


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mean_squared_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        MSE = mean((y_true - y_pred)²).

    Examples
    --------
    >>> mean_squared_error([3, -0.5, 2], [2.5, 0.0, 2])
    0.375
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        RMSE = sqrt(MSE).

    Examples
    --------
    >>> root_mean_squared_error([3, -0.5, 2], [2.5, 0.0, 2])
    0.6123...
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        MAE = mean(|y_true - y_pred|).

    Examples
    --------
    >>> mean_absolute_error([3, -0.5, 2], [2.5, 0.0, 2])
    0.5
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred):
    """
    Compute the coefficient of determination (R²).

    R² measures the proportion of variance in the target that is explained
    by the model. A score of 1.0 indicates a perfect fit; 0.0 means the
    model is no better than predicting the mean; negative values indicate
    the model is worse than a constant mean predictor.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R² = 1 - SS_res / SS_tot.

    Examples
    --------
    >>> r_squared([1, 2, 3, 4], [1, 2, 3, 4])
    1.0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Fraction of correctly classified samples in [0, 1].

    Examples
    --------
    >>> accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for a classification problem.

    Rows correspond to true classes, columns to predicted classes.
    Classes are ordered by their sorted unique values.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where entry [i, j] is the number of samples
        with true label i predicted as label j.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    array([[2, 0],
           [1, 1]])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm


def precision(y_true, y_pred, pos_label=1):
    """
    Compute precision for binary classification.

    Precision = TP / (TP + FP).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    pos_label : int or str, default=1
        The label of the positive class.

    Returns
    -------
    float
        Precision score in [0, 1]. Returns 0.0 if no positive predictions.

    Examples
    --------
    >>> precision([0, 1, 1, 0], [0, 1, 0, 0])
    1.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, pos_label=1):
    """
    Compute recall (sensitivity) for binary classification.

    Recall = TP / (TP + FN).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    pos_label : int or str, default=1
        The label of the positive class.

    Returns
    -------
    float
        Recall score in [0, 1]. Returns 0.0 if no positive true samples.

    Examples
    --------
    >>> recall([0, 1, 1, 0], [0, 1, 0, 0])
    0.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """
    Compute the F1 score for binary classification.

    F1 is the harmonic mean of precision and recall:
        F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels.
    pos_label : int or str, default=1
        The label of the positive class.

    Returns
    -------
    float
        F1 score in [0, 1]. Returns 0.0 if precision + recall == 0.

    Examples
    --------
    >>> f1_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.6666...
    """
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
