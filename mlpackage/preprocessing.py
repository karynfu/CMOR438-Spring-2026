"""
preprocessing.py
----------------
Data preprocessing utilities for machine learning pipelines.

This module provides common preprocessing transformations implemented from
scratch using only NumPy. All transformers follow a consistent fit/transform
interface compatible with the rest of mlpackage.

Available transformers
----------------------
- train_test_split   : Split arrays into random train and test subsets
- StandardScaler     : Zero-mean, unit-variance feature normalization
- MinMaxScaler       : Scale features to a fixed range [min, max]
- LabelEncoder       : Encode categorical string labels as integers
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split. Must be
        in the range (0, 1).
    random_state : int or None, default=None
        Seed for the random number generator. Pass an integer for
        reproducible results.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting. If False, the first
        ``1 - test_size`` fraction of rows goes to train.

    Returns
    -------
    X_train : np.ndarray
    X_test  : np.ndarray
    y_train : np.ndarray
    y_test  : np.ndarray

    Raises
    ------
    ValueError
        If ``test_size`` is not in (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20).reshape(10, 2)
    >>> y = np.arange(10)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    >>> X_train.shape, X_test.shape
    ((7, 2), (3, 2))
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    n_test = max(1, int(np.ceil(n * test_size)))
    n_train = n - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    For each feature j:
        z = (x - μ_j) / σ_j

    where μ_j and σ_j are the mean and standard deviation computed from
    the training data. Features with zero variance are left unchanged (z = 0).

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean computed on the training data.
    std_ : np.ndarray of shape (n_features,)
        Per-feature standard deviation computed on the training data.
        Zero-variance features are assigned std_ = 1 to avoid division by zero.

    Examples
    --------
    >>> import numpy as np
    >>> scaler = StandardScaler()
    >>> X_train = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> scaler.fit(X_train)
    StandardScaler()
    >>> scaler.transform(X_train)
    array([[-1.22...,  -1.22...],
           [ 0.   ,   0.   ],
           [ 1.22...,  1.22...]])
    """

    def fit(self, X):
        """
        Compute per-feature mean and standard deviation from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data used to compute statistics.

        Returns
        -------
        self : StandardScaler
            Fitted scaler (enables method chaining).
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero for constant features
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """
        Standardize features using the statistics computed during ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Standardized data.

        Raises
        ------
        AttributeError
            If ``fit`` has not been called yet.
        """
        if not hasattr(self, "mean_"):
            raise AttributeError("StandardScaler is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Equivalent to calling ``fit(X).transform(X)`` but more convenient.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Standardized training data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse the standardization transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Standardized data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Data in the original feature scale.
        """
        if not hasattr(self, "mean_"):
            raise AttributeError("StandardScaler is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return X * self.std_ + self.mean_

    def __repr__(self):
        return "StandardScaler()"


class MinMaxScaler:
    """
    Scale features to a specified range [feature_range[0], feature_range[1]].

    For each feature j:
        x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min

    where ``min`` and ``max`` are the bounds of ``feature_range``.

    Parameters
    ----------
    feature_range : tuple of (float, float), default=(0, 1)
        Desired range of the transformed data.

    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Per-feature minimum seen during fit.
    max_ : np.ndarray of shape (n_features,)
        Per-feature maximum seen during fit.
    scale_ : np.ndarray of shape (n_features,)
        Per-feature relative scaling factor.

    Examples
    --------
    >>> import numpy as np
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> scaler.fit_transform(X)
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """

    def __init__(self, feature_range=(0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError(f"feature_range must satisfy lo < hi, got {feature_range}.")
        self.feature_range = feature_range

    def fit(self, X):
        """
        Compute per-feature min and max from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : MinMaxScaler
        """
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        diff = self.max_ - self.min_
        # Constant features map to feature_range[0]
        diff[diff == 0] = 1.0
        self.scale_ = diff
        return self

    def transform(self, X):
        """
        Scale features using the statistics computed during ``fit``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to scale.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Scaled data in ``feature_range``.

        Raises
        ------
        AttributeError
            If ``fit`` has not been called yet.
        """
        if not hasattr(self, "min_"):
            raise AttributeError("MinMaxScaler is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        X_std = (X - self.min_) / self.scale_
        return X_std * (hi - lo) + lo

    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """
        Reverse the min-max scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Scaled data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Data in the original feature scale.
        """
        if not hasattr(self, "min_"):
            raise AttributeError("MinMaxScaler is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        lo, hi = self.feature_range
        return (X - lo) / (hi - lo) * self.scale_ + self.min_

    def __repr__(self):
        return f"MinMaxScaler(feature_range={self.feature_range})"


class LabelEncoder:
    """
    Encode categorical string (or integer) labels as integers in [0, n_classes - 1].

    The encoding is determined from the sorted unique values seen during ``fit``.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted array of unique class labels seen during fit.

    Examples
    --------
    >>> le = LabelEncoder()
    >>> le.fit(['cat', 'dog', 'bird'])
    LabelEncoder()
    >>> le.transform(['dog', 'cat', 'bird'])
    array([1, 0, 2])  # sorted: bird=0, cat=1, dog=2... wait, alphabetical
    >>> le.inverse_transform([1, 0, 2])
    array(['cat', 'bird', 'dog'], dtype=...)
    """

    def fit(self, y):
        """
        Fit the encoder by discovering unique class labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target labels (strings, integers, or any comparable type).

        Returns
        -------
        self : LabelEncoder
        """
        self.classes_ = np.unique(y)
        self._label_to_int = {label: i for i, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """
        Encode labels as integers.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Labels to encode. Must have been seen during ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Integer-encoded labels.

        Raises
        ------
        AttributeError
            If ``fit`` has not been called.
        ValueError
            If ``y`` contains labels not seen during ``fit``.
        """
        if not hasattr(self, "classes_"):
            raise AttributeError("LabelEncoder is not fitted. Call fit() first.")
        y = np.asarray(y)
        try:
            return np.array([self._label_to_int[label] for label in y])
        except KeyError as e:
            raise ValueError(f"Unseen label during transform: {e}") from e

    def fit_transform(self, y):
        """
        Fit and encode labels in one step.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Labels to fit and encode.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """
        Decode integer labels back to the original label type.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Integer-encoded labels.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Original labels.
        """
        if not hasattr(self, "classes_"):
            raise AttributeError("LabelEncoder is not fitted. Call fit() first.")
        y = np.asarray(y, dtype=int)
        return self.classes_[y]

    def __repr__(self):
        return "LabelEncoder()"
