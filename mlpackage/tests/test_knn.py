import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlpackage.supervised_learning.knn import KNN


@pytest.fixture
def iris_data():
    """Load and split the Iris dataset for testing."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_knn_accuracy_iris(iris_data):
    """KNN should achieve at least 90% accuracy on Iris test set."""
    X_train, X_test, y_train, y_test = iris_data
    model = KNN(k=3)
    model.fit(X_train, y_train)
    assert model.accuracy(X_test, y_test) > 0.90


def test_knn_predict_shape(iris_data):
    """Predictions should have the same length as the test set."""
    X_train, X_test, y_train, y_test = iris_data
    model = KNN(k=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_knn_k1_perfect_train(iris_data):
    """With k=1, KNN should perfectly memorize training data."""
    X_train, X_test, y_train, y_test = iris_data
    model = KNN(k=1)
    model.fit(X_train, y_train)
    assert model.accuracy(X_train, y_train) == 1.0


def test_knn_confusion_matrix_shape(iris_data):
    """Confusion matrix should be square with size = number of classes."""
    X_train, X_test, y_train, y_test = iris_data
    model = KNN(k=3)
    model.fit(X_train, y_train)
    cm = model.confusion_matrix(X_test, y_test)
    assert cm.shape == (3, 3)
