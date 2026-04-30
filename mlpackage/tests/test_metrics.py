import numpy as np
import pytest
from mlpackage.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def test_mse_perfect():
    """MSE should be zero when predictions equal targets."""
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == pytest.approx(0.0)


def test_mse_known_value():
    """MSE on a simple example with known result."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5,  0.0, 2.0, 8.0])
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.375)


def test_rmse_is_sqrt_mse():
    """RMSE should equal sqrt(MSE) for any inputs."""
    y_true = np.array([1.0, 2.0, 5.0])
    y_pred = np.array([1.5, 2.5, 4.0])
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )


def test_mae_perfect():
    """MAE should be zero on perfect predictions."""
    y = np.array([0.0, 1.0, -1.0])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_mae_known_value():
    """MAE on a known example."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2 / 3)


def test_r_squared_perfect():
    """R² should be 1.0 on perfect predictions."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_mean_baseline():
    """Predicting the mean should give R² = 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.full(3, np.mean(y_true))
    assert r_squared(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def test_accuracy_all_correct():
    """Accuracy should be 1.0 when all predictions match."""
    y = np.array([0, 1, 2, 1, 0])
    assert accuracy(y, y) == pytest.approx(1.0)


def test_accuracy_known():
    """Accuracy on a known example."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_confusion_matrix_shape():
    """Confusion matrix should be square with size = number of classes."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 2])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)


def test_confusion_matrix_diagonal_perfect():
    """Confusion matrix should be diagonal for perfect predictions."""
    y = np.array([0, 1, 2])
    cm = confusion_matrix(y, y)
    assert np.all(cm == np.diag(np.diag(cm)))
    assert cm.trace() == len(y)


def test_precision_perfect():
    """Precision should be 1.0 when no false positives."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 1])
    assert precision(y_true, y_pred) == pytest.approx(1.0)


def test_recall_perfect():
    """Recall should be 1.0 when no false negatives."""
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 0])
    assert recall(y_true, y_pred) == pytest.approx(1.0)


def test_f1_harmonic_mean():
    """F1 should be the harmonic mean of precision and recall."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    expected_f1 = 2 * p * r / (p + r)
    assert f1_score(y_true, y_pred) == pytest.approx(expected_f1)


def test_precision_no_positive_predictions():
    """Precision should return 0.0 when no positive predictions are made."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert precision(y_true, y_pred) == pytest.approx(0.0)


def test_recall_no_positive_true():
    """Recall should return 0.0 when there are no true positives."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert recall(y_true, y_pred) == pytest.approx(0.0)
