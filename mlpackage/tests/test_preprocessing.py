import numpy as np
import pytest
from mlpackage.preprocessing import (
    train_test_split,
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
)


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

def test_split_sizes():
    """train_test_split should produce correctly sized splits."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    assert X_train.shape[0] + X_test.shape[0] == 10
    assert X_test.shape[0] == 3


def test_split_no_overlap():
    """Train and test sets should not share any rows."""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    train_set = set(map(tuple, X_train.tolist()))
    test_set = set(map(tuple, X_test.tolist()))
    assert len(train_set & test_set) == 0


def test_split_reproducible():
    """Same random_state should produce the same split every time."""
    X = np.random.rand(30, 4)
    y = np.random.rand(30)
    split1 = train_test_split(X, y, test_size=0.2, random_state=42)
    split2 = train_test_split(X, y, test_size=0.2, random_state=42)
    for a, b in zip(split1, split2):
        np.testing.assert_array_equal(a, b)


def test_split_invalid_test_size():
    """test_size outside (0, 1) should raise ValueError."""
    X, y = np.ones((10, 2)), np.ones(10)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=0.0)


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

def test_standard_scaler_mean_zero():
    """After StandardScaler fit_transform, each feature should have mean ≈ 0."""
    X = np.array([[1., 10.], [2., 20.], [3., 30.]])
    X_scaled = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-10)


def test_standard_scaler_std_one():
    """After StandardScaler fit_transform, each feature should have std ≈ 1."""
    X = np.array([[1., 10.], [2., 20.], [3., 30.]])
    X_scaled = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-10)


def test_standard_scaler_inverse():
    """inverse_transform should recover the original data."""
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, X, atol=1e-10)


def test_standard_scaler_not_fitted():
    """Calling transform before fit should raise AttributeError."""
    scaler = StandardScaler()
    with pytest.raises(AttributeError):
        scaler.transform(np.ones((3, 2)))


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

def test_minmax_scaler_range():
    """MinMaxScaler should map features exactly to [0, 1] by default."""
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    X_scaled = MinMaxScaler().fit_transform(X)
    assert X_scaled.min() == pytest.approx(0.0)
    assert X_scaled.max() == pytest.approx(1.0)


def test_minmax_scaler_custom_range():
    """MinMaxScaler should respect a custom feature_range."""
    X = np.array([[0.], [5.], [10.]])
    X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    assert X_scaled.min() == pytest.approx(-1.0)
    assert X_scaled.max() == pytest.approx(1.0)


def test_minmax_scaler_inverse():
    """inverse_transform should recover the original data."""
    X = np.array([[1., 50.], [2., 100.], [3., 150.]])
    scaler = MinMaxScaler()
    X_recovered = scaler.inverse_transform(scaler.fit_transform(X))
    np.testing.assert_allclose(X_recovered, X, atol=1e-10)


def test_minmax_invalid_range():
    """feature_range with lo >= hi should raise ValueError."""
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


# ---------------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------------

def test_label_encoder_integers():
    """LabelEncoder should map classes to sorted integer indices."""
    le = LabelEncoder()
    y_enc = le.fit_transform(['cat', 'dog', 'bird', 'cat'])
    # Sorted: bird=0, cat=1, dog=2
    assert list(y_enc) == [1, 2, 0, 1]


def test_label_encoder_inverse():
    """inverse_transform should recover original labels."""
    le = LabelEncoder()
    y = np.array([0, 1, 2, 1, 0])
    le.fit(y)
    np.testing.assert_array_equal(le.inverse_transform(le.transform(y)), y)


def test_label_encoder_unseen_label():
    """transform with unseen labels should raise ValueError."""
    le = LabelEncoder()
    le.fit(['a', 'b', 'c'])
    with pytest.raises(ValueError):
        le.transform(['d'])


def test_label_encoder_not_fitted():
    """transform before fit should raise AttributeError."""
    le = LabelEncoder()
    with pytest.raises(AttributeError):
        le.transform(['a'])
