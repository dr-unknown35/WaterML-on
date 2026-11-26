import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from WaterML_on.data_utils.preprocessor import _add_intercept, one_hot
from WaterML_on.data_utils.scaler import MinMaxScaler, StandardScaler
from WaterML_on.data_utils.splitter import Kfolds, Stratified_Kfolds


def test_kfolds_split_covers_every_sample_exactly_once_as_test(diabetes_data):
    X, _ = diabetes_data
    folds = list(Kfolds(nb_folds=5, random_state=0).split(X))

    assert len(folds) == 5
    all_test_indices = []
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == X.shape[0]
        all_test_indices.extend(test_idx.tolist())

    assert sorted(all_test_indices) == list(range(X.shape[0]))


def test_kfolds_split_is_reproducible_with_same_seed(diabetes_data):
    X, _ = diabetes_data
    split_a = list(Kfolds(nb_folds=5, random_state=42).split(X))
    split_b = list(Kfolds(nb_folds=5, random_state=42).split(X))

    for (train_a, test_a), (train_b, test_b) in zip(split_a, split_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)


def test_kfolds_split_without_shuffle_uses_contiguous_folds(diabetes_data):
    X, _ = diabetes_data
    folds = list(Kfolds(nb_folds=5, shuffle=False).split(X))

    for fold_idx, (_, test_idx) in enumerate(folds):
        expected = np.array_split(np.arange(X.shape[0]), 5)[fold_idx]
        assert np.array_equal(test_idx, expected)


def test_stratified_kfolds_preserves_class_ratios(breast_cancer_data):
    X, y = breast_cancer_data
    folds = list(Stratified_Kfolds(nb_folds=5, random_state=0).split(X, y))

    assert len(folds) == 5
    all_test_indices = []
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert np.isclose(np.mean(y[test_idx] == 1), np.mean(y == 1), atol=0.03)
        all_test_indices.extend(test_idx.tolist())

    assert sorted(all_test_indices) == list(range(X.shape[0]))


def test_stratified_kfolds_rejects_class_smaller_than_fold_count():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0] * 9 + [1])

    with pytest.raises(ValueError):
        list(Stratified_Kfolds(nb_folds=5, shuffle=False).split(X, y))


def test_standard_scaler_matches_sklearn(diabetes_data):
    X, _ = diabetes_data
    expected = SKStandardScaler().fit_transform(X)
    actual = StandardScaler().fit_transform(X)

    assert np.allclose(actual, expected)


def test_standard_scaler_uses_training_stats_on_new_data(diabetes_data):
    X, _ = diabetes_data
    scaler = StandardScaler().fit(X[:300])

    actual = scaler.transform(X)
    expected = (X - np.mean(X[:300], axis=0)) / np.std(X[:300], axis=0)

    assert np.allclose(actual, expected)
    assert not np.allclose(actual.mean(axis=0), 0, atol=1e-8)


def test_minmax_scaler_matches_sklearn(diabetes_data):
    X, _ = diabetes_data
    expected = SKMinMaxScaler().fit_transform(X)
    actual = MinMaxScaler().fit_transform(X)

    assert np.allclose(actual, expected)
    assert np.allclose(actual.min(axis=0), 0)
    assert np.allclose(actual.max(axis=0), 1)


def test_add_intercept_prepends_column_of_ones():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    actual = _add_intercept(X)

    assert actual.shape == (2, 3)
    assert np.array_equal(actual[:, 0], np.ones(2))
    assert np.array_equal(actual[:, 1:], X)


def test_one_hot_infers_classes_and_respects_true_count():
    y = np.array([2, 0, 2, 1])

    inferred = one_hot(y)
    explicit = one_hot(y, true_count=3)

    assert np.array_equal(inferred, np.eye(3, dtype=int)[y])
    assert np.array_equal(explicit, inferred)
