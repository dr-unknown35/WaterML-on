"""
Tests for WaterML_on.data_utils

ASSUMPTIONS: train_test_split/Kfolds_split live in
WaterML_on.data_utils.splitting; StandardScaler/MinMaxScaler in
WaterML_on.data_utils.scaling; add_intercept in
WaterML_on.data_utils.preprocessing. Adjust import paths if yours differ.
"""
import numpy as np

from WaterML_on.data_utils.splitter import train_test_split, Kfolds_split
from WaterML_on.data_utils.scaler import StandardScaler, MinMaxScaler
from WaterML_on.data_utils.preprocessor import _add_intercept


def test_train_test_split_sizes_and_no_overlap(diabetes_data):
    X, y = diabetes_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    n_samples = X.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == n_samples
    assert X_test.shape[0] == int(n_samples * 0.2)

    # No row should appear in both splits — check via a value-based set
    # comparison since we don't have the raw indices here.
    train_set = {tuple(row) for row in X_train}
    test_set = {tuple(row) for row in X_test}
    assert train_set.isdisjoint(test_set)


def test_train_test_split_is_reproducible_with_same_seed(diabetes_data):
    X, y = diabetes_data
    split_a = train_test_split(X, y, test_size=0.2, random_state=42)
    split_b = train_test_split(X, y, test_size=0.2, random_state=42)
    for a, b in zip(split_a, split_b):
        assert np.array_equal(a, b)


def test_kfold_split_covers_every_sample_exactly_once_as_test(diabetes_data):
    X, y = diabetes_data
    n_samples = X.shape[0]
    nb_folds = 5

    all_test_indices = []
    for train_idx, test_idx in Kfolds_split(n_samples, nb_folds, random_state=0):
        # train and test indices for this fold shouldn't overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        # every fold's train+test should cover the whole dataset
        assert len(train_idx) + len(test_idx) == n_samples
        all_test_indices.extend(test_idx)

    # across all folds, every sample should show up as "test" exactly once
    assert sorted(all_test_indices) == list(range(n_samples))


def test_standard_scaler_produces_zero_mean_unit_std(diabetes_data):
    X, _ = diabetes_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-8)


def test_standard_scaler_uses_training_stats_on_new_data(diabetes_data):
    """
    The whole point of separate fit/transform: transforming a different
    dataset should NOT recompute statistics from that new data.
    """
    X, _ = diabetes_data
    scaler = StandardScaler().fit(X[:300])

    transformed_full = scaler.transform(X)
    # mean of the full transformed set should generally NOT be exactly 0,
    # since it was scaled using only the first 300 rows' statistics
    assert not np.allclose(transformed_full.mean(axis=0), 0, atol=1e-8)


def test_minmax_scaler_bounds_are_zero_and_one(diabetes_data):
    X, _ = diabetes_data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.min(axis=0), 0, atol=1e-8)
    assert np.allclose(X_scaled.max(axis=0), 1, atol=1e-8)


def test_add_intercept_prepends_column_of_ones():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_augmented = _add_intercept(X)

    assert X_augmented.shape == (2, 3)
    assert np.array_equal(X_augmented[:, 0], np.ones(2))
    assert np.array_equal(X_augmented[:, 1:], X)
