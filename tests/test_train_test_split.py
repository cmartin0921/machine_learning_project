# pylint: disable=import-outside-toplevel
"""Tests for split_dataset function."""
import pandas as pd
import pytest


def test_split_dataset_is_importable_and_callable():
    from ml_project.data import split_dataset
    assert callable(split_dataset)


def test_split_dataset_returns_four_elements():
    from ml_project.data import split_dataset

    df = pd.DataFrame({
        "feature_1": range(10),
        "feature_2": range(10, 20),
        "pm25": range(20, 30)
    })
    result = split_dataset(df, target_col="pm25")

    assert len(result) == 4


def test_split_dataset_dimensions_match():
    from ml_project.data import split_dataset

    df = pd.DataFrame({
        "feature_1": range(10),
        "feature_2": range(10, 20),
        "pm25": range(20, 30)
    })
    x_train, x_test, y_train, y_test = split_dataset(df, target_col="pm25", test_size=0.3)

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]


def test_split_dataset_excludes_target_from_features():
    from ml_project.data import split_dataset

    df = pd.DataFrame({
        "feature_1": range(5),
        "feature_2": range(5, 10),
        "pm25": range(10, 15)
    })
    x_train, x_test, _y_train, _y_test = split_dataset(df, target_col="pm25")

    assert "pm25" not in x_train.columns
    assert "pm25" not in x_test.columns


def test_split_dataset_raises_error_for_missing_target():
    from ml_project.data import split_dataset

    df = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [10, 20, 30]
    })

    with pytest.raises(ValueError):
        split_dataset(df, target_col="nonexistent_column")


def test_split_dataset_respects_test_size():
    from ml_project.data import split_dataset

    df = pd.DataFrame({
        "feature_1": range(100),
        "pm25": range(100)
    })
    x_train, x_test, _y_train, _y_test = split_dataset(df, target_col="pm25", test_size=0.2)

    assert len(x_test) == 20
    assert len(x_train) == 80
