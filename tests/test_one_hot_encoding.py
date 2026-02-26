# pylint: disable=import-outside-toplevel
"""Tests for one_hot_encoding function."""
import pandas as pd
import numpy as np


def test_one_hot_encoding_is_importable_and_callable():
    from ml_project.feature_engineering import one_hot_encoding
    assert callable(one_hot_encoding)


def test_one_hot_encoding_returns_dataframe():
    from ml_project.feature_engineering import one_hot_encoding

    df = pd.DataFrame({
        "season": ["Winter", "Spring", "Summer", "Autumn"],
        "pm25": [10.0, 15.0, 20.0, 25.0]
    })
    result = one_hot_encoding(df)

    assert isinstance(result, pd.DataFrame)


def test_one_hot_encoding_creates_binary_columns():
    from ml_project.feature_engineering import one_hot_encoding

    df = pd.DataFrame({
        "season": ["Winter", "Spring", "Summer", "Autumn"],
        "pm25": [10.0, 15.0, 20.0, 25.0]
    })
    result = one_hot_encoding(df, drop_first=False)

    assert "season" not in result.columns
    season_cols = [col for col in result.columns if col.startswith("season_")]
    assert len(season_cols) > 0


def test_one_hot_encoding_drop_first():
    from ml_project.feature_engineering import one_hot_encoding

    df = pd.DataFrame({
        "season": ["Winter", "Spring", "Summer", "Autumn"],
        "pm25": [10.0, 15.0, 20.0, 25.0]
    })
    result_drop = one_hot_encoding(df, drop_first=True)
    result_no_drop = one_hot_encoding(df, drop_first=False)

    cols_drop = [col for col in result_drop.columns if col.startswith("season_")]
    cols_no_drop = [col for col in result_no_drop.columns if col.startswith("season_")]

    assert len(cols_drop) == len(cols_no_drop) - 1


def test_one_hot_encoding_preserves_numeric_columns():
    from ml_project.feature_engineering import one_hot_encoding

    df = pd.DataFrame({
        "season": ["Winter", "Spring"],
        "pm25": [10.0, 15.0],
        "temperature": [5.0, 10.0]
    })
    result = one_hot_encoding(df)

    assert "pm25" in result.columns
    assert "temperature" in result.columns


def test_one_hot_encoding_handles_no_categorical_columns():
    from ml_project.feature_engineering import one_hot_encoding

    df = pd.DataFrame({
        "pm25": [10.0, 15.0, 20.0],
        "temperature": [5.0, 10.0, 15.0]
    })
    result = one_hot_encoding(df)

    assert result.shape == df.shape
