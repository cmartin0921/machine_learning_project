# pylint: disable=import-outside-toplevel
"""Tests for impute_missing_data function."""
import pandas as pd
import numpy as np


def test_impute_missing_data_is_importable_and_callable():
    from ml_project.feature_engineering import impute_missing_data
    assert callable(impute_missing_data)


def test_impute_missing_data_returns_dataframe():
    from ml_project.feature_engineering import impute_missing_data

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=10),
        "pm25": [10, np.nan, 12, 13, np.nan, 15, 16, 17, 18, 19]
    })
    result = impute_missing_data(df)

    assert isinstance(result, pd.DataFrame)


def test_impute_missing_data_fills_nan_values():
    from ml_project.feature_engineering import impute_missing_data

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=10),
        "pm25": [10, np.nan, 12, 13, np.nan, 15, 16, 17, 18, 19]
    })
    result = impute_missing_data(df)

    assert result["pm25"].isna().sum() == 0


def test_impute_missing_data_preserves_non_null_values():
    from ml_project.feature_engineering import impute_missing_data

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=5),
        "pm25": [10.0, np.nan, 12.0, 13.0, 14.0]
    })
    result = impute_missing_data(df)

    assert 10.0 in result["pm25"].values
    assert 12.0 in result["pm25"].values
    assert 13.0 in result["pm25"].values
    assert 14.0 in result["pm25"].values


def test_impute_missing_data_handles_multiple_columns():
    from ml_project.feature_engineering import impute_missing_data

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=5),
        "pm25": [10.0, np.nan, 12.0, 13.0, 14.0],
        "temperature": [20.0, 21.0, np.nan, 23.0, 24.0]
    })
    result = impute_missing_data(df)

    assert result["pm25"].isna().sum() == 0
    assert result["temperature"].isna().sum() == 0
