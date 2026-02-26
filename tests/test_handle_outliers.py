# pylint: disable=import-outside-toplevel
"""Tests for handle_outliers function."""
import pandas as pd
import numpy as np


def test_handle_outliers_is_importable_and_callable():
    from ml_project.feature_engineering import handle_outliers
    assert callable(handle_outliers)


def test_handle_outliers_returns_dataframe():
    from ml_project.feature_engineering import handle_outliers

    df = pd.DataFrame({"value": [10, 15, 20, 25, 30, 35, 40, 45, 50, 200]})
    result = handle_outliers(df)

    assert isinstance(result, pd.DataFrame)


def test_handle_outliers_cap_method_clips_values():
    from ml_project.feature_engineering import handle_outliers

    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    result = handle_outliers(df, method="cap")

    assert len(result) == len(df)
    assert result["value"].max() < 1000


def test_handle_outliers_remove_method_removes_rows():
    from ml_project.feature_engineering import handle_outliers

    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    result = handle_outliers(df, method="remove")

    assert len(result) < len(df)
    assert 1000 not in result["value"].values


def test_handle_outliers_preserves_nan_values():
    from ml_project.feature_engineering import handle_outliers

    df = pd.DataFrame({"value": [10, 11, np.nan, 13, 14, 15, 16, 17, 18, 19]})
    result = handle_outliers(df, method="cap")

    assert result["value"].isna().sum() == 1


def test_handle_outliers_respects_exclude_columns():
    from ml_project.feature_engineering import handle_outliers

    df = pd.DataFrame({
        "sensor_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
        "pm25": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    })
    result = handle_outliers(df, exclude_columns=["sensor_id"], method="cap")

    assert result["sensor_id"].max() == 1000
