# pylint: disable=import-outside-toplevel
"""Tests for detect_outliers function."""
import pandas as pd


def test_detect_outliers_is_importable_and_callable():
    from ml_project.feature_engineering import detect_outliers
    assert callable(detect_outliers)


def test_detect_outliers_returns_dict():
    from ml_project.feature_engineering import detect_outliers

    df = pd.DataFrame({"value": [10, 15, 20, 25, 30, 35, 40, 45, 50, 200]})
    result = detect_outliers(df)

    assert isinstance(result, dict)


def test_detect_outliers_identifies_extreme_values():
    from ml_project.feature_engineering import detect_outliers

    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})
    result = detect_outliers(df)

    assert "value" in result
    assert len(result["value"]) > 0


def test_detect_outliers_returns_empty_for_normal_data():
    from ml_project.feature_engineering import detect_outliers

    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]})
    result = detect_outliers(df)

    assert len(result) == 0 or "value" not in result


def test_detect_outliers_excludes_specified_columns():
    from ml_project.feature_engineering import detect_outliers

    df = pd.DataFrame({
        "sensor_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],
        "pm25": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    })
    result = detect_outliers(df, exclude_columns=["sensor_id"])

    assert "sensor_id" not in result
