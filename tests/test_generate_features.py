# pylint: disable=import-outside-toplevel
"""Tests for generate_features function."""
import pandas as pd


def test_generate_features_is_importable_and_callable():
    from ml_project.feature_engineering import generate_features
    assert callable(generate_features)


def test_generate_features_returns_dataframe():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=5),
        "tavg": [10.0, 12.0, 14.0, 16.0, 18.0],
        "wspd": [5.0, 6.0, 7.0, 8.0, 9.0]
    })
    result = generate_features(df)

    assert isinstance(result, pd.DataFrame)


def test_generate_features_creates_temp_wind_interaction():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({"tavg": [10.0, 20.0], "wspd": [5.0, 10.0]})
    result = generate_features(df)

    assert "temp_wind_interaction" in result.columns
    assert result["temp_wind_interaction"].iloc[0] == 50.0
    assert result["temp_wind_interaction"].iloc[1] == 200.0


def test_generate_features_creates_temp_diurnal_range():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({"tmax": [25.0, 30.0], "tmin": [15.0, 18.0]})
    result = generate_features(df)

    assert "temp_diurnal_range" in result.columns
    assert result["temp_diurnal_range"].iloc[0] == 10.0
    assert result["temp_diurnal_range"].iloc[1] == 12.0


def test_generate_features_creates_cyclical_date_features():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({"reading_date": pd.date_range("2023-01-01", periods=5)})
    result = generate_features(df)

    expected_cols = ["day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos"]
    for col in expected_cols:
        assert col in result.columns


def test_generate_features_creates_is_weekend():
    from ml_project.feature_engineering import generate_features

    # 2023-01-06=Fri, 2023-01-07=Sat, 2023-01-08=Sun, 2023-01-09=Mon
    df = pd.DataFrame({
        "reading_date": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09"])
    })
    result = generate_features(df)

    assert "is_weekend" in result.columns
    assert result["is_weekend"].iloc[0] == 0  # Friday
    assert result["is_weekend"].iloc[1] == 1  # Saturday
    assert result["is_weekend"].iloc[2] == 1  # Sunday
    assert result["is_weekend"].iloc[3] == 0  # Monday


def test_generate_features_creates_season():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({
        "reading_date": pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"])
    })
    result = generate_features(df)

    assert "season" in result.columns
    assert result["season"].iloc[0] == "Winter"
    assert result["season"].iloc[1] == "Spring"
    assert result["season"].iloc[2] == "Summer"
    assert result["season"].iloc[3] == "Autumn"


def test_generate_features_preserves_original_columns():
    from ml_project.feature_engineering import generate_features

    df = pd.DataFrame({
        "reading_date": pd.date_range("2023-01-01", periods=3),
        "pm25": [10.0, 15.0, 20.0],
        "tavg": [5.0, 10.0, 15.0]
    })
    result = generate_features(df)

    assert "pm25" in result.columns
    assert "tavg" in result.columns
