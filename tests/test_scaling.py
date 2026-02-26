# pylint: disable=import-outside-toplevel
"""Tests for scaling function."""
import pandas as pd


def test_scaling_is_importable_and_callable():
    from ml_project.feature_engineering import scaling
    assert callable(scaling)


def test_scaling_returns_tuple():
    from ml_project.feature_engineering import scaling

    df = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0]
    })
    result = scaling(df, target_col="pm25")

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_scaling_standardizes_features():
    from ml_project.feature_engineering import scaling

    df = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    })
    df_scaled, scaler = scaling(df, target_col="pm25")

    mean_val = df_scaled["tavg"].mean()
    std_val = df_scaled["tavg"].std()

    assert abs(mean_val) < 0.01
    assert abs(std_val - 1.0) < 0.15


def test_scaling_excludes_target_column():
    from ml_project.feature_engineering import scaling

    df = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0]
    })
    original_pm25 = df["pm25"].copy()
    df_scaled, scaler = scaling(df, target_col="pm25")

    pd.testing.assert_series_equal(df_scaled["pm25"], original_pm25)


def test_scaling_with_existing_scaler():
    from ml_project.feature_engineering import scaling

    train_df = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0]
    })
    test_df = pd.DataFrame({
        "tavg": [15.0, 25.0, 35.0],
        "pm25": [7.0, 12.0, 17.0]
    })

    train_scaled, train_scaler = scaling(train_df, target_col="pm25")
    test_scaled, _ = scaling(test_df, target_col="pm25", existing_scaler=train_scaler)

    assert isinstance(test_scaled, pd.DataFrame)
    assert test_scaled.shape == test_df.shape


def test_scaling_returns_scaler_object():
    from ml_project.feature_engineering import scaling
    from sklearn.preprocessing import StandardScaler

    df = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0]
    })
    df_scaled, scaler = scaling(df, target_col="pm25")

    assert isinstance(scaler, StandardScaler)