# pylint: disable=import-outside-toplevel
import pandas as pd


def test_scaling_is_importable_and_callable():
    from ml_project.feature_engineering import scaling

    assert callable(scaling)


def test_scaling_integrity():
    """
    REQUIREMENT: Unit testing function to validate data processing.
    Checks if a standardized feature has a mean near 0 and standard deviation near 1.
    """
    from ml_project.feature_engineering import scaling

    # Create sample data with numeric features
    sample_data = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "humidity": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    })

    df_scaled = scaling(sample_data, target_col="pm25")

    # Check that 'tavg' has mean near 0 and std near 1
    mean_val = df_scaled["tavg"].mean()
    std_val = df_scaled["tavg"].std()

    assert abs(mean_val) < 0.001, f"Scaling failed: Mean is {mean_val}"
    assert abs(std_val - 1.0) < 0.001, f"Scaling failed: Std is {std_val}"


def test_scaling_excludes_target_column():
    """
    Validates that the target column is not scaled.
    """
    from ml_project.feature_engineering import scaling

    sample_data = pd.DataFrame({
        "tavg": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pm25": [5.0, 10.0, 15.0, 20.0, 25.0]
    })

    original_pm25 = sample_data["pm25"].copy()
    df_scaled = scaling(sample_data, target_col="pm25")

    # Target column should remain unchanged
    pd.testing.assert_series_equal(df_scaled["pm25"], original_pm25)