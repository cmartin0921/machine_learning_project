# pylint: disable=import-outside-toplevel
import pandas as pd

def test_split_dataset_is_importable_and_callable():
    from ml_project.data import split_dataset

    assert callable(split_dataset)


def test_data_split_dimensions_match():
    """
    REQUIREMENT: Unit testing function.
    Validates that the dimensions of the generated training and testing 
    sets align perfectly to prevent model errors.
    """
    from ml_project.data import split_dataset
    import math

    # Create a sample dataframe for testing
    sample_data = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "pm25": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    })

    test_size=0.3
    x_train, x_test, y_train, y_test = split_dataset(sample_data, target_col="pm25", test_size=test_size)

    # Verify that the number of rows in features matches the target labels
    assert x_train.shape[0] == y_train.shape[0], "Training features and labels length mismatch!"
    assert x_test.shape[0] == y_test.shape[0], "Testing features and labels length mismatch!"
    assert x_train.shape[0] == math.floor(((1 - test_size) * sample_data.shape[0])), "Number of rows for x_train is not expected."
    assert y_train.shape[0] == ((1 - test_size) * sample_data.shape[0]), "Number of rows for y_train is not expected."
    assert x_test.shape[0] == ((test_size) * sample_data.shape[0]), "Number of rows for x_test is not expected."
    assert y_test.shape[0] == ((test_size) * sample_data.shape[0]), "Number of rows for y_test is not expected."

    # Verify that the number of columns (features) is identical in both sets
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of features between train and test sets!"


def test_data_split_raises_error_for_missing_target():
    """
    Validates that split_dataset raises a ValueError when the target column is missing.
    """
    from ml_project.data import split_dataset
    import pytest

    sample_data = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [10, 20, 30]
    })

    with pytest.raises(ValueError):
        split_dataset(sample_data, target_col="nonexistent_column")