def test_can_import_root_package():
    import ml_project
    assert ml_project is not None


def test_can_import_subpackages():
    from ml_project import data, feature_engineering, models, evaluation

    assert data is not None
    assert feature_engineering is not None
    assert models is not None
    assert evaluation is not None


def test_data_functions_are_importable_and_callable():
    from ml_project.data import load_raw_data, clean_data

    assert callable(load_raw_data)
    assert callable(clean_data)


def test_feature_engineering_functions_are_importable_and_callable():
    from ml_project.feature_engineering import (
        impute_missing_data,
        remove_outliers,
        one_hot_encoding,
        scaling,
        generate_features,
    )

    assert callable(impute_missing_data)
    assert callable(remove_outliers)
    assert callable(one_hot_encoding)
    assert callable(scaling)
    assert callable(generate_features)


def test_models_and_evaluation_are_importable_and_callable():
    from ml_project.models import train_model
    from ml_project.evaluation import evaluate_model

    assert callable(train_model)
    assert callable(evaluate_model)
