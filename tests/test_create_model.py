import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from ml_project.models.create_model import create_model


def test_predictions_between_zero_and_one():
    rng = np.random.RandomState(2)
    X = rng.randn(40, 3)
    y = rng.rand(40)  # targets in [0, 1]

    params = {"model_type": "random_forest", "n_estimators": 5, "random_state": 2, "n_jobs": 1}
    model = create_model(params, X, y)

    preds = model.predict(X)
    assert (preds >= -1e-8).all()
    assert (preds <= 1 + 1e-8).all()
