import numpy as np
from ml_project.models.create_model import create_model


def test_predictions_between_zero_and_one():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 3))
    y = rng.random(40)  # targets in [0, 1]

    params = {"model_type": "random_forest", "n_estimators": 5, "random_state": 2, "n_jobs": 1}
    model = create_model(params, x, y)

    preds = model.predict(x)
    assert (preds >= -1e-8).all()
    assert (preds <= 1 + 1e-8).all()
