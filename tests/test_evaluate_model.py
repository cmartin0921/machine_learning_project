import numpy as np
from pathlib import Path

from ml_project.models.create_model import create_model
from ml_project.evaluation.evaluate_model import evaluate_model


def test_evaluate_model_produces_metrics_and_plots(tmp_path):
    rng = np.random.RandomState(1)
    X_train = rng.randn(50, 3)
    y_train = X_train[:, 0] * 1.5 + rng.randn(50) * 0.1

    params = {"model_type": "random_forest", "n_estimators": 5, "random_state": 1, "n_jobs": 1}
    model = create_model(params, X_train, y_train)

    X_test = rng.randn(10, 3)
    y_test = X_test[:, 0] * 1.5 + rng.randn(10) * 0.1

    result = evaluate_model(model, params, X_test, y_test, tmp_path)

    assert isinstance(result, dict)
    assert result.get("model_type") == "random_forest"

    metrics = result.get("metrics", {})
    assert all(k in metrics for k in ("r2", "mse", "mae"))

    plots = result.get("plots", {})
    assert isinstance(plots, dict)
    for p in plots.values():
        assert Path(p).exists()
