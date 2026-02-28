from pathlib import Path
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_model(model, model_params, x_test, y_test, reports_dir, logger=None):
    """
    Evaluate a trained model on test data.
    Returns a structured dict with metrics + plot paths.
    """

    metrics = {}
    plots = {}

    # reports folder
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # same pattern as create_model
    model_type = model_params.get("model_type", "random_forest")

    # predict once
    y_pred = model.predict(x_test)
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_test - y_pred

    if model_type == "random_forest":

        # ---------- metrics ----------
        metrics["r2"] = r2_score(y_test, y_pred)
        metrics["mse"] = mean_squared_error(y_test, y_pred)
        metrics["mae"] = mean_absolute_error(y_test, y_pred)

        # ---------- plot 1: Actual vs Predicted ----------
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val])

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")

        plot_path = reports_path / "actual_vs_predicted.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        plots["actual_vs_predicted"] = str(plot_path)

        # ---------- plot 2: Predicted vs Residuals ----------
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals)

        ax.axhline(0)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Predicted vs Residuals")

        plot_path = reports_path / "predicted_vs_residuals.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        plots["predicted_vs_residuals"] = str(plot_path)

        # ---------- plot 3: Residual distribution ----------
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=30)

        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Distribution")

        plot_path = reports_path / "residual_distribution.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        plots["residual_distribution"] = str(plot_path)

        # ---------- plot 4: Q-Q plot ----------
        fig, ax = plt.subplots()
        stats.probplot(residuals, dist="norm", plot=ax)

        ax.set_title("Q-Q Plot (Residuals)")

        plot_path = reports_path / "qq_plot_residuals.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        plots["qq_plot_residuals"] = str(plot_path)

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Supported options: ['random_forest']"
        )

    # ---------- logging ----------
    if logger is not None:
        logger.info(f"model_type={model_type}")

        for k, v in metrics.items():
            logger.info(f"{k}={v}")

        for k, p in plots.items():
            logger.info(f"plot:{k} -> {p}")

    return {
        "model_type": model_type,
        "metrics": metrics,
        "plots": plots,
    }