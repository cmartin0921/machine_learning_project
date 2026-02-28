import os
from pprint import pprint

import pandas as pd
from dotenv import load_dotenv
import yaml

from openaq import OpenAQ
from ml_project.utils import get_project_directories, setup_logger
from ml_project.data import openaq_extract_data, meteostat_extract_data, split_dataset
from ml_project.cleaning import clean_data
from ml_project.feature_engineering import (
    generate_features,
    handle_outliers,
    detect_outliers,
    impute_missing_data,
    one_hot_encoding,
    scaling,
)
from ml_project.models import create_model
from ml_project.evaluation import evaluate_model
from ml_project.data.MeteoStatClient import MeteoStatClient


def main(extract_data=False):
    directory_paths_dict = get_project_directories()
    env_path = directory_paths_dict["root"] / ".env"
    load_dotenv(dotenv_path=env_path)

    log_file_name = "ml_project.log"
    log_file_path = directory_paths_dict["root"] / "logs"
    logger = setup_logger(
        log_file_path=log_file_path / log_file_name,
        name="ml_project",
    )

    cfg_path = directory_paths_dict["configs"] / "data.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Default state is to not extract data. Would require API keys in order to extract data.
    # In the .zip file, the data is already extracted as .csv files in data/
    if extract_data:
        open_aq_api = os.getenv("OPEN_AQ_API_KEY")
        meteostat_api = os.getenv("METEOSTAT_API_KEY")

        open_aq_client = OpenAQ(api_key=open_aq_api)
        meteo_client = MeteoStatClient(api_key=meteostat_api)

        openaq_extract_data(open_aq_client, cfg, directory_paths_dict)
        open_aq_client.close()

        meteostat_extract_data(meteo_client, cfg, directory_paths_dict)

    logger.info(
        "Begin ML Project with the following params:\n%s",
        yaml.dump(cfg, default_flow_style=False),
    )

    # Loading the data
    locations_path = (
        directory_paths_dict["data_raw"]
        / cfg["outputs"]["files"]["openaq"]["locations"]
    )
    sensors_metadata_path = (
        directory_paths_dict["data_raw"]
        / cfg["outputs"]["files"]["openaq"]["sensors_metadata"]
    )
    sensors_measurements_path = (
        directory_paths_dict["data_raw"]
        / cfg["outputs"]["files"]["openaq"]["sensors_measurements"]
    )
    weather_path = (
        directory_paths_dict["data_raw"]
        / cfg["outputs"]["files"]["meteostat"]["weather_daily"]
    )

    dataframes_dict_raw = {
        "locations": pd.read_csv(locations_path),
        "sensors_metadata": pd.read_csv(sensors_metadata_path),
        "sensors_measurements": pd.read_csv(sensors_measurements_path),
        "weather": pd.read_csv(weather_path),
    }

    for name, df in dataframes_dict_raw.items():
        logger.info("Loaded %s: %d rows, %d columns", name, df.shape[0], df.shape[1])

    # Cleaning the data
    cleaned_data_dict = clean_data(dataframes_dict_raw)
    cleaned_df = cleaned_data_dict["cleaned"]

    logger.info(
        "Cleaned dataframe shape: %d rows, %d columns",
        cleaned_df.shape[0],
        cleaned_df.shape[1],
    )
    logger.info("Columns: %s", list(cleaned_df.columns))

    # Log missing values summary
    missing_counts = cleaned_df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        logger.info("Missing values per column:")
        for col, count in missing_cols.items():
            pct = (count / len(cleaned_df)) * 100
            logger.info("  %s: %d (%.1f%%)", col, count, pct)
    else:
        logger.info("No missing values found")

    # Detect outliers first (for inspection/logging)
    outliers = detect_outliers(cleaned_df)

    if outliers:
        logger.info("Columns with outliers: %s", list(outliers.keys()))
        total_outliers = sum(len(v) for v in outliers.values())
        logger.info("Total outliers detected: %d", total_outliers)
        for col, outlier_list in outliers.items():
            logger.info("  %s: %d outliers", col, len(outlier_list))

        # Handle outliers by capping values at IQR fences
        cleaned_df = handle_outliers(cleaned_df, method="cap")
        logger.info(
            "Outliers capped. Shape after handling: %d rows, %d columns",
            cleaned_df.shape[0],
            cleaned_df.shape[1],
        )
    else:
        logger.info("No outliers detected")

    # Impute missing data
    cleaned_df = impute_missing_data(cleaned_df)
    missing_after = cleaned_df.isna().sum().sum()
    logger.info("Missing values after imputation: %d", missing_after)
    logger.info(
        "Shape after imputation: %d rows, %d columns",
        cleaned_df.shape[0],
        cleaned_df.shape[1],
    )

    featured_df = generate_features(cleaned_df)
    new_features = set(featured_df.columns) - set(cleaned_df.columns)
    logger.info("Generated %d new features", len(new_features))
    if new_features:
        logger.info("New features: %s", list(new_features))
    logger.info(
        "Shape after feature generation: %d rows, %d columns",
        featured_df.shape[0],
        featured_df.shape[1],
    )

    encoded_df = one_hot_encoding(featured_df)
    new_encoded_cols = encoded_df.shape[1] - featured_df.shape[1]
    logger.info("Added %d columns from one-hot encoding", new_encoded_cols)
    logger.info(
        "Shape after one-hot encoding: %d rows, %d columns",
        encoded_df.shape[0],
        encoded_df.shape[1],
    )

    # Train-test split (test_size can be configured in the config files)
    train_test_split_pct = 0.3
    x_train, x_test, y_train, y_test = split_dataset(encoded_df, test_size=train_test_split_pct)
    logger.info(
        "Train set: %d rows, Test set: %d rows",
        len(x_train) if x_train is not None else 0,
        len(x_test) if x_test is not None else 0,
    )

    x_train_scaled, train_scaler = scaling(x_train)
    x_test_scaled, _ = scaling(x_test, existing_scaler=train_scaler)
    logger.info(
        "Scaling complete. Train shape: %s, Test shape: %s",
        x_train_scaled.shape if x_train_scaled is not None else None,
        x_test_scaled.shape if x_test_scaled is not None else None,
    )

    # Load model configuration
    model_cfg_path = directory_paths_dict["configs"] / "model.yaml"
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    model_params = model_cfg.get("model_params", {})
    logger.info("Creating model with params:")
    logger.info("%s", yaml.dump(model_params, default_flow_style=False))

    model = create_model(model_params, x_train_scaled, y_train)
    logger.info(
        "Model created and training complete: %s",
        type(model).__name__ if model else "None",
    )

    # Evaluate the model
    metrics = evaluate_model(
        model,
        model_params,
        x_test_scaled,
        y_test,
        directory_paths_dict["reports"],
    )
    logger.info("Model evaluation complete: %s", metrics)
    pprint(metrics)


if __name__ == "__main__":
    main()
