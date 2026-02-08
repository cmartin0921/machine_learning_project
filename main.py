import os
import pandas as pd
from dotenv import load_dotenv
import yaml

from openaq import OpenAQ
from ml_project.utils import get_project_directories, setup_logger
from ml_project.data import openaq_extract_data, meteostat_extract_data, clean_data
from ml_project.feature_engineering import (
    generate_features, impute_missing_data,
    one_hot_encoding, remove_outliers,
    scaling
)
from ml_project.data.MeteoStatClient import MeteoStatClient

def main():
    directory_paths_dict = get_project_directories()
    env_path = directory_paths_dict["root"] / ".env"
    load_dotenv(dotenv_path=env_path)

    log_file_name = "ml_project.log"
    log_file_path = directory_paths_dict["root"] / "logs"
    logger = setup_logger(log_file_path=log_file_path / log_file_name, name="ml_project")

    open_aq_api = os.getenv("OPEN_AQ_API_KEY")
    meteostat_api = os.getenv("METEOSTAT_API_KEY")

    open_aq_client = OpenAQ(api_key=open_aq_api)
    meteo_client = MeteoStatClient(api_key=meteostat_api)

    cfg_path = directory_paths_dict["configs"] / "data.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("Begin ML Project with the following params:\n%s", yaml.dump(cfg, default_flow_style=False))

    openaq_extract_data(open_aq_client, cfg, directory_paths_dict)
    open_aq_client.close()

    meteostat_extract_data(meteo_client, cfg, directory_paths_dict)

    # Loading the data
    locations_df_raw = pd.read_csv(directory_paths_dict["data_raw"] / cfg["outputs"]["files"]["openaq"]["locations"])
    sensors_metadata_df_raw = pd.read_csv(directory_paths_dict["data_raw"] / cfg["outputs"]["files"]["openaq"]["sensors_metadata"])
    sensors_measurements_df_raw = pd.read_csv(directory_paths_dict["data_raw"] / cfg["outputs"]["files"]["openaq"]["sensors_measurements"])
    weather_daily_df_raw = pd.read_csv(directory_paths_dict["data_raw"] / cfg["outputs"]["files"]["meteostat"]["weather_daily"])

    dataframes_dict = {
        "locations": locations_df_raw,
        "sensors_metadata": sensors_metadata_df_raw,
        "sensors_measurements": sensors_measurements_df_raw,
        "weather": weather_daily_df_raw
    }

    # # Cleaning the data
    cleaned_dataframes_dict = clean_data(dataframes_dict)

if __name__ == "__main__":
    main()