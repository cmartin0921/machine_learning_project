import os
import csv
import pandas as pd
from dotenv import load_dotenv
import yaml

from openaq import OpenAQ
from ml_project.utils import get_project_directories
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

    open_aq_api = os.getenv("OPEN_AQ_API_KEY")
    meteostat_api = os.getenv("METEOSTAT_API_KEY")

    open_aq_client = OpenAQ(api_key=open_aq_api)
    meteo_client = MeteoStatClient(api_key=meteostat_api)

    cfg_path = directory_paths_dict["configs"] / "data.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openaq_extract_data(open_aq_client, cfg, directory_paths_dict)
    open_aq_client.close()

    meteostat_extract_data(meteo_client, cfg, directory_paths_dict)
    # meteo_client.close()

    # # Loading the data
    # sensors_df_raw = pd.read_csv(directory_paths_dict['data_raw'] / "sensors_measurement.csv")
    # weather_df_raw = pd.read_csv(directory_paths_dict['data_raw'] / "weather_daily.csv")

    # # Cleaning the data
    # sensors_df, weather_df = clean_data(sensors_df_raw, weather_df_raw)

if __name__ == "__main__":
    main()