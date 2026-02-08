import sys
from pathlib import Path

# dodaj root projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pandas as pd
from src.validation.data_validation import validate_dataset, ValidationConfig

sensor_df = pd.read_csv("data/raw/sensors_measurement.csv")
weather_df = pd.read_csv("data/raw/weather_daily.csv")

report = validate_dataset(sensor_df, weather_df, ValidationConfig(stuck_run_length=24))

print(report["summary"])
print("\nNEGATIVE VALUES:\n", report["negative_values"].head(20))
print("\nSTUCK SENSORS:\n", report["stuck_sensors"].head(20))
print("\nOUTLIERS:\n", report["outliers_zscore"].head(20))

if "weather_logic" in report:
    for k, df in report["weather_logic"].items():
        print(f"{k}: {len(df)} rows")
print(report["weather_logic"]["temp_order_violations"].head(10))

