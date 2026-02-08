from pathlib import Path
from typing import Dict

import pandas as pd

WEATHER_COLS = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]

def _clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime_from"] = pd.to_datetime(df["datetime_from"], errors="coerce", utc=True)
    df["datetime_to"] = pd.to_datetime(df["datetime_to"], errors="coerce", utc=True)
    df["timestamp_rollup"] = df["timestamp_rollup"].astype(str).str.lower().str.strip()
    df["metric_name"] = df["metric_name"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sensor_id", "datetime_to", "value"])

    # Remove obviously bad readings (negative particulate concentration).
    df["value"] = df["value"].where(
        (df["metric_name"] == "temperature") | (df["value"] >= 0)
        , 0
    )

    df["sensor_id"] = df["sensor_id"].astype(int)
    df["reading_date"] = df["datetime_to"].dt.floor("D").dt.tz_localize(None)
    df = df.rename(columns={"units": "reading_units"})
    
    return df.reset_index(drop=True)


def _clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if col in ("prcp", "snow"):
            df[col] = df[col].fillna(0)
    df = df.dropna(subset=["date"])
    
    return df.reset_index(drop=True)


def _clean_locations(df: pd.DataFrame) -> pd.DataFrame:
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["country_id"] = pd.to_numeric(df["country_id"], errors="coerce").astype("Int64")
    df["first_read_at"] = pd.to_datetime(df["first_read_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["last_read_at"] = pd.to_datetime(df["last_read_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["location_id"])
    df["location_id"] = df["location_id"].astype(int)
    df["country_id"] = df["country_id"].astype(int)
    
    return df.reset_index(drop=True)


def _clean_sensor_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce").astype("Int64")
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
    df["measurement_name"] = df["measurement_name"].astype(str).str.lower().str.strip()
    df["measurement"] = df["measurement"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sensor_id", "location_id"])
    df["sensor_id"] = df["sensor_id"].astype(int)
    df["location_id"] = df["location_id"].astype(int)
    df = df.rename(columns={"units": "sensor_units"})
    
    return df.reset_index(drop=True)


def _merge_datasets(measurements: pd.DataFrame, metadata: pd.DataFrame, locations: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    combined = measurements.merge(metadata, on="sensor_id", how="left", suffixes=("", "_meta"))
    combined = combined.merge(locations, on="location_id", how="left", suffixes=("", "_location"))
    combined = combined.merge(weather, left_on="reading_date", right_on="date", how="left")
    combined = combined.drop(columns=["date"], errors="ignore")
    combined = combined.rename(columns={"metric_name": "metric", "timestamp_rollup": "rollup"})
    combined = combined.sort_values(["reading_date", "sensor_id"]).reset_index(drop=True)
    
    return combined


def clean_data(
    df_dicts
) -> pd.DataFrame:
    """
    Load, clean, and merge raw CSV datasets into a single DataFrame.

    Parameters
    ----------
    df_dicts:
        Dictionary containing all dataframes that is to be cleaned.

    Returns
    -------
    dict:
        Cleaned and merged dataset containing measurements, sensor metadata,
        location context, and weather observations in the same dictionary package
    """

    # locations = _clean_locations(df_dicts["locations"])
    # metadata = _clean_sensor_metadata(df_dicts["sensors_metadata"])
    # measurements = _clean_measurements(df_dicts["sensors_measurements"])
    weather = _clean_weather(df_dicts["weather"])

    # combined = _merge_datasets(measurements, metadata, locations, weather)

    return weather
