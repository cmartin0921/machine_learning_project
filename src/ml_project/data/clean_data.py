from pathlib import Path
from typing import Dict

import pandas as pd


WEATHER_NUMERIC_COLUMNS = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]


def _drop_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove accidental header rows that appear as data (seen in raw CSVs).
    """
    header = [str(col) for col in df.columns]
    mask = ~df.astype(str).eq(header).all(axis=1)
    return df.loc[mask].copy()


def _clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_header_rows(df).drop_duplicates()
    df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime_from"] = pd.to_datetime(df["datetime_from"], errors="coerce", utc=True)
    df["datetime_to"] = pd.to_datetime(df["datetime_to"], errors="coerce", utc=True)
    df["timestamp_rollup"] = df["timestamp_rollup"].astype(str).str.lower().str.strip()
    df["metric_name"] = df["metric_name"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sensor_id", "datetime_from", "value"])

    # Remove obviously bad readings (negative particulate concentration).
    df = df[df["value"] >= 0]

    df["sensor_id"] = df["sensor_id"].astype(int)
    df["reading_date"] = df["datetime_from"].dt.floor("D").dt.tz_localize(None)
    df = df.rename(columns={"units": "reading_units"})
    return df.reset_index(drop=True)


def _clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_header_rows(df).drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in WEATHER_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.reset_index(drop=True)


def _clean_locations(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_header_rows(df).drop_duplicates()
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
    df = _drop_header_rows(df).drop_duplicates()
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
    raw_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load, clean, and merge raw CSV datasets into a single DataFrame.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw CSV files. Defaults to `<project_root>/data/raw`.
    output_path:
        Optional path to persist the cleaned, merged dataset (CSV). If omitted,
        the dataset is returned but not saved.

    Returns
    -------
    pandas.DataFrame
        Cleaned and merged dataset containing measurements, sensor metadata,
        location context, and weather observations.
    """
    raw_data: Dict[str, pd.DataFrame] = load_raw_data(raw_dir)
    measurements = _clean_measurements(raw_data["measurements"])
    weather = _clean_weather(raw_data["weather"])
    locations = _clean_locations(raw_data["locations"])
    metadata = _clean_sensor_metadata(raw_data["sensors_metadata"])

    combined = _merge_datasets(measurements, metadata, locations, weather)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return combined
