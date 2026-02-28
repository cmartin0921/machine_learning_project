import pandas as pd

WEATHER_COLS = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]

def _clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize sensor measurements DataFrame.

    Converts types, drops rows with missing critical fields, removes
    clearly invalid readings (e.g. negative particulate values),
    deduplicates and filters out long-interval duplicates.

    Returns a cleaned DataFrame ready for merging.
    """

    df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["datetime_from"] = pd.to_datetime(df["datetime_from"], errors="coerce", utc=True)
    df["datetime_to"] = pd.to_datetime(df["datetime_to"], errors="coerce", utc=True)
    df["timestamp_rollup"] = df["timestamp_rollup"].astype(str).str.lower().str.strip()
    df["metric_name"] = df["metric_name"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sensor_id", "datetime_to", "metric_name", "value"]).copy()

    # Remove obviously bad readings (negative particulate concentration).
    df["value"] = df["value"].where(
        (df["metric_name"] == "temperature") | (df["value"] >= 0)
        , 0
    )

    df["sensor_id"] = df["sensor_id"].astype(int)
    df["reading_date"] = df["datetime_to"].dt.floor("D").dt.tz_localize(None)
    df = df.rename(columns={"units": "reading_units"})

    df = df.drop_duplicates(subset=["sensor_id", "reading_date", "value"], keep="first")

    # Help find sensors with multiple readings a day
    df["is_duplicate_date"] = df.duplicated(subset=["sensor_id", "reading_date"], keep=False)
    df["seconds_count"] = (df["datetime_to"] - df["datetime_from"]).dt.total_seconds()
    df = df[
        # 86400: number of seconds in a day
        ~((df["seconds_count"] > 86400) & (df["is_duplicate_date"]))
    ]
    df = df.drop(columns=["is_duplicate_date", "seconds_count"], errors="ignore")

    return df.reset_index(drop=True)


def _clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Clean MeteoStat daily weather DataFrame.

    Normalizes the `date` column, coerces numeric weather columns and
    fills precipitation/snow missing values with zero. Drops columns
    that contain no data.
    """

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)

    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if col in ("prcp", "snow"):
            df[col] = df[col].fillna(0)
    df = df.dropna(subset=["date"])

    # Removing columns without any data
    cols_with_null = df.isna().sum()
    row_count = df.shape[0]
    cols_without_data = cols_with_null[cols_with_null == row_count]
    df = df.drop(columns=cols_without_data.index, errors="ignore")

    return df.reset_index(drop=True)


def _clean_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean location metadata DataFrame.

    Coerces numeric identifiers and coordinates, parses timestamps and
    drops rows missing `location_id`.
    """

    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["country_id"] = pd.to_numeric(df["country_id"], errors="coerce").astype("Int64")
    df["first_read_at"] = (
        pd.to_datetime(df["first_read_at"], errors="coerce", utc=True).dt.tz_localize(None)
    )
    df["last_read_at"] = (
        pd.to_datetime(df["last_read_at"], errors="coerce", utc=True).dt.tz_localize(None)
    )
    df = df.dropna(subset=["location_id"]).copy()
    df["location_id"] = df["location_id"].astype(int)
    df["country_id"] = df["country_id"].astype(int)

    return df.reset_index(drop=True)


def _clean_sensor_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean sensor metadata DataFrame.

    Normalizes sensor and location identifiers and lowercases
    measurement names. Renames the `units` column to `sensor_units`.
    """

    df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce").astype("Int64")
    df["location_id"] = pd.to_numeric(df["location_id"], errors="coerce").astype("Int64")
    df["measurement_name"] = df["measurement_name"].astype(str).str.lower().str.strip()
    df["measurement"] = df["measurement"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sensor_id", "location_id"]).copy()
    df["sensor_id"] = df["sensor_id"].astype(int)
    df["location_id"] = df["location_id"].astype(int)
    df = df.rename(columns={"units": "sensor_units"})

    return df.reset_index(drop=True)


def _merge_datasets(
    measurements: pd.DataFrame,
    metadata: pd.DataFrame,
    locations: pd.DataFrame,
    weather: pd.DataFrame
) -> pd.DataFrame:
    """Merge cleaned measurements, metadata, locations and weather.

    Returns a combined DataFrame with context columns and sorted by
    `reading_date` and `sensor_id`.
    """

    combined = measurements.merge(metadata, on="sensor_id", how="left", suffixes=("", "_meta"))
    combined = combined.merge(
        locations, on="location_id", how="left", suffixes=("", "_location")
    )
    combined = combined.merge(weather, left_on="reading_date", right_on="date", how="left")
    combined = combined.drop(columns=["date"], errors="ignore")
    combined = combined.rename(columns={"metric_name": "metric", "timestamp_rollup": "rollup"})
    combined = combined.sort_values(["reading_date", "sensor_id"]).reset_index(drop=True)

    return combined

def _data_transform(
    measurements: pd.DataFrame,
    weather: pd.DataFrame
) -> pd.DataFrame:
    """Pivot measurements to daily metrics and merge with weather.

    Produces a per-day aggregated table of measurements (one row per
    date) with sensor counts and joined weather features. Filters out
    days without `pm25` values.
    """
    pivoted_measurements = (
        measurements
        .pivot_table(
            index="reading_date",
            columns="metric_name",
            values="value",
            aggfunc="mean"
        )
    )
    # Count sensors with non-null values per day
    pivoted_measurements["sensor_count"] = (
        measurements
        .dropna(subset=["value"])
        .groupby("reading_date")["sensor_id"]
        .nunique()
    )
    pivoted_measurements = pivoted_measurements.reset_index()

    # Removing columns without any data or where majority of data is missing
    cols_with_null = pivoted_measurements.isna().sum()
    row_count = pivoted_measurements.shape[0]
    # Determines the minimum threshold of data that needs to exist in order to not be dropped
    ratio = 0.5
    cols_without_data = cols_with_null[cols_with_null >= (row_count * ratio)]
    pivoted_measurements = pivoted_measurements.drop(
        columns=cols_without_data.index, errors="ignore"
    )

    combined = (
        pivoted_measurements
            .merge(weather, left_on="reading_date", right_on="date", how="left")
    )
    combined = combined.drop(columns=["date"], errors="ignore")
    if "pm25" in combined.columns:
        combined = combined[combined["pm25"].notna()]

    return combined


def clean_data(
    df_dicts
) -> dict:
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
        location context, and weather observations in the same dictionary package.
        Also included is a cleaned dataset that is to be used for feature engineering
    """

    locations = _clean_locations(df_dicts["locations"])
    metadata = _clean_sensor_metadata(df_dicts["sensors_metadata"])
    measurements = _clean_measurements(df_dicts["sensors_measurements"])
    weather = _clean_weather(df_dicts["weather"])

    combined = _data_transform(measurements, weather)

    if "pm25" in combined.columns:
        combined = combined[combined["pm25"].notna()]

    return {
        "locations": locations,
        "sensors_metadata": metadata,
        "sensors_measurements": measurements,
        "weather": weather,
        "cleaned": combined
    }
