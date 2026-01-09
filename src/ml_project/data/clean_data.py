import pandas as pd
import numpy as np

def clean_data(
    sensors_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    sensors_meta_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    cities_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean and merge sensor, weather, location, and city data into a single daily pivot table.
    
    Parameters
    ----------
    sensors_df : pd.DataFrame
        Raw sensor measurements data
    weather_df : pd.DataFrame
        Raw weather data
    sensors_meta_df : pd.DataFrame
        Sensor metadata
    locations_df : pd.DataFrame
        Location data with coordinates
    cities_df : pd.DataFrame
        City information
    
    Returns
    -------
    pd.DataFrame
        Cleaned and merged daily pivot table with sensor metrics and weather data
    """
    
    # Step 1: Merge all sensor-related data
    join_key = "sensor_id"
    overlapping_cols = [
        col for col in sensors_meta_df.columns 
        if col in sensors_df.columns and col != join_key
    ]
    
    sensors_all_data = (
        sensors_df
        .merge(
            sensors_meta_df.drop(columns=overlapping_cols),
            how="left",
            on=join_key
        )
        .merge(
            locations_df[["location_id", "latitude", "longitude", "city_latitude", "city_longitude"]],
            how="left",
            on="location_id"
        )
        .merge(
            cities_df,
            how="left",
            left_on=["city_latitude", "city_longitude"],
            right_on=["latitude", "longitude"],
        )
    )
    
    # Clean up duplicate columns from merges
    cols_to_drop = ["measurement_name", "latitude_y", "longitude_y"]
    cols_to_drop = [c for c in cols_to_drop if c in sensors_all_data.columns]
    sensors_all_data.drop(columns=cols_to_drop, inplace=True)
    
    if "latitude_x" in sensors_all_data.columns:
        sensors_all_data.rename(
            columns={"latitude_x": "latitude", "longitude_x": "longitude"},
            inplace=True
        )
    
    # Step 2: Normalize datetime to date level
    sensors_all_data["datetime"] = (
        pd.to_datetime(sensors_all_data["datetime_to"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    
    # Step 3: Get sensor count per day/city
    sensor_counts = (
        sensors_all_data
        .groupby(["datetime", "city_name", "city_latitude", "city_longitude"])["sensor_id"]
        .nunique()
        .reset_index(name="sensor_count")
    )
    
    # Step 4: Daily mean per metric
    daily_metrics = (
        sensors_all_data
        .groupby(["datetime", "city_name", "city_latitude", "city_longitude", "metric_name"])["value"]
        .mean()
        .reset_index()
    )
    
    # Step 5: Pivot metrics to columns
    sensors_daily_pivot = daily_metrics.pivot(
        index=["datetime", "city_name", "city_latitude", "city_longitude"],
        columns="metric_name",
        values="value"
    ).reset_index()
    
    # Merge sensor count
    sensors_daily_pivot = sensors_daily_pivot.merge(
        sensor_counts,
        on=["datetime", "city_name", "city_latitude", "city_longitude"],
        how="left"
    )
    sensors_daily_pivot.columns.name = None
    
    # Step 6: Fill missing dates in the date range
    sensors_daily_pivot["datetime"] = pd.to_datetime(sensors_daily_pivot["datetime"])
    min_date = sensors_daily_pivot["datetime"].min()
    max_date = sensors_daily_pivot["datetime"].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    
    # Get unique city combinations
    cities = sensors_daily_pivot[["city_name", "city_latitude", "city_longitude"]].drop_duplicates()
    
    # Create a full grid of dates × cities
    full_index = pd.MultiIndex.from_product(
        [full_date_range, cities["city_name"].unique()],
        names=["datetime", "city_name"]
    )
    full_df = pd.DataFrame(index=full_index).reset_index()
    
    # Add city lat/lon back
    full_df = full_df.merge(cities, on="city_name", how="left")
    
    # Merge with existing data (missing dates will have NaN for metrics)
    sensors_daily_pivot = full_df.merge(
        sensors_daily_pivot,
        on=["datetime", "city_name", "city_latitude", "city_longitude"],
        how="left"
    )
    
    # Fill sensor_count with 0 for missing dates
    sensors_daily_pivot["sensor_count"] = sensors_daily_pivot["sensor_count"].fillna(0).astype(int)
    
    # Sort by city and date
    sensors_daily_pivot = sensors_daily_pivot.sort_values(
        ["city_name", "datetime"]
    ).reset_index(drop=True)
    
    # Step 7: Merge weather data
    weather_df["date"] = (
        pd.to_datetime(weather_df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )
    sensors_daily_pivot["datetime"] = (
        pd.to_datetime(sensors_daily_pivot["datetime"])
        .dt.tz_localize(None)
    )
    
    sensors_daily_pivot = sensors_daily_pivot.merge(
        weather_df,
        how="left",
        left_on=["datetime", "city_latitude", "city_longitude"],
        right_on=["date", "latitude", "longitude"]
    )
    
    # Drop duplicate columns from the weather merge
    sensors_daily_pivot = sensors_daily_pivot.drop(
        columns=["date", "latitude", "longitude"], 
        errors="ignore"
    )
    
    return sensors_daily_pivot