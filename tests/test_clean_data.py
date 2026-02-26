# pylint: disable=import-outside-toplevel
"""Tests for clean_data function."""
import pandas as pd
import numpy as np


def test_clean_data_is_importable_and_callable():
    from ml_project.cleaning import clean_data
    assert callable(clean_data)


def test_clean_data_returns_expected_keys():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    expected_keys = ["locations", "sensors_metadata", "sensors_measurements", "weather", "cleaned"]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_clean_data_output_is_dataframe():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert isinstance(result["cleaned"], pd.DataFrame)


# =============================================================================
# MEASUREMENTS CLEANING TESTS
# =============================================================================

def test_measurements_sensor_id_converted_to_int():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"]["sensor_id"] = ["101", "102"]  # strings
    result = clean_data(df_dicts)

    assert result["sensors_measurements"]["sensor_id"].dtype in [np.int64, np.int32, int]


def test_measurements_value_converted_to_numeric():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"]["value"] = ["25.5", "30.0"]  # strings
    result = clean_data(df_dicts)

    assert pd.api.types.is_numeric_dtype(result["sensors_measurements"]["value"])


def test_measurements_datetime_parsed():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert pd.api.types.is_datetime64_any_dtype(result["sensors_measurements"]["datetime_to"])
    assert pd.api.types.is_datetime64_any_dtype(result["sensors_measurements"]["datetime_from"])


def test_measurements_metric_name_normalized_to_lowercase():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"]["metric_name"] = ["  PM25  ", "  NO2  "]
    result = clean_data(df_dicts)

    assert result["sensors_measurements"]["metric_name"].iloc[0] == "pm25"
    assert result["sensors_measurements"]["metric_name"].iloc[1] == "no2"


def test_measurements_negative_pollutant_values_set_to_zero():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"]["value"] = [-5.0, 25.0]
    df_dicts["sensors_measurements"]["metric_name"] = ["pm25", "pm25"]
    result = clean_data(df_dicts)

    assert (result["sensors_measurements"]["value"] >= 0).all()


def test_measurements_negative_temperature_preserved():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"] = pd.DataFrame({
        "sensor_id": [101],
        "value": [-10.0],
        "datetime_from": ["2023-01-15T00:00:00Z"],
        "datetime_to": ["2023-01-15T23:59:59Z"],
        "timestamp_rollup": ["daily"],
        "metric_name": ["temperature"],
        "units": ["°C"]
    })
    result = clean_data(df_dicts)

    assert -10.0 in result["sensors_measurements"]["value"].values


def test_measurements_reading_date_column_created():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert "reading_date" in result["sensors_measurements"].columns
    assert pd.api.types.is_datetime64_any_dtype(result["sensors_measurements"]["reading_date"])


def test_measurements_units_renamed_to_reading_units():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert "reading_units" in result["sensors_measurements"].columns
    assert "units" not in result["sensors_measurements"].columns


def test_measurements_drops_rows_with_null_required_fields():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_measurements"] = pd.DataFrame({
        "sensor_id": [101, None, 103],
        "value": [25.0, 30.0, None],
        "datetime_from": ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z", "2023-06-03T00:00:00Z"],
        "datetime_to": ["2023-06-01T23:59:59Z", "2023-06-02T23:59:59Z", "2023-06-03T23:59:59Z"],
        "timestamp_rollup": ["daily", "daily", "daily"],
        "metric_name": ["pm25", "pm25", "pm25"],
        "units": ["µg/m³", "µg/m³", "µg/m³"]
    })
    result = clean_data(df_dicts)

    # Should only have 1 row (101 with value 25.0)
    assert len(result["sensors_measurements"]) == 1


# =============================================================================
# WEATHER CLEANING TESTS
# =============================================================================

def test_weather_date_parsed():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert pd.api.types.is_datetime64_any_dtype(result["weather"]["date"])


def test_weather_numeric_columns_converted():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["weather"]["tavg"] = ["20.0", "22.0"]  # strings
    df_dicts["weather"]["wspd"] = ["10.0", "12.0"]  # strings
    result = clean_data(df_dicts)

    assert pd.api.types.is_numeric_dtype(result["weather"]["tavg"])
    assert pd.api.types.is_numeric_dtype(result["weather"]["wspd"])


def test_weather_prcp_snow_fillna_with_zero():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["weather"]["prcp"] = [None, 5.0]
    df_dicts["weather"]["snow"] = [None, None]
    result = clean_data(df_dicts)

    assert result["weather"]["prcp"].isna().sum() == 0
    assert result["weather"]["snow"].isna().sum() == 0
    assert result["weather"]["prcp"].iloc[0] == 0.0


def test_weather_drops_empty_columns():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["weather"]["tsun"] = [None, None]  # all nulls
    result = clean_data(df_dicts)

    assert "tsun" not in result["weather"].columns


# =============================================================================
# LOCATIONS CLEANING TESTS
# =============================================================================

def test_locations_ids_converted_to_int():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["locations"]["location_id"] = ["1", "2"]  # strings
    df_dicts["locations"]["country_id"] = ["1", "1"]  # strings
    result = clean_data(df_dicts)

    assert result["locations"]["location_id"].dtype in [np.int64, np.int32, int]
    assert result["locations"]["country_id"].dtype in [np.int64, np.int32, int]


def test_locations_coordinates_converted_to_numeric():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["locations"]["latitude"] = ["52.2297", "52.2298"]
    df_dicts["locations"]["longitude"] = ["21.0122", "21.0123"]
    result = clean_data(df_dicts)

    assert pd.api.types.is_numeric_dtype(result["locations"]["latitude"])
    assert pd.api.types.is_numeric_dtype(result["locations"]["longitude"])


def test_locations_datetime_parsed():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert pd.api.types.is_datetime64_any_dtype(result["locations"]["first_read_at"])
    assert pd.api.types.is_datetime64_any_dtype(result["locations"]["last_read_at"])


def test_locations_drops_rows_without_location_id():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["locations"] = pd.DataFrame({
        "location_id": [1, None],
        "latitude": [52.2297, 52.2298],
        "longitude": [21.0122, 21.0123],
        "country_id": [1, 1],
        "first_read_at": ["2023-01-01", "2023-01-01"],
        "last_read_at": ["2023-12-31", "2023-12-31"]
    })
    result = clean_data(df_dicts)

    assert len(result["locations"]) == 1


# =============================================================================
# SENSOR METADATA CLEANING TESTS
# =============================================================================

def test_metadata_ids_converted_to_int():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_metadata"]["sensor_id"] = ["101", "102"]
    df_dicts["sensors_metadata"]["location_id"] = ["1", "2"]
    result = clean_data(df_dicts)

    assert result["sensors_metadata"]["sensor_id"].dtype in [np.int64, np.int32, int]
    assert result["sensors_metadata"]["location_id"].dtype in [np.int64, np.int32, int]


def test_metadata_strings_normalized_to_lowercase():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    df_dicts["sensors_metadata"]["measurement_name"] = ["  PM2.5  ", "  NO2  "]
    df_dicts["sensors_metadata"]["measurement"] = ["  PM25  ", "  NO2  "]
    result = clean_data(df_dicts)

    assert result["sensors_metadata"]["measurement_name"].iloc[0] == "pm2.5"
    assert result["sensors_metadata"]["measurement"].iloc[0] == "pm25"


def test_metadata_units_renamed_to_sensor_units():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert "sensor_units" in result["sensors_metadata"].columns
    assert "units" not in result["sensors_metadata"].columns


# =============================================================================
# DATA TRANSFORM / CLEANED OUTPUT TESTS
# =============================================================================

def test_cleaned_output_has_reading_date():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert "reading_date" in result["cleaned"].columns


def test_cleaned_output_has_weather_columns():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    weather_cols = ["tavg", "tmin", "tmax"]
    for col in weather_cols:
        assert col in result["cleaned"].columns


def test_cleaned_output_has_sensor_count():
    from ml_project.cleaning import clean_data

    df_dicts = _create_sample_raw_data()
    result = clean_data(df_dicts)

    assert "sensor_count" in result["cleaned"].columns


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_sample_raw_data():
    """Create sample raw data dictionary for testing."""
    locations = pd.DataFrame({
        "location_id": [1, 2],
        "latitude": [52.2297, 52.2298],
        "longitude": [21.0122, 21.0123],
        "country_id": [1, 1],
        "first_read_at": ["2023-01-01", "2023-01-01"],
        "last_read_at": ["2023-12-31", "2023-12-31"]
    })

    sensors_metadata = pd.DataFrame({
        "sensor_id": [101, 102],
        "location_id": [1, 2],
        "measurement_name": ["PM2.5", "NO2"],
        "measurement": ["pm25", "no2"],
        "units": ["µg/m³", "µg/m³"]
    })

    sensors_measurements = pd.DataFrame({
        "sensor_id": [101, 102],
        "value": [25.0, 30.0],
        "datetime_from": ["2023-06-01T00:00:00Z", "2023-06-02T00:00:00Z"],
        "datetime_to": ["2023-06-01T23:59:59Z", "2023-06-02T23:59:59Z"],
        "timestamp_rollup": ["daily", "daily"],
        "metric_name": ["pm25", "no2"],
        "units": ["µg/m³", "µg/m³"]
    })

    weather = pd.DataFrame({
        "date": ["2023-06-01", "2023-06-02"],
        "tavg": [20.0, 22.0],
        "tmin": [15.0, 17.0],
        "tmax": [25.0, 27.0],
        "prcp": [0.0, 0.0],
        "snow": [0.0, 0.0],
        "wdir": [180.0, 190.0],
        "wspd": [10.0, 12.0]
    })

    return {
        "locations": locations,
        "sensors_metadata": sensors_metadata,
        "sensors_measurements": sensors_measurements,
        "weather": weather
    }
