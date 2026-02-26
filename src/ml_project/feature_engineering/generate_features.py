"""
Feature Generation Module.
This module creates interaction terms and composite variables
to improve the predictive performance of the PM2.5 regression model, and
creates sophisticated cyclical datetime transformations for time-series air quality data.
"""

import numpy as np
import pandas as pd

def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generates new features from existing weather and pollutant data.

    Args:
        dataframe (pd.DataFrame): The cleaned dataset after imputation.

    Returns:
        pd.DataFrame: Dataframe containing original and newly created features.
    """
    # Create a copy to prevent modifying the original dataframe
    df_enhanced = dataframe.copy()

    # 1. Weather Interaction: Temperature * Wind Speed
    # Rationale: Higher wind speeds usually help disperse pollutants,
    # but the effect can vary depending on the temperature.
    if 'tavg' in df_enhanced.columns and 'wspd' in df_enhanced.columns:
        df_enhanced['temp_wind_interaction'] = (
            df_enhanced['tavg'] * df_enhanced['wspd']
        )

    # 2. Atmospheric Stability: Temperature Range
    # Rationale: A large difference between max and min temperature
    # can indicate air stagnation, which traps PM2.5.
    if 'tmax' in df_enhanced.columns and 'tmin' in df_enhanced.columns:
        df_enhanced['temp_diurnal_range'] = (
            df_enhanced['tmax'] - df_enhanced['tmin']
        )

    # 3. Combustion Indicator: NO2 / CO Ratio
    # Rationale: Different ratios can point to traffic vs. industrial sources.
    if 'no2' in df_enhanced.columns and 'co' in df_enhanced.columns:
        # Added 0.001 to prevent division by zero
        df_enhanced['pollutant_ratio'] = (
            df_enhanced['no2'] / (df_enhanced['co'] + 0.001)
        )

    if 'reading_date' in df_enhanced.columns:
        # Convert to datetime object
        df_enhanced['reading_date'] = pd.to_datetime(df_enhanced['reading_date'])

        # Extract numerical time components
        day_of_year = df_enhanced['reading_date'].dt.dayofyear
        month = df_enhanced['reading_date'].dt.month

        # Day of Year Cycle (using 365.25 for leap years)
        df_enhanced['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_enhanced['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Monthly Cycle
        df_enhanced['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Weekend indicator (reduced traffic/industrial activity)
        df_enhanced['is_weekend'] = (df_enhanced['reading_date'].dt.dayofweek >= 5).astype(int)

        # Season classification
        df_enhanced['season'] = df_enhanced['reading_date'].dt.month.apply(
            lambda x: 'Winter' if x in [12, 1, 2]
            else 'Spring' if x in [3, 4, 5]
            else 'Summer' if x in [6, 7, 8]
            else 'Autumn'
        )

    return df_enhanced
