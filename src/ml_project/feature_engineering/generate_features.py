"""
Feature Generation Module.
This module creates interaction terms and composite variables 
to improve the predictive performance of the PM2.5 regression model.
"""

import pandas as pd

def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generates new features from existing weather and pollutant data.
    
    Args:
        dataframe (pd.DataFrame): The cleaned dataset after KNN imputation.
        
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

    return df_enhanced

def test_feature_generation(dataframe: pd.DataFrame):
    """
    Unit test to verify that the new features were successfully created.
    """
    expected_features = ['temp_wind_interaction', 'temp_diurnal_range', 'pollutant_ratio']
    
    for feature in expected_features:
        assert feature in dataframe.columns, f"Feature {feature} missing!"
    
    print("Unit Test Passed: All interaction features generated correctly.")

def test_data_split(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    REQUIREMENT: Unit testing function.
    Validates that the dimensions of the generated training and testing 
    sets align perfectly to prevent model errors.
    """
    # Verify that the number of rows in features matches the target labels
    assert x_train.shape[0] == y_train.shape[0], "Training features and labels length mismatch!"
    assert x_test.shape[0] == y_test.shape[0], "Testing features and labels length mismatch!"
    
    # Verify that the number of columns (features) is identical in both sets
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of features between train and test sets!"
    
    print("Unit Test Passed: Train-Test Split executed successfully.")
    print(f"-> Training Set Size:   {x_train.shape[0]} rows")
    print(f"-> Testing Set Size:    {x_test.shape[0]} rows")
    print(f"-> Number of Features:  {x_train.shape[1]} columns")