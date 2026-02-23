"""
Feature Scaling Module for Air Quality Prediction.
This module standardizes numeric features using StandardScaler to ensure
optimal performance for the Machine Learning model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scaling(dataframe: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
    """
    Standardizes numeric features to have a mean of 0 and a standard deviation of 1.
    
    Args:
        dataframe (pd.DataFrame): The dataset containing raw or engineered features.
        target_col (str): The name of the target variable to be excluded from scaling.
        
    Returns:
        pd.DataFrame: A new dataframe with scaled numeric features.
    """
    # Create a copy to avoid SettingWithCopyWarning in Pandas
    df_scaled = dataframe.copy()

    # Automatically identify numeric columns (float and int)
    numeric_features = df_scaled.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Remove the target variable and constant geographical columns from scaling list
    # Constant values like latitude/longitude can cause errors during standardization
    to_exclude = [target_col, 'latitude', 'longitude', 'snow']
    features_to_scale = [col for col in numeric_features if col not in to_exclude]

    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    
    if features_to_scale:
        df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

    return df_scaled

def test_scaling_integrity(df_scaled: pd.DataFrame):
    """
    REQUIREMENT: Unit testing function to validate data processing.
    Checks if a standardized feature has a mean near 0 and standard deviation near 1.
    """
    # We use 'tavg' as a sample check if it exists in the columns
    if 'tavg' in df_scaled.columns:
        mean_val = df_scaled['tavg'].mean()
        std_val = df_scaled['tavg'].std()
        
        # In computational math, mean won't be exactly 0 but extremely close (e.g., 1e-15)
        assert abs(mean_val) < 0.001, f"Scaling failed: Mean is {mean_val}"
        assert abs(std_val - 1.0) < 0.001, f"Scaling failed: Std is {std_val}"
        
        print("Unit Test Passed: Features successfully standardized (Mean=0, Std=1).")
    else:
        print("Unit Test Warning: 'tavg' column not found for validation.")