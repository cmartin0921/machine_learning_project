"""
Feature Scaling Module for Air Quality Prediction.
This module standardizes numeric features using StandardScaler to ensure
optimal performance for the Machine Learning model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# IMPORTANT: To prevent data leakage, scaling must be done as follows:
#   1. FIT the scaler on the TRAINING data only (learn mean/std or min/max)
#   2. TRANSFORM both training and test data using those same parameters

def scaling(
    dataframe: pd.DataFrame, target_col: str = 'pm25', existing_scaler=None
) -> pd.DataFrame:
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
    to_exclude = [target_col, 'latitude', 'longitude', 'snow', 'reading_date']
    features_to_scale = [col for col in numeric_features if col not in to_exclude]

    # Initialize and apply StandardScaler
    if existing_scaler is None:
        scaler = StandardScaler()
    else:
        scaler = existing_scaler

    if features_to_scale:
        df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

    return df_scaled, scaler
