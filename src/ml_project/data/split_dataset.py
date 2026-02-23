from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    dataframe: pd.DataFrame, 
    target_col: str = "pm25", 
    test_size: float = 0.3, 
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates the target variable from the features and splits the data 
    into a training and testing set.
    
    Args:
        dataframe (pd.DataFrame): The fully processed dataset.
        target_col (str): The column representing the target variable to predict.
        test_size (float): The proportion of the dataset to allocate to the test set.
        random_state (int): A seed value to ensure the split is reproducible.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            A tuple containing (X_train, X_test, y_train, y_test).
            
    Raises:
        ValueError: If the target column is missing from the dataframe.
    """
    if target_col not in dataframe.columns:
        raise ValueError(f"Error: Target column '{target_col}' not found in the dataset.")

    # Isolate the independent variables (Features) and dependent variable (Target)
    x_features = dataframe.drop(columns=[target_col])
    y_target = dataframe[target_col]

    # Perform the split
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, 
        y_target, 
        test_size=test_size, 
    )

    return x_train, x_test, y_train, y_test
