import pandas as pd

def one_hot_encoding(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_first: bool = True
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with categorical columns.
        columns (list[str] | None): List of column names to encode. 
            If None, automatically detects categorical columns (object and category dtypes).
        drop_first (bool): Whether to drop the first category to avoid multicollinearity.
            Default is False.
    
    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    df_encoded = df.copy()
    
    # If no columns specified, auto-detect categorical columns
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude datetime-like columns that might be stored as object
        columns = [col for col in columns if col not in ['reading_date']]
    
    # Filter to only columns that exist in the dataframe
    columns_to_encode = [col for col in columns if col in df_encoded.columns]
    
    if not columns_to_encode:
        # No categorical columns to encode
        return df_encoded
    
    # Apply one-hot encoding using pd.get_dummies
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=columns_to_encode,
        drop_first=drop_first,
        dtype=int  # Use int instead of bool for better compatibility with ML models
    )
    
    return df_encoded

