import pandas as pd

def impute_missing_data(
    df: pd.DataFrame,
    *,
    date_column: str = "reading_date",
    window_size: int = 7,
) -> pd.DataFrame:
    """
    Impute missing values using a 7-day rolling average for numeric features.

    Parameters
    ----------
    df:
        Input dataframe to impute.
    date_column:
        Name of the date column to sort by for rolling average calculation.
        Defaults to "reading_date".
    window_size:
        Size of the rolling window in days for calculating the rolling average.
        Defaults to 7.

    Returns
    -------
    pandas.DataFrame
        Imputed dataframe with missing numeric values filled using rolling average.
    """
    result = df.copy()

    # Sort by date column for proper rolling average calculation
    if date_column in result.columns:
        result = result.sort_values(date_column).reset_index(drop=True)
        # Create helper columns for fallback imputation
        result["_month_year"] = result[date_column].dt.to_period("M")
        result["_month"] = result[date_column].dt.month

    # Get numeric columns, excluding the date column
    numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        if result[col].isna().any():
            # Calculate rolling average with min_periods=1 to handle edge cases
            rolling_avg = result[col].rolling(
                window=window_size, min_periods=1, center=True
            ).mean()

            # Fill NaN values with the rolling average
            result[col] = result[col].fillna(rolling_avg)

            # For any remaining NaN values (e.g., if entire window is NaN),
            # fall back to month-year mean
            if result[col].isna().any() and "_month_year" in result.columns:
                month_year_means = result.groupby("_month_year")[col].transform("mean")
                result[col] = result[col].fillna(month_year_means)

            # For any remaining NaN values, fall back to same month across all years
            if result[col].isna().any() and "_month" in result.columns:
                month_means = result.groupby("_month")[col].transform("mean")
                result[col] = result[col].fillna(month_means)
                
    # Remove helper columns
    result = result.drop(columns=["_month_year", "_month"], errors="ignore")

    return result
