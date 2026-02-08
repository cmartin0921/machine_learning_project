from __future__ import annotations

from typing import Dict, List, Literal, Sequence, Tuple

import pandas as pd


# Default columns to exclude from outlier detection (non-measurement columns)
DEFAULT_EXCLUDE_COLUMNS = [
    "reading_date", "date",
    "latitude", "longitude",
    "sensor_count",
    "location_id", "sensor_id",
]


def detect_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """
    Detect outliers and return their locations for inspection/testing.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Numeric columns to evaluate. Defaults to all numeric columns
        (excluding those in `exclude_columns`).
    exclude_columns:
        Columns to exclude from outlier detection.
    iqr_multiplier:
        Multiplier for the IQR to compute upper/lower fences.

    Returns
    -------
    Dict[str, List[Tuple[int, float, float, float]]]:
        Dictionary mapping column names to list of tuples containing:
        (row_index, value, lower_fence, upper_fence) for each outlier.
    """
    # Determine which columns to exclude
    exclude_set = set(exclude_columns) if exclude_columns is not None else set(DEFAULT_EXCLUDE_COLUMNS)
    
    # Determine which columns to process
    if columns is not None:
        num_cols: List[str] = [c for c in columns if c not in exclude_set]
    else:
        all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in all_numeric if c not in exclude_set]
    
    outliers: Dict[str, List[Tuple[int, float, float, float]]] = {}
    
    for col in num_cols:
        series = df[col]
        if series.dropna().empty:
            continue
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        
        # Find outlier indices (excluding NaN values)
        outlier_mask = ~series.between(lower, upper) & series.notna()
        outlier_indices = df.index[outlier_mask].tolist()
        
        if outlier_indices:
            outliers[col] = [
                (idx, series.loc[idx], lower, upper)
                for idx in outlier_indices
            ]
    
    return outliers


def handle_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    exclude_columns: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
    method: Literal["remove", "cap"] = "cap",
) -> pd.DataFrame:
    """
    Handle outliers using the IQR method for the specified numeric columns.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Numeric columns to evaluate. Defaults to all numeric columns
        (excluding those in `exclude_columns`).
    exclude_columns:
        Columns to exclude from outlier detection. Defaults to common
        non-measurement columns like 'reading_date', 'latitude', 'longitude',
        'sensor_count', etc.
    iqr_multiplier:
        Multiplier for the IQR to compute upper/lower fences. Typical choices are
        1.5 (moderate) or 3.0 (conservative).
    method:
        How to handle outliers:
        - "remove": Remove rows containing outliers (original behavior)
        - "cap": Cap outlier values at the fence boundaries (winsorization)

    Returns
    -------
    pandas.DataFrame
        Dataframe with outliers handled according to the specified method.
        NaNs are preserved.
    """
    df_out = df.copy()
    
    # Determine which columns to exclude
    exclude_set = set(exclude_columns) if exclude_columns is not None else set(DEFAULT_EXCLUDE_COLUMNS)
    
    # Determine which columns to process
    if columns is not None:
        num_cols: List[str] = [c for c in columns if c not in exclude_set]
    else:
        all_numeric = df_out.select_dtypes(include=["number"]).columns.tolist()
        num_cols = [c for c in all_numeric if c not in exclude_set]
    
    if not num_cols:
        return df_out

    if method == "remove":
        # Original behavior: remove rows with outliers
        mask = pd.Series(True, index=df_out.index)
        for col in num_cols:
            series = df_out[col]
            if series.dropna().empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask &= series.between(lower, upper) | series.isna()
        return df_out.loc[mask].reset_index(drop=True)
    
    else:  # method == "cap"
        # Cap outliers at the fence boundaries (winsorization)
        for col in num_cols:
            series = df_out[col]
            if series.dropna().empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            df_out[col] = series.clip(lower=lower, upper=upper)
        return df_out
