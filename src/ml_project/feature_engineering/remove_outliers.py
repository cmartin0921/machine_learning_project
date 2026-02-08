from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def remove_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers using the IQR method for the specified numeric columns.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Numeric columns to evaluate. Defaults to all numeric columns.
    iqr_multiplier:
        Multiplier for the IQR to compute upper/lower fences. Typical choices are
        1.5 (moderate) or 3.0 (conservative).

    Returns
    -------
    pandas.DataFrame
        Dataframe with outlier rows removed (NaNs are preserved).
    """
    df_out = df.copy()
    num_cols: List[str] = list(columns) if columns is not None else list(df_out.select_dtypes(include=["number"]).columns)
    if not num_cols:
        return df_out

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
