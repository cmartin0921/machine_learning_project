from __future__ import annotations

from typing import Sequence

import pandas as pd


def one_hot_encoding(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Columns to encode. Defaults to object/category/bool columns.
    drop_first:
        Drop the first level to avoid strict multicollinearity.
    dummy_na:
        If True, adds a column to indicate NaNs.
    """
    cat_cols = list(columns) if columns is not None else list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, drop_first=drop_first, dummy_na=dummy_na)
