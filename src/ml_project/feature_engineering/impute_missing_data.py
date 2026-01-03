from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def _infer_column_types(
    df: pd.DataFrame,
    numeric_columns: Sequence[str] | None,
    categorical_columns: Sequence[str] | None,
) -> Tuple[List[str], List[str]]:
    inferred_numeric = list(df.select_dtypes(include=["number"]).columns) if numeric_columns is None else list(numeric_columns)
    inferred_categorical = (
        list(df.select_dtypes(exclude=["number", "datetime64[ns]", "datetime64[ns, UTC]"]).columns)
        if categorical_columns is None
        else list(categorical_columns)
    )
    return inferred_numeric, inferred_categorical


def impute_missing_data(
    df: pd.DataFrame,
    *,
    strategy: str = "mean",
    n_neighbors: int = 5,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    return_imputers: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Impute missing values using mean or KNN for numeric features and most frequent
    value for categorical features.

    Parameters
    ----------
    df:
        Input dataframe to impute.
    strategy:
        ``\"mean\"`` for :class:`sklearn.impute.SimpleImputer` or ``\"knn\"`` for
        :class:`sklearn.impute.KNNImputer`.
    n_neighbors:
        Number of neighbors when using KNN imputation.
    numeric_columns:
        Optional list of numeric columns to impute. Defaults to all numeric cols.
    categorical_columns:
        Optional list of categorical columns to impute. Defaults to non-numeric,
        non-datetime columns.
    return_imputers:
        If True, also return the fitted imputer instances for reuse.

    Returns
    -------
    pandas.DataFrame or (DataFrame, dict)
        Imputed dataframe, and optionally the fitted imputers keyed by type.
    """
    num_cols, cat_cols = _infer_column_types(df, numeric_columns, categorical_columns)
    result = df.copy()
    imputers: Dict[str, object] = {}

    if num_cols:
        if strategy.lower() == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif strategy.lower() == "knn":
            num_imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        else:
            raise ValueError("strategy must be either 'mean' or 'knn'")
        result[num_cols] = num_imputer.fit_transform(result[num_cols])
        for col in num_cols:
            if result[col].isna().any():
                filler = result[col].mean()
                if pd.isna(filler):
                    filler = 0.0
                result[col] = result[col].fillna(filler)
        imputers["numeric"] = num_imputer

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = cat_imputer.fit_transform(result[cat_cols])
        imputers["categorical"] = cat_imputer

    if return_imputers:
        return result, imputers
    return result
