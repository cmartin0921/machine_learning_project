from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler


def _infer_numeric_columns(df: pd.DataFrame, numeric_columns: Sequence[str] | None) -> List[str]:
    if numeric_columns is None:
        return list(df.select_dtypes(include=["number"]).columns)
    return list(numeric_columns)


def scaling(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] | None = None,
    method: str = "standard",
    apply_power_transform: bool = False,
    return_transformers: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Scale numeric features with configurable strategies.

    Parameters
    ----------
    df:
        Input dataframe.
    numeric_columns:
        Numeric columns to scale. Defaults to all numeric columns.
    method:
        ``\"standard\"`` (z-score), ``\"minmax\"``, or ``\"robust\"`` (IQR based).
    apply_power_transform:
        If True, applies a Yeo-Johnson power transform before scaling to reduce
        skew.
    return_transformers:
        If True, return the fitted transformers along with the dataframe.

    Returns
    -------
    pandas.DataFrame or (DataFrame, dict)
        Scaled dataframe and optionally a dict of fitted transformers.
    """
    num_cols = _infer_numeric_columns(df, numeric_columns)
    scaled = df.copy()
    transformers: Dict[str, object] = {}

    if not num_cols:
        return (scaled, transformers) if return_transformers else scaled

    if apply_power_transform:
        power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        scaled[num_cols] = power_transformer.fit_transform(scaled[num_cols])
        transformers["power_transformer"] = power_transformer

    method = method.lower()
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("method must be one of ['standard', 'minmax', 'robust']")

    scaled[num_cols] = scaler.fit_transform(scaled[num_cols])
    transformers["scaler"] = scaler

    if return_transformers:
        return scaled, transformers
    return scaled
