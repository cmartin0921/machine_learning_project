from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .impute_missing_data import impute_missing_data
from .one_hot_encoding import one_hot_encoding
from .remove_outliers import remove_outliers
from .scaling import scaling


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    date_col = None
    for candidate in ("reading_date", "datetime_from", "datetime"):
        if candidate in df_feat.columns and np.issubdtype(df_feat[candidate].dtype, np.datetime64):
            date_col = candidate
            break
    if date_col is None:
        return df_feat

    df_feat["dayofyear"] = df_feat[date_col].dt.dayofyear
    df_feat["dayofweek"] = df_feat[date_col].dt.dayofweek
    df_feat["month"] = df_feat[date_col].dt.month

    # Cyclical encoding for seasonality.
    df_feat["season_sin"] = np.sin(2 * np.pi * df_feat["dayofyear"] / 365.25)
    df_feat["season_cos"] = np.cos(2 * np.pi * df_feat["dayofyear"] / 365.25)
    return df_feat


def _add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    if {"tmax", "tmin"}.issubset(df_feat.columns):
        df_feat["temp_range"] = df_feat["tmax"] - df_feat["tmin"]
    if {"tavg", "prcp"}.issubset(df_feat.columns):
        df_feat["precip_intensity"] = df_feat["prcp"] / (df_feat["tavg"].abs() + 0.1)
    if {"wspd", "wpgt"}.issubset(df_feat.columns):
        df_feat["wind_gust_ratio"] = df_feat["wspd"] / (df_feat["wpgt"] + 0.1)
    if {"pres", "tavg"}.issubset(df_feat.columns):
        df_feat["pressure_temp_ratio"] = df_feat["pres"] / (df_feat["tavg"].abs() + 0.1)
    return df_feat


def _add_location_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    if {"latitude", "longitude"}.issubset(df_feat.columns):
        df_feat["lat_long_product"] = df_feat["latitude"] * df_feat["longitude"]
        df_feat["lat_long_magnitude"] = np.sqrt(df_feat["latitude"] ** 2 + df_feat["longitude"] ** 2)
    return df_feat


def _add_value_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    if {"value", "tavg"}.issubset(df_feat.columns):
        df_feat["value_per_temp"] = df_feat["value"] / (df_feat["tavg"].abs() + 0.1)
    if {"value", "wspd"}.issubset(df_feat.columns):
        df_feat["value_per_wind"] = df_feat["value"] / (df_feat["wspd"].abs() + 0.1)
    return df_feat


def _add_interaction_and_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = _add_temporal_features(df)
    df_feat = _add_weather_interactions(df_feat)
    df_feat = _add_location_interactions(df_feat)
    df_feat = _add_value_based_features(df_feat)
    return df_feat


def _id_like_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if "id" in col.lower()]


def _infer_categorical_columns(df: pd.DataFrame, provided: Sequence[str] | None) -> List[str]:
    if provided is not None:
        return list(provided)
    base = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    base.extend(_id_like_columns(df))
    # Preserve order while deduplicating.
    seen = set()
    inferred: List[str] = []
    for col in base:
        if col not in seen:
            seen.add(col)
            inferred.append(col)
    return inferred


def _infer_numeric_columns(df: pd.DataFrame, provided: Sequence[str] | None, excluded: Sequence[str]) -> List[str]:
    if provided is not None:
        return list(provided)
    excluded_set = set(excluded)
    return [col for col in df.select_dtypes(include=["number"]).columns if col not in excluded_set]


def _infer_outlier_columns(df: pd.DataFrame, provided: Sequence[str] | None, excluded: Sequence[str]) -> List[str]:
    if provided is not None:
        return list(provided)
    excluded_set = set(excluded)
    return [col for col in df.select_dtypes(include=["number"]).columns if col not in excluded_set]


def generate_features(
    df: pd.DataFrame,
    *,
    imputation_strategy: str = "mean",
    knn_neighbors: int = 5,
    outlier_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
    numeric_columns: Sequence[str] | None = None,
    scaling_method: str = "standard",
    apply_power_transform: bool = False,
    drop_first_dummies: bool = True,
    save_path: str | Path | None = None,
    return_artifacts: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Full feature-engineering routine:
    1. Impute missing values (mean or KNN).
    2. Remove outliers via IQR.
    3. Add interaction/composite features.
    4. Scale numeric columns.
    5. One-hot encode categorical features.

    Parameters
    ----------
    df:
        Input dataframe (ideally cleaned).
    imputation_strategy:
        ``\"mean\"`` or ``\"knn\"`` for numeric imputation.
    knn_neighbors:
        Number of neighbors for KNN imputation.
    outlier_columns:
        Numeric columns to check for outliers. Defaults to all numeric columns.
    categorical_columns:
        Columns to one-hot encode. Defaults to object/category/bool columns.
    numeric_columns:
        Numeric columns to scale. Defaults to all numeric columns after feature
        creation.
    scaling_method:
        Scaling strategy: ``standard``, ``minmax``, or ``robust``.
    apply_power_transform:
        If True, applies a Yeo-Johnson transform before scaling to tame skew.
    drop_first_dummies:
        Drop the first level when creating one-hot encoded columns.
    save_path:
        Optional CSV path to persist the engineered dataset.
    return_artifacts:
        If True, return fitted imputers/scalers in addition to the dataframe.
    """
    artifacts: Dict[str, object] = {}

    id_like_cols = _id_like_columns(df)
    categorical_cols = _infer_categorical_columns(df, categorical_columns)
    numeric_cols_for_imputation = _infer_numeric_columns(df, numeric_columns, excluded=id_like_cols)
    outlier_cols = _infer_outlier_columns(df, outlier_columns, excluded=id_like_cols)

    imputed, imputers = impute_missing_data(
        df,
        strategy=imputation_strategy,
        n_neighbors=knn_neighbors,
        numeric_columns=numeric_cols_for_imputation,
        categorical_columns=categorical_cols,
        return_imputers=True,
    )
    artifacts["imputers"] = imputers

    trimmed = remove_outliers(imputed, columns=outlier_cols)
    with_composites = _add_interaction_and_composite_features(trimmed)

    numeric_cols_for_scaling = (
        numeric_cols_for_imputation
        if numeric_columns is not None
        else _infer_numeric_columns(with_composites, None, excluded=id_like_cols)
    )
    scaled, transformers = scaling(
        with_composites,
        numeric_columns=numeric_cols_for_scaling,
        method=scaling_method,
        apply_power_transform=apply_power_transform,
        return_transformers=True,
    )
    artifacts["transformers"] = transformers

    encoded = one_hot_encoding(scaled, columns=categorical_cols, drop_first=drop_first_dummies)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(save_path, index=False)

    if return_artifacts:
        return encoded, artifacts
    return encoded
