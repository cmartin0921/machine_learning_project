from typing import Dict, Tuple

import pandas as pd
from sklearn.impute import KNNImputer


def impute_missing_data(
    df: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    return_imputers: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Impute missing values using KNN for numeric features.

    Parameters
    ----------
    df:
        Input dataframe to impute.
    n_neighbors:
        Number of neighbors for KNN imputation. Defaults to 5.
    return_imputers:
        If True, also return the fitted imputer instances for reuse.

    Returns
    -------
    pandas.DataFrame or (DataFrame, dict)
        Imputed dataframe with missing numeric values filled, and optionally
        the fitted imputers keyed by type.
    """
    result = df.copy()
    imputers: Dict[str, object] = {}

    # Get numeric columns
    numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        num_imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        result[numeric_cols] = num_imputer.fit_transform(result[numeric_cols])

        # Handle any remaining NaN values (edge cases)
        for col in numeric_cols:
            if result[col].isna().any():
                filler = result[col].mean()
                if pd.isna(filler):
                    filler = 0.0
                result[col] = result[col].fillna(filler)

        imputers["numeric"] = num_imputer

    if return_imputers:
        return result, imputers
    return result
