import pandas as pd

# IMPORTANT: To prevent data leakage, scaling must be done as follows:
#   1. FIT the scaler on the TRAINING data only (learn mean/std or min/max)
#   2. TRANSFORM both training and test data using those same parameters

def scaling(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    numeric_cols: list = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    return x_train, x_test