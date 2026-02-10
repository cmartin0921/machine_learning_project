import pandas as pd

def generate_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    df["testing"]=1
    return df
