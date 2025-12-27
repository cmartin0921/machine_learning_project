import pandas as pd
import numpy as np

def clean_data(sensors_df, weather_df):

    for col in ("datetime_from", "datetime_to"):
        if col in sensors_df.columns:
            sensors_df[col] = (
                pd.to_datetime(sensors_df[col], errors="coerce", utc=True)
                  .dt.tz_convert(None)
                  .dt.normalize()
            )

    return sensors_df, weather_df

