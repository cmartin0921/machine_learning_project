# src/validation/data_validation.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    # ile kolejnych identycznych wartości uznajemy za "stuck"
    stuck_run_length: int = 48  # np. 48 godzin jeśli hourly
    # próg outlierów (z-score)
    zscore_threshold: float = 6.0
    # czy logic checks są strict (<) czy allow equal (<=)
    allow_equal_weather: bool = True


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _count_negative(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        neg = (df[c] < 0).sum(skipna=True)
        if neg > 0:
            rows.append({"column": c, "negative_count": int(neg)})

    out = pd.DataFrame(rows, columns=["column", "negative_count"])
    return out.sort_values("negative_count", ascending=False)



def _stuck_runs(series: pd.Series, run_length: int) -> int:
    """
    Zwraca maksymalną długość serii identycznych wartości (ignoring NaN).
    """
    s = series.dropna()
    if s.empty:
        return 0
    # porównanie z poprzednią wartością
    same_as_prev = s.eq(s.shift(1))
    # długości runów: gdy false -> nowa grupa
    grp = (~same_as_prev).cumsum()
    run_sizes = s.groupby(grp).size()
    return int(run_sizes.max())


def stuck_sensor_report(df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    cols = [c for c in _numeric_cols(df) if not c.lower().endswith("id") and c.lower() not in {"id", "sensor_id"}]
    rows = []
    for c in cols:
        max_run = _stuck_runs(df[c], cfg.stuck_run_length)
        if max_run >= cfg.stuck_run_length:
            rows.append({"column": c, "max_constant_run": max_run})

    out = pd.DataFrame(rows, columns=["column", "max_constant_run"])
    return out.sort_values("max_constant_run", ascending=False)





def outlier_report_zscore(df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    cols = [c for c in _numeric_cols(df) if not c.lower().endswith("id") and c.lower() not in {"id", "sensor_id"}]
    rows = []
    for c in cols:
        s = df[c]
        if s.dropna().size < 10:
            continue
        mu = s.mean(skipna=True)
        sigma = s.std(skipna=True)
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (s - mu) / sigma
        count = (np.abs(z) > cfg.zscore_threshold).sum(skipna=True)
        if count > 0:
            rows.append(
                {"column": c, "outlier_count_z": int(count), "mean": float(mu), "std": float(sigma)}
            )

    out = pd.DataFrame(rows, columns=["column", "outlier_count_z", "mean", "std"])
    return out.sort_values("outlier_count_z", ascending=False)



def weather_logic_checks(weather_df: pd.DataFrame, cfg: ValidationConfig) -> Dict[str, pd.DataFrame]:
    """
    Szuka typowych kolumn pogodowych (tmin,tavg,tmax,wspd,wpgt) i raportuje wiersze łamiące logikę.
    Jeśli Twoje nazwy są inne, łatwo podmienić mapowanie.
    """
    cols = {c.lower(): c for c in weather_df.columns}

    def get_col(name: str) -> Optional[str]:
        return cols.get(name)

    tmin = get_col("tmin")
    tavg = get_col("tavg")
    tmax = get_col("tmax")
    wspd = get_col("wspd")
    wpgt = get_col("wpgt")

    reports: Dict[str, pd.DataFrame] = {}

    if tmin and tavg and tmax:
        mask = weather_df[[tmin, tavg, tmax]].notna().all(axis=1)
        if cfg.allow_equal_weather:
            ok = (weather_df[tmin] <= weather_df[tavg]) & (weather_df[tavg] <= weather_df[tmax])
        else:
            ok = (weather_df[tmin] < weather_df[tavg]) & (weather_df[tavg] < weather_df[tmax])
        bad = weather_df[mask & ~ok]
        reports["temp_order_violations"] = bad[[tmin, tavg, tmax]].copy()


    if wspd and wpgt:
        mask = weather_df[[wspd, wpgt]].notna().all(axis=1)
        if cfg.allow_equal_weather:
            ok = weather_df[wspd] <= weather_df[wpgt]
        else:
            ok = weather_df[wspd] < weather_df[wpgt]
        bad = weather_df[mask & ~ok]
        reports["wind_speed_vs_gust_violations"] = bad[[wspd, wpgt]].copy()


    return reports


def validate_dataset(
    sensor_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    cfg: Optional[ValidationConfig] = None
) -> Dict[str, object]:
    cfg = cfg or ValidationConfig()

    numeric_sensor = _numeric_cols(sensor_df)

    summary = {
        "sensor_shape": sensor_df.shape,
        "sensor_numeric_cols": len(numeric_sensor),
        "sensor_missing_total": int(sensor_df.isna().sum().sum()),
        "sensor_duplicates": int(sensor_df.duplicated().sum()),
    }

    neg = _count_negative(sensor_df, numeric_sensor)
    stuck = stuck_sensor_report(sensor_df, cfg)
    outliers = outlier_report_zscore(sensor_df, cfg)

    result: Dict[str, object] = {
        "summary": summary,
        "negative_values": neg,
        "stuck_sensors": stuck,
        "outliers_zscore": outliers,
    }

    if weather_df is not None:
        weather_numeric = _numeric_cols(weather_df)
        result["weather_summary"] = {
            "weather_shape": weather_df.shape,
            "weather_numeric_cols": len(weather_numeric),
            "weather_missing_total": int(weather_df.isna().sum().sum()),
            "weather_duplicates": int(weather_df.duplicated().sum()),
        }
        result["weather_logic"] = weather_logic_checks(weather_df, cfg)

    return result

