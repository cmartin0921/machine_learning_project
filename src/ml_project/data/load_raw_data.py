from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _default_raw_dir() -> Path:
    """
    Resolve the default raw data directory (../data/raw relative to repo root).
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "raw"


def load_raw_data(raw_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load all raw CSV assets into DataFrames.

    Parameters
    ----------
    raw_dir:
        Base directory containing the raw CSV files. If omitted, uses
        `<project_root>/data/raw`.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys: ``measurements``, ``weather``, ``locations``, and
        ``sensors_metadata``.
    """
    base_dir = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    file_map = {
        "measurements": "sensors_measurement.csv",
        "weather": "weather_daily.csv",
        "locations": "locations_dataset.csv",
        "sensors_metadata": "sensors_metadata.csv",
    }

    datasets: Dict[str, pd.DataFrame] = {}
    for key, filename in file_map.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected data file missing: {path}")
        datasets[key] = pd.read_csv(path)

    return datasets
