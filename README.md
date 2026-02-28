# ML Project Pipeline Documentation

## Quickstart — Run `main.py`

Outputs (logs, intermediate files, reports) are written under the `logs/`, `data/`, and `reports/` directories.

Note: runtime logs are written to `logs/ml_project.log` — inspect this file for detailed execution information and troubleshooting.

This document provides a detailed description of each function used in `main.py`, explaining what data is edited, transformed, and created at each step of the machine learning pipeline.

### Running tests

Run the test suite with `pytest`. Examples from the project root:

```bash
# Run all tests (recommended)
pytest tests/ -q

# To run a single test file or a single test:
pytest tests/test_clean_data.py -q
pytest tests/test_clean_data.py::test_function_name -q
```

---

## Table of Contents

1. [Setup & Configuration](#1-setup--configuration)
2. [Data Loading](#2-data-loading)
3. [Data Cleaning](#3-data-cleaning)
4. [Outlier Detection & Handling](#4-outlier-detection--handling)
5. [Missing Data Imputation](#5-missing-data-imputation)
6. [Feature Generation](#6-feature-generation)
7. [One-Hot Encoding](#7-one-hot-encoding)
8. [Train-Test Split](#8-train-test-split)
9. [Feature Scaling](#9-feature-scaling)
10. [Model Creation & Training](#10-model-creation--training)
11. [Model Evaluation](#11-model-evaluation)

---

## 1. Setup & Configuration

### `get_project_directories()`

**Purpose:** Establishes a dictionary of all project directory paths for consistent file access.

**What it creates:**
- A dictionary containing `Path` objects for:
  - `root` - Project root directory (found by locating `.git`)
  - `src` - Source code directory
  - `configs` - Configuration files directory
  - `notebooks` - Jupyter notebooks directory
  - `data` - Main data directory
  - `data_raw` - Raw data storage (`data/raw/`)
  - `data_interim` - Intermediate processed data (`data/interim/`)
  - `data_processed` - Final processed data (`data/processed/`)
  - `models` - Saved models directory
  - `reports` - Reports and figures directory

**Output:** `Dict[str, Path]`

---

### `setup_logger(log_file_path, name, level)`

**Purpose:** Configures a logging system for tracking pipeline execution.

**What it creates:**
- A `Logger` instance with:
  - File handler writing to `logs/ml_project.log`
  - Timestamp formatting (`YYYY-MM-DD HH:MM:SS`)
  - DEBUG level logging by default
  - UTF-8 encoding support

**Output:** `logging.Logger`

---

## 2. Data Loading

**Purpose:** Load raw CSV files into pandas DataFrames.

**Input Files:**
| Key | File | Description |
|-----|------|-------------|
| `locations` | `_warsaw_locations.csv` | Air quality monitoring station locations |
| `sensors_metadata` | `_warsaw_sensors_metadata.csv` | Sensor specifications and measurement types |
| `sensors_measurements` | `_warsaw_sensors_measurements.csv` | Actual pollutant readings |
| `weather` | `_warsaw_weather_daily.csv` | Daily weather observations |

**Output:** `Dict[str, pd.DataFrame]` - Dictionary of raw DataFrames

---

## 3. Data Cleaning

### `clean_data(df_dicts)`

**Purpose:** Clean, validate, and merge all raw datasets into a single analysis-ready DataFrame.

#### Internal Functions:

##### `_clean_measurements(df)`
**Transformations:**
- Converts `sensor_id` to integer (coerces invalid values)
- Converts `value` to numeric
- Parses `datetime_from` and `datetime_to` to UTC datetime
- Normalizes `timestamp_rollup` and `metric_name` (lowercase, stripped)
- Drops rows with null essential fields (`sensor_id`, `datetime_to`, `metric_name`, `value`)
- Sets negative pollutant values to 0 (except temperature)
- Creates `reading_date` column (date only, no time)
- Renames `units` → `reading_units`
- Removes duplicate readings (same sensor, date, value)
- Filters out overlapping multi-day readings

##### `_clean_weather(df)`
**Transformations:**
- Parses `date` to datetime (timezone-naive)
- Converts weather columns to numeric: `tavg`, `tmin`, `tmax`, `prcp`, `snow`, `wdir`, `wspd`, `wpgt`, `pres`, `tsun`
- Fills `prcp` (precipitation) and `snow` with 0 where missing
- Drops columns with 100% null values

##### `_clean_locations(df)`
**Transformations:**
- Converts IDs to integers: `location_id`, `country_id`
- Converts coordinates to numeric: `latitude`, `longitude`
- Parses `first_read_at` and `last_read_at` to datetime
- Drops rows missing `location_id`

##### `_clean_sensor_metadata(df)`
**Transformations:**
- Converts `sensor_id` and `location_id` to integers
- Normalizes `measurement_name` and `measurement` (lowercase, stripped)
- Renames `units` → `sensor_units`

##### `_data_transform(measurements, weather)`
**Transformations:**
- Pivots measurements: rows = `reading_date`, columns = `metric_name`, values = mean of `value`
- Adds `sensor_count` column (unique sensors per day)
- Drops columns with ≥50% missing values
- Merges pivoted measurements with weather data on date

**Output:** 
```python
{
    "locations": pd.DataFrame,        # Cleaned location data
    "sensors_metadata": pd.DataFrame, # Cleaned sensor metadata
    "sensors_measurements": pd.DataFrame, # Cleaned measurements
    "weather": pd.DataFrame,          # Cleaned weather data
    "cleaned": pd.DataFrame           # Final merged dataset for ML
}
```

---

## 4. Outlier Detection & Handling

### `detect_outliers(df, columns, exclude_columns, iqr_multiplier)`

**Purpose:** Identify outliers using the IQR (Interquartile Range) method for inspection/logging.

**Method:**
1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
2. Compute IQR = Q3 - Q1
3. Define fences: Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR
4. Flag values outside fences as outliers

**Excluded Columns (by default):**
- `reading_date`, `date`, `latitude`, `longitude`, `sensor_count`, `location_id`, `sensor_id`

**Output:** `Dict[str, List[Tuple[int, float, float, float]]]`
- Maps column names to list of tuples: `(row_index, value, lower_fence, upper_fence)`

---

### `handle_outliers(df, method="cap")`

**Purpose:** Handle detected outliers by capping (winsorization) or removal.

**Methods:**
| Method | Action |
|--------|--------|
| `"cap"` | Clips outlier values to the fence boundaries (default) |
| `"remove"` | Removes entire rows containing outliers |

**Transformation:**
- For each numeric column, values below lower fence → set to lower fence
- Values above upper fence → set to upper fence
- NaN values are preserved

**Output:** `pd.DataFrame` with outliers handled

---

## 5. Missing Data Imputation

### `impute_missing_data(df, date_column, window_size)`

**Purpose:** Fill missing numeric values using intelligent time-based imputation.

**Imputation Strategy (cascading fallbacks):**

1. **7-day Rolling Average** (primary)
   - Centered window with `min_periods=1`
   - Captures local temporal patterns

2. **Month-Year Mean** (fallback 1)
   - Groups by specific month-year period
   - Handles cases where entire rolling window is NaN

3. **Same Month Across Years** (fallback 2)
   - Groups by month only
   - Handles sparse early data

**Transformations:**
- Sorts data by `reading_date`
- Creates temporary helper columns (`_month_year`, `_month`)
- Applies cascading imputation to all numeric columns
- Removes helper columns after imputation

**Output:** `pd.DataFrame` with no missing values in numeric columns

---

## 6. Feature Generation

### `generate_features(dataframe)`

**Purpose:** Create new predictive features from existing weather and pollutant data.

**New Features Created:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `temp_wind_interaction` | `tavg × wspd` | Wind disperses pollutants differently at various temperatures |
| `temp_diurnal_range` | `tmax - tmin` | Large temperature swings indicate atmospheric stagnation |
| `pollutant_ratio` | `no2 / (co + 0.001)` | Distinguishes traffic vs. industrial pollution sources |
| `day_of_year_sin` | `sin(2π × day/365.25)` | Cyclical encoding of seasonal patterns |
| `day_of_year_cos` | `cos(2π × day/365.25)` | Cyclical encoding of seasonal patterns |
| `month_sin` | `sin(2π × month/12)` | Cyclical encoding of monthly patterns |
| `month_cos` | `cos(2π × month/12)` | Cyclical encoding of monthly patterns |
| `is_weekend` | `0` or `1` | Weekend indicator (reduced traffic/industrial activity) |
| `season` | `'Winter'`, `'Spring'`, `'Summer'`, `'Autumn'` | Categorical season classification |

**Output:** `pd.DataFrame` with original + new feature columns

---

## 7. One-Hot Encoding

### `one_hot_encoding(df, columns, drop_first)`

**Purpose:** Convert categorical variables into binary indicator columns for ML model compatibility.

**Behavior:**
- Auto-detects `object` and `category` dtype columns if none specified
- Excludes `reading_date` from encoding
- Uses `pd.get_dummies()` internally
- Drops first category by default to avoid multicollinearity

**Example Transformation:**
```
season → season_Spring, season_Summer, season_Winter
(Autumn dropped as reference)
```

**Output:** `pd.DataFrame` with categorical columns replaced by binary indicators

---

## 8. Train-Test Split

### `split_dataset(dataframe, target_col, test_size)`

**Purpose:** Separate features from target and split data into training/testing sets.

**Parameters:**
- `target_col`: Column to predict (default: `"pm25"`)
- `test_size`: Proportion for test set (default: `0.3` = 30%)

**Transformations:**
1. Separates `X` (features) from `y` (target)
2. Randomly splits into train/test sets using `sklearn.train_test_split`

**Output:** `Tuple[X_train, X_test, y_train, y_test]`
- `X_train`: Training features (70% of data)
- `X_test`: Test features (30% of data)
- `y_train`: Training target values
- `y_test`: Test target values

---

## 9. Feature Scaling

### `scaling(dataframe, target_col, existing_scaler)`

**Purpose:** Standardize numeric features to mean=0, std=1 for optimal ML performance.

**Method:** `StandardScaler` from scikit-learn

**Excluded from Scaling:**
- `pm25` (target variable)
- `latitude`, `longitude` (constant geographic values)
- `snow` (mostly zeros)
- `reading_date` (datetime)

**Usage Pattern:**
```python
# Fit on training data
x_train_scaled, scaler = scaling(x_train)

# Transform test data using same scaler (no refitting)
x_test_scaled, _ = scaling(x_test, existing_scaler=scaler)
```

**Output:** `Tuple[pd.DataFrame, StandardScaler]`
- Scaled DataFrame
- Fitted scaler object (for consistent test set transformation)

---

## 10. Model Creation & Training

### `create_model(model_params, x_train, y_train)`

**Purpose:** Build **and fit** a regression model using hyperparameters from `configs/model.yaml`.

**Supported models:**
- `random_forest` → `sklearn.ensemble.RandomForestRegressor`

**Inputs:**
- `model_params` (`dict`): Model configuration values. Keys used:
  - `model_type` (default: `"random_forest"`)
  - `n_estimators` (required for `random_forest`)
  - `max_depth` (optional)
  - `random_state` (optional)
  - `n_jobs` (optional)
- `x_train` (array-like / `pd.DataFrame`): Training features, shape `(n_samples, n_features)`
- `y_train` (array-like / `pd.Series`): Training target, shape `(n_samples,)`

**Behavior:**
1. Reads `model_type` from `model_params` (defaults to `"random_forest"`).
2. Instantiates the corresponding estimator with the provided hyperparameters.
3. Calls `.fit(x_train, y_train)` and returns the trained model.

**Output:** Trained scikit-learn estimator.

**Raises:**
- `ValueError` if `model_type` is not supported.
- Scikit-learn validation errors if required params (e.g., `n_estimators`) are missing/invalid.

**Example:**
```python
model = create_model(model_params, x_train_scaled, y_train)
```

---

## 11. Model Evaluation

### `evaluate_model(model, model_params, x_test, y_test, reports_dir, logger=None)`

**Purpose:** Evaluate a trained regression model on test data, compute metrics, and write diagnostic plots to disk.

**Inputs:**
- `model`: Trained scikit-learn estimator (must implement `.predict()`).
- `model_params` (`dict`): Used to read `model_type` (default: `"random_forest"`).
- `x_test` (array-like / `pd.DataFrame`): Test features.
- `y_test` (array-like / `pd.Series`): Ground-truth test target.
- `reports_dir` (`str` or `Path`): Output directory for plots (created if missing).
- `logger` (`logging.Logger`, optional): If provided, logs model type, metrics, and plot paths.

**Metrics returned (for `random_forest`):**
- `r2` (`sklearn.metrics.r2_score`)
- `mse` (`sklearn.metrics.mean_squared_error`)
- `mae` (`sklearn.metrics.mean_absolute_error`)

**Plots saved to `reports_dir`:**
- `actual_vs_predicted.png` (scatter with 45° reference line)
- `predicted_vs_residuals.png` (residuals vs predicted with 0-line)
- `residual_distribution.png` (histogram, 30 bins)
- `qq_plot_residuals.png` (Q–Q plot of residuals vs normal)

**Output:**
```python
{
    "model_type": "random_forest",
    "metrics": {"r2": float, "mse": float, "mae": float},
    "plots": {
        "actual_vs_predicted": "path/to/actual_vs_predicted.png",
        "predicted_vs_residuals": "path/to/predicted_vs_residuals.png",
        "residual_distribution": "path/to/residual_distribution.png",
        "qq_plot_residuals": "path/to/qq_plot_residuals.png"
    }
}
```

**Raises:**
- `ValueError` if `model_type` is not supported.


---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA                                  │
│  locations.csv, sensors_metadata.csv, measurements.csv, weather │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLEAN DATA                                  │
│  • Type conversions  • Null handling  • Merge datasets          │
│  • Pivot measurements  • Remove invalid readings                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HANDLE OUTLIERS                               │
│  • IQR-based detection  • Cap values at fences                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IMPUTE MISSING DATA                            │
│  • 7-day rolling average  • Month-year fallback                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GENERATE FEATURES                              │
│  • Interaction terms  • Cyclical encodings  • Season extraction │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ONE-HOT ENCODING                               │
│  • Convert categorical → binary indicators                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRAIN-TEST SPLIT                               │
│  • 70% training / 30% testing  • Separate X and y               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SCALING                                     │
│  • StandardScaler on training  • Transform test with same scaler│
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING                                  │
│  • Fit model to X_train, y_train                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL EVALUATION                                │
│  • Predict on X_test  • Calculate metrics vs y_test             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Files

### `configs/data.yaml`
Contains data extraction parameters:
- City configurations
- Date ranges
- Output file names for OpenAQ and Meteostat data

### `configs/model.yaml`
Contains model hyperparameters passed to `create_model()`

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `ml_project.log` | `logs/` | Execution logs with timestamps |

---
