"""Utility helpers for the stock market analytics project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


def ensure_directories() -> None:
    """Create the standard project folders when they are missing."""

    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_api_key(env_var: str = "ALPHAVANTAGE_API_KEY") -> str | None:
    """Return the Alpha Vantage API key from the environment."""

    value = os.getenv(env_var)
    return value.strip() if value else None


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case for reliable downstream processing."""

    renamed = df.copy()
    renamed.columns = (
        renamed.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return renamed


def coerce_numeric_columns(df: pd.DataFrame, exclude: Iterable[str] | None = None) -> pd.DataFrame:
    """Convert all non-excluded columns to numeric where possible."""

    excluded = set(exclude or [])
    coerced = df.copy()
    for column in coerced.columns:
        if column not in excluded:
            coerced[column] = pd.to_numeric(coerced[column], errors="ignore")
    return coerced


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Persist a dataframe to CSV and return the resulting path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV file if it exists, otherwise raise a clear error."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the standard regression evaluation metrics."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def safe_percent_change(series: pd.Series) -> pd.Series:
    """Compute percentage change while preserving the original series index."""

    return series.pct_change().replace([float("inf"), float("-inf")], pd.NA)
