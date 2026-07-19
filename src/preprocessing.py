"""Cleaning and normalization helpers for stock market time series data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import coerce_numeric_columns, standardize_column_names


EXPECTED_PRICE_COLUMNS = ["open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient"]


@dataclass
class CleaningResult:
    """Return payload for preprocessing operations."""

    data: pd.DataFrame
    rows_before: int
    rows_after: int
    duplicate_rows_removed: int
    missing_values_filled: int


def _clip_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)


def clean_stock_data(dataframe: pd.DataFrame, date_column: str = "date", clip_outliers: bool = False) -> CleaningResult:
    """Normalize Alpha Vantage stock data for analysis and modeling."""

    cleaned = standardize_column_names(dataframe)
    rows_before = len(cleaned)

    if date_column not in cleaned.columns:
        raise KeyError(f"Expected date column '{date_column}' not found.")

    cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[date_column])
    cleaned = cleaned.drop_duplicates(subset=[date_column], keep="last")

    cleaned = coerce_numeric_columns(cleaned, exclude=[date_column, "symbol"])
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns

    missing_values_filled = 0
    for column in numeric_columns:
        missing_values_filled += int(cleaned[column].isna().sum())
        cleaned[column] = cleaned[column].interpolate(method="linear", limit_direction="both")
        cleaned[column] = cleaned[column].bfill().ffill()

    if clip_outliers:
        for column in [col for col in EXPECTED_PRICE_COLUMNS if col in cleaned.columns]:
            cleaned[column] = _clip_iqr(cleaned[column])

    cleaned = cleaned.sort_values(date_column).reset_index(drop=True)
    cleaned = cleaned.dropna(subset=[col for col in ["open", "high", "low", "close"] if col in cleaned.columns])
    rows_after = len(cleaned)
    duplicate_rows_removed = rows_before - rows_after

    return CleaningResult(
        data=cleaned,
        rows_before=rows_before,
        rows_after=rows_after,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values_filled=missing_values_filled,
    )


def add_time_features(dataframe: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add calendar features that help EDA and seasonal analysis."""

    enriched = dataframe.copy()
    enriched[date_column] = pd.to_datetime(enriched[date_column])
    enriched["year"] = enriched[date_column].dt.year
    enriched["month"] = enriched[date_column].dt.month
    enriched["quarter"] = enriched[date_column].dt.quarter
    enriched["day_of_week"] = enriched[date_column].dt.dayofweek
    enriched["day_name"] = enriched[date_column].dt.day_name()
    enriched["month_name"] = enriched[date_column].dt.month_name()
    return enriched
