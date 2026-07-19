"""Exploratory data analysis helpers and report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def summary_statistics(dataframe: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """Return a rich statistical summary for the dataset."""

    if numeric_only:
        dataframe = dataframe.select_dtypes(include=[np.number])
    stats_frame = dataframe.describe().T
    stats_frame["median"] = dataframe.median(numeric_only=True)
    stats_frame["variance"] = dataframe.var(numeric_only=True)
    stats_frame["skewness"] = dataframe.skew(numeric_only=True)
    stats_frame["kurtosis"] = dataframe.kurtosis(numeric_only=True)
    return stats_frame


def correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute the pairwise Pearson correlation matrix for numeric fields."""

    return dataframe.select_dtypes(include=[np.number]).corr()


def detect_outliers_iqr(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """Flag observations outside the IQR bounds."""

    series = dataframe[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return dataframe.loc[mask].copy()


def detect_outliers_zscore(dataframe: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Flag observations using a z-score threshold."""

    z_scores = stats.zscore(dataframe[column].dropna())
    mask = np.abs(z_scores) > threshold
    return dataframe.loc[dataframe[column].dropna().index[mask]].copy()


def seasonality_profile(dataframe: pd.DataFrame, date_column: str = "date", value_column: str = "close") -> pd.DataFrame:
    """Create a month-by-year seasonality table."""

    seasonal = dataframe.copy()
    seasonal[date_column] = pd.to_datetime(seasonal[date_column])
    seasonal["year"] = seasonal[date_column].dt.year
    seasonal["month"] = seasonal[date_column].dt.month
    return seasonal.pivot_table(index="year", columns="month", values=value_column, aggfunc="mean")


def trend_by_period(dataframe: pd.DataFrame, date_column: str = "date", value_column: str = "close", rule: str = "M") -> pd.DataFrame:
    """Aggregate prices by a time frequency such as monthly or quarterly."""

    trend = dataframe.copy()
    trend[date_column] = pd.to_datetime(trend[date_column])
    return trend.set_index(date_column)[value_column].resample(rule).mean().to_frame(name=f"{value_column}_mean")


def seasonal_decomposition_report(dataframe: pd.DataFrame, date_column: str = "date", value_column: str = "close", period: int = 30):
    """Run classical seasonal decomposition when there is enough history."""

    series = dataframe.copy().sort_values(date_column)
    series[date_column] = pd.to_datetime(series[date_column])
    ts = series.set_index(date_column)[value_column].dropna()
    if len(ts) < period * 2:
        raise ValueError("Not enough observations for seasonal decomposition.")
    return seasonal_decompose(ts, model="additive", period=period)


def generate_ydata_profiling_report(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a ydata-profiling HTML report when the package is installed."""

    try:
        from ydata_profiling import ProfileReport
    except ImportError as exc:
        raise RuntimeError(
            "ydata-profiling is optional and is not available in this Python 3.14 environment. "
            "Use a compatible Python version to generate the report."
        ) from exc

    report = ProfileReport(dataframe, title="Stock Market Profiling Report", explorative=True)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_file(output_file)
    return output_file


def generate_sweetviz_report(dataframe: pd.DataFrame, output_path: str | Path, target_feat: str | None = None) -> Path:
    """Create a Sweetviz HTML report when the package is installed."""

    try:
        import sweetviz as sv
    except ImportError as exc:
        raise RuntimeError(
            "Sweetviz is optional and is not available in this Python 3.14 environment. "
            "Use a compatible Python version to generate the report."
        ) from exc

    report = sv.analyze(dataframe, target_feat=target_feat)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.show_html(str(output_file), open_browser=False)
    return output_file
