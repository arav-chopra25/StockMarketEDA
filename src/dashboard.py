"""Streamlit dashboard for the stock market trends project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .api import AlphaVantageClient, load_local_raw_data
from .config import DEFAULT_SYMBOLS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .eda import correlation_matrix, detect_outliers_iqr, summary_statistics
from .feature_engineering import add_technical_indicators
from .machine_learning import (
    RegressionResult,
    feature_importance,
    train_decision_tree_regression,
    train_linear_regression,
    train_random_forest_regression,
)
from .preprocessing import add_time_features, clean_stock_data
from .visualization import (
    area_chart,
    bar_chart,
    candlestick_chart,
    correlation_heatmap,
    daily_return_chart,
    histogram,
    line_chart,
    macd_chart,
    moving_average_chart,
    ohlc_chart,
    rsi_chart,
    volume_chart,
    volatility_chart,
)
from .utils import ensure_directories, save_dataframe


MODEL_RUNNERS = {
    "Linear Regression": train_linear_regression,
    "Random Forest Regression": train_random_forest_regression,
    "Decision Tree Regression": train_decision_tree_regression,
}


def _load_or_fetch(symbol: str, outputsize: str = "full") -> pd.DataFrame:
    local_frame = load_local_raw_data(symbol)
    if local_frame is not None:
        return local_frame

    client = AlphaVantageClient()
    raw_frame = client.fetch_daily_adjusted(symbol=symbol, outputsize=outputsize)
    save_dataframe(raw_frame, RAW_DATA_DIR / f"{symbol.upper()}_daily_adjusted.csv")
    return raw_frame


def _prepare_dataframe(symbol: str) -> tuple[pd.DataFrame, dict]:
    raw_frame = _load_or_fetch(symbol)
    cleaning_result = clean_stock_data(raw_frame)
    cleaned = add_time_features(cleaning_result.data)
    engineered = add_technical_indicators(cleaned)
    return engineered, {
        "rows_before": cleaning_result.rows_before,
        "rows_after": cleaning_result.rows_after,
        "duplicates_removed": cleaning_result.duplicate_rows_removed,
        "missing_values_filled": cleaning_result.missing_values_filled,
    }


def _render_sidebar() -> dict:
    st.sidebar.header("Project Controls")
    symbol = st.sidebar.selectbox("Select Company", DEFAULT_SYMBOLS, index=0)
    model_name = st.sidebar.selectbox("Prediction Model", list(MODEL_RUNNERS.keys()), index=1)
    date_range_days = st.sidebar.slider("Preview Last N Trading Days", min_value=30, max_value=2520, value=365, step=30)
    moving_average_choice = st.sidebar.multiselect("Moving Averages", ["sma_20", "sma_50", "sma_100", "sma_200"], default=["sma_20", "sma_50"])
    indicator_choice = st.sidebar.multiselect("Technical Indicators", ["rsi", "macd", "volume", "volatility"], default=["rsi", "macd"])
    prediction_horizon = st.sidebar.slider("Prediction Horizon", min_value=1, max_value=30, value=1)
    return {
        "symbol": symbol,
        "model_name": model_name,
        "date_range_days": date_range_days,
        "moving_average_choice": moving_average_choice,
        "indicator_choice": indicator_choice,
        "prediction_horizon": prediction_horizon,
    }


def _show_metrics(dataframe: pd.DataFrame) -> None:
    latest = dataframe.iloc[-1]
    highest = dataframe["high"].max()
    lowest = dataframe["low"].min()
    volume = dataframe["volume"].iloc[-1] if "volume" in dataframe.columns else None

    metric_columns = st.columns(4)
    metric_columns[0].metric("Latest Close", f"{latest['close']:.2f}")
    metric_columns[1].metric("Highest Price", f"{highest:.2f}")
    metric_columns[2].metric("Lowest Price", f"{lowest:.2f}")
    metric_columns[3].metric("Latest Volume", f"{volume:,.0f}" if volume is not None else "N/A")


def _render_charts(dataframe: pd.DataFrame, controls: dict) -> None:
    price_frame = dataframe.tail(controls["date_range_days"])
    tabs = st.tabs(["Overview", "Indicators", "Distributions", "Correlation", "OHLC"])

    with tabs[0]:
        st.plotly_chart(line_chart(price_frame, "timestamp", "close", title=f"{controls['symbol']} Closing Price"), use_container_width=True)
        st.plotly_chart(area_chart(price_frame, "timestamp", "close", title="Area Chart of Closing Price"), use_container_width=True)
        st.plotly_chart(moving_average_chart(price_frame, "timestamp", "close", controls["moving_average_choice"]), use_container_width=True)

    with tabs[1]:
        if "rsi" in controls["indicator_choice"] and "rsi" in price_frame.columns:
            st.plotly_chart(rsi_chart(price_frame, "timestamp"), use_container_width=True)
        if "macd" in controls["indicator_choice"] and {"macd", "signal_line"}.issubset(price_frame.columns):
            st.plotly_chart(macd_chart(price_frame, "timestamp"), use_container_width=True)
        if "volume" in controls["indicator_choice"] and "volume" in price_frame.columns:
            st.plotly_chart(volume_chart(price_frame, "timestamp"), use_container_width=True)
        if "volatility" in controls["indicator_choice"] and "volatility_20" in price_frame.columns:
            st.plotly_chart(volatility_chart(price_frame, "timestamp"), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(histogram(price_frame, "daily_return", title="Daily Return Distribution"), use_container_width=True)
        st.dataframe(summary_statistics(price_frame), use_container_width=True)

    with tabs[3]:
        corr = correlation_matrix(price_frame)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
        st.dataframe(corr, use_container_width=True)

    with tabs[4]:
        if {"open", "high", "low", "close"}.issubset(price_frame.columns):
            st.plotly_chart(candlestick_chart(price_frame), use_container_width=True)
            st.plotly_chart(ohlc_chart(price_frame), use_container_width=True)


def _run_model(dataframe: pd.DataFrame, model_name: str) -> RegressionResult:
    runner = MODEL_RUNNERS[model_name]
    return runner(dataframe)


def _render_machine_learning(dataframe: pd.DataFrame, model_name: str) -> None:
    st.subheader("Machine Learning Results")
    result = _run_model(dataframe, model_name)
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", f"{result.metrics['mae']:.4f}")
    metric_columns[1].metric("MSE", f"{result.metrics['mse']:.4f}")
    metric_columns[2].metric("RMSE", f"{result.metrics['rmse']:.4f}")
    metric_columns[3].metric("R2", f"{result.metrics['r2']:.4f}")
    comparison = pd.DataFrame({"Actual": result.y_test, "Predicted": result.predictions})
    st.line_chart(comparison)

    if hasattr(result.model, "named_steps"):
        estimator = result.model.named_steps.get("regressor")
    else:
        estimator = result.model
    if estimator is not None:
        importances = feature_importance(estimator, result.feature_columns)
        if not importances.empty:
            st.dataframe(importances, use_container_width=True)


def main() -> None:
    ensure_directories()
    st.set_page_config(page_title="Stock Market Trends Dashboard", page_icon="\U0001F4C8", layout="wide")
    st.title("Exploring Stock Market Trends with Plotly")
    st.caption("Interactive EDA, technical indicators, anomaly detection, and machine learning for stock data.")

    controls = _render_sidebar()
    with st.spinner(f"Loading {controls['symbol']} data and computing indicators..."):
        dataframe, cleaning_summary = _prepare_dataframe(controls["symbol"])

    st.success(
        f"Data ready for {controls['symbol']}: {cleaning_summary['rows_after']} cleaned rows, "
        f"{cleaning_summary['duplicates_removed']} removed duplicates, "
        f"{cleaning_summary['missing_values_filled']} missing values processed."
    )

    _show_metrics(dataframe)
    _render_charts(dataframe, controls)

    st.subheader("Business Insights")
    outliers = detect_outliers_iqr(dataframe, "close")
    st.write(f"Potential closing-price outliers detected: {len(outliers)}")
    st.write("Highest trading-day volume and trend insights can be expanded in the final analysis report.")

    _render_machine_learning(dataframe, controls["model_name"])

    st.subheader("Processed Data")
    st.dataframe(dataframe.tail(25), use_container_width=True)
    csv_data = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button("Download Processed CSV", data=csv_data, file_name=f"{controls['symbol']}_processed.csv", mime="text/csv")
