"""Technical indicator and feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TECHNICAL_WINDOWS


def _base_price_series(dataframe: pd.DataFrame, price_column: str = "adjusted_close") -> pd.Series:
    if price_column in dataframe.columns:
        return dataframe[price_column]
    if "close" in dataframe.columns:
        return dataframe["close"]
    raise KeyError("A price column such as 'close' or 'adjusted_close' is required.")


def add_return_features(dataframe: pd.DataFrame, price_column: str = "adjusted_close") -> pd.DataFrame:
    enriched = dataframe.copy()
    price = _base_price_series(enriched, price_column)

    enriched["daily_return"] = price.pct_change()
    enriched["percentage_change"] = enriched["daily_return"] * 100
    enriched["log_returns"] = np.log(price / price.shift(1))
    enriched["cumulative_returns"] = (1 + enriched["daily_return"].fillna(0)).cumprod() - 1
    enriched["momentum"] = price - price.shift(TECHNICAL_WINDOWS["momentum"])
    if {"open", "close"}.issubset(enriched.columns):
        enriched["price_difference"] = enriched["close"] - enriched["open"]
    if {"high", "low"}.issubset(enriched.columns):
        enriched["price_range"] = enriched["high"] - enriched["low"]
    return enriched


def add_moving_averages(dataframe: pd.DataFrame, price_column: str = "adjusted_close") -> pd.DataFrame:
    enriched = dataframe.copy()
    price = _base_price_series(enriched, price_column)

    for label, window in {
        "sma_20": TECHNICAL_WINDOWS["sma_short"],
        "sma_50": TECHNICAL_WINDOWS["sma_medium"],
        "sma_100": TECHNICAL_WINDOWS["sma_long"],
        "sma_200": TECHNICAL_WINDOWS["sma_very_long"],
    }.items():
        enriched[label] = price.rolling(window=window).mean()

    enriched["ema_20"] = price.ewm(span=TECHNICAL_WINDOWS["ema_short"], adjust=False).mean()
    enriched["ema_50"] = price.ewm(span=TECHNICAL_WINDOWS["ema_medium"], adjust=False).mean()
    enriched["rolling_mean_20"] = price.rolling(window=TECHNICAL_WINDOWS["volatility"]).mean()
    enriched["rolling_std_20"] = price.rolling(window=TECHNICAL_WINDOWS["volatility"]).std()
    enriched["volatility_20"] = enriched["daily_return"].rolling(window=TECHNICAL_WINDOWS["volatility"]).std()
    return enriched


def add_technical_indicators(dataframe: pd.DataFrame, price_column: str = "adjusted_close") -> pd.DataFrame:
    """Add a standard technical-analysis feature set."""

    enriched = add_return_features(dataframe, price_column=price_column)
    enriched = add_moving_averages(enriched, price_column=price_column)
    price = _base_price_series(enriched, price_column)

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=TECHNICAL_WINDOWS["rsi"]).mean()
    avg_loss = loss.rolling(window=TECHNICAL_WINDOWS["rsi"]).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    enriched["rsi"] = 100 - (100 / (1 + rs))

    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    enriched["macd"] = ema_12 - ema_26
    enriched["signal_line"] = enriched["macd"].ewm(span=9, adjust=False).mean()
    enriched["macd_histogram"] = enriched["macd"] - enriched["signal_line"]

    rolling_mean = price.rolling(window=TECHNICAL_WINDOWS["bollinger"]).mean()
    rolling_std = price.rolling(window=TECHNICAL_WINDOWS["bollinger"]).std()
    enriched["bollinger_middle"] = rolling_mean
    enriched["bollinger_upper"] = rolling_mean + (rolling_std * 2)
    enriched["bollinger_lower"] = rolling_mean - (rolling_std * 2)

    if {"high", "low", "close"}.issubset(enriched.columns):
        previous_close = enriched["close"].shift(1)
        tr_components = pd.concat(
            [
                enriched["high"] - enriched["low"],
                (enriched["high"] - previous_close).abs(),
                (enriched["low"] - previous_close).abs(),
            ],
            axis=1,
        )
        enriched["atr"] = tr_components.max(axis=1).rolling(window=TECHNICAL_WINDOWS["atr"]).mean()

    return enriched


def build_supervised_frame(dataframe: pd.DataFrame, target_column: str = "close", horizon: int = 1) -> pd.DataFrame:
    """Create a supervised-learning table with next-day target values."""

    if target_column not in dataframe.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    supervised = dataframe.copy()
    supervised[f"target_{target_column}"] = supervised[target_column].shift(-horizon)
    supervised = supervised.dropna().reset_index(drop=True)
    return supervised
