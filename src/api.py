"""Alpha Vantage data collection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .config import ALPHA_VANTAGE_BASE_URL, RAW_DATA_DIR
from .utils import ensure_directories, get_api_key, save_dataframe


@dataclass(slots=True)
class AlphaVantageClient:
    """Small wrapper around the Alpha Vantage CSV endpoint."""

    api_key: str | None = None
    timeout: int = 60

    def __post_init__(self) -> None:
        ensure_directories()
        if self.api_key is None:
            self.api_key = get_api_key()

    def fetch_daily_adjusted(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Download daily adjusted OHLCV data for a ticker symbol."""

        if not self.api_key:
            raise ValueError("Missing Alpha Vantage API key. Set ALPHAVANTAGE_API_KEY first.")

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol.upper(),
            "datatype": "csv",
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = pd.read_csv(StringIO(response.text))
        if data.empty:
            raise ValueError(f"No rows returned for symbol {symbol}.")
        data["symbol"] = symbol.upper()
        return data

    def save_daily_adjusted(self, symbol: str, outputsize: str = "full") -> Path:
        """Fetch and store raw data as a CSV file."""

        dataframe = self.fetch_daily_adjusted(symbol=symbol, outputsize=outputsize)
        output_path = RAW_DATA_DIR / f"{symbol.upper()}_daily_adjusted.csv"
        return save_dataframe(dataframe, output_path, index=False)

    def fetch_many(self, symbols: list[str], outputsize: str = "full") -> dict[str, pd.DataFrame]:
        """Download multiple tickers in one call."""

        return {symbol.upper(): self.fetch_daily_adjusted(symbol=symbol, outputsize=outputsize) for symbol in symbols}


def load_local_raw_data(symbol: str) -> pd.DataFrame | None:
    """Load a previously downloaded raw file if it exists."""

    file_path = RAW_DATA_DIR / f"{symbol.upper()}_daily_adjusted.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None
