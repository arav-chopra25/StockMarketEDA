"""Stock market data collection helpers built on top of yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is missing
    yf = None

from .config import RAW_DATA_DIR

EXPECTED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical daily OHLCV data for a ticker.

    Parameters
    ----------
    ticker:
        Stock symbol to download, such as ``AAPL`` or ``RELIANCE.NS``.
    start_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date:
        Exclusive end date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the columns ``Date``, ``Open``, ``High``, ``Low``,
        ``Close``, ``Adj Close``, and ``Volume``.

    Raises
    ------
    ValueError
        Raised when yfinance returns no data for the requested ticker and date
        range.
    """

    if yf is None:
        raise ImportError(
            "yfinance is required to download new market data. Install it or use a saved raw CSV."
        )

    downloaded = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if downloaded.empty:
        raise ValueError(
            f"No stock data returned for '{ticker}' between {start_date} and {end_date}. "
            "Check the ticker symbol and date range."
        )

    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)

    downloaded = downloaded.reset_index()
    first_column = downloaded.columns[0]
    if first_column != "Date":
        downloaded = downloaded.rename(columns={first_column: "Date"})

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in downloaded.columns]
    if missing_columns:
        raise ValueError(
            f"Missing expected columns from yfinance response for '{ticker}': {missing_columns}"
        )

    downloaded["Date"] = pd.to_datetime(downloaded["Date"], errors="coerce")
    if downloaded["Date"].isna().any():
        raise ValueError(f"Invalid date values returned for ticker '{ticker}'.")

    downloaded = downloaded[EXPECTED_COLUMNS].copy()
    downloaded = downloaded.sort_values("Date").reset_index(drop=True)
    return downloaded


def save_raw_data(df: pd.DataFrame, ticker: str, output_dir: str = "data/raw") -> Path:
    """Save raw market data to a CSV file.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`fetch_stock_data`.
    ticker:
        Ticker symbol used to generate the output file name.
    output_dir:
        Directory where the CSV file should be written.

    Returns
    -------
    pathlib.Path
        Path to the saved CSV file.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{ticker}_raw.csv"
    df.to_csv(file_path, index=False)
    return file_path


def load_local_raw_data(ticker: str) -> pd.DataFrame | None:
    """Load a previously saved raw CSV for a ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol to load, such as ``AAPL`` or ``RELIANCE.NS``.

    Returns
    -------
    pandas.DataFrame | None
        Loaded DataFrame when the file exists, otherwise ``None``.
    """

    candidate_paths = [
        RAW_DATA_DIR / f"{ticker}_raw.csv",
        RAW_DATA_DIR / f"{ticker.upper()}_raw.csv",
        RAW_DATA_DIR / f"{ticker.upper()}_daily_adjusted.csv",
    ]

    for file_path in candidate_paths:
        if file_path.exists():
            raw_frame = pd.read_csv(file_path)
            if "Date" in raw_frame.columns:
                raw_frame["Date"] = pd.to_datetime(raw_frame["Date"], errors="coerce")
            elif "timestamp" in raw_frame.columns:
                raw_frame = raw_frame.rename(columns={"timestamp": "Date"})
                raw_frame["Date"] = pd.to_datetime(raw_frame["Date"], errors="coerce")
            return raw_frame
    return None


@dataclass(slots=True)
class StockDataClient:
    """Compatibility wrapper that exposes the legacy client-style API."""

    default_years: int = 5

    def _default_date_range(self) -> tuple[str, str]:
        """Return a default five-year date range for standalone usage."""

        end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start_date = end_date - pd.DateOffset(years=self.default_years)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def fetch_daily_adjusted(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Backward-compatible wrapper around :func:`fetch_stock_data`.

        Parameters
        ----------
        symbol:
            Ticker symbol to fetch.
        outputsize:
            Retained for compatibility with the old interface and ignored.

        Returns
        -------
        pandas.DataFrame
            Historical OHLCV data for the requested symbol.
        """

        start_date, end_date = self._default_date_range()
        return fetch_stock_data(symbol, start_date=start_date, end_date=end_date)

    def save_daily_adjusted(self, symbol: str, outputsize: str = "full") -> Path:
        """Backward-compatible wrapper that saves the fetched raw data.

        Parameters
        ----------
        symbol:
            Ticker symbol to fetch and save.
        outputsize:
            Retained for compatibility with the old interface and ignored.

        Returns
        -------
        pathlib.Path
            Path to the saved CSV file.
        """

        dataframe = self.fetch_daily_adjusted(symbol=symbol, outputsize=outputsize)
        return save_raw_data(dataframe, symbol, output_dir=str(RAW_DATA_DIR))

    def fetch_many(self, symbols: list[str], outputsize: str = "full") -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols using the compatibility wrapper.

        Parameters
        ----------
        symbols:
            List of ticker symbols to download.
        outputsize:
            Retained for compatibility with the old interface and ignored.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Mapping of ticker symbol to downloaded DataFrame.
        """

        return {
            symbol: self.fetch_daily_adjusted(symbol=symbol, outputsize=outputsize)
            for symbol in symbols
        }


AlphaVantageClient = StockDataClient


if __name__ == "__main__":
    example_ticker = "AAPL"
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.DateOffset(years=5)

    example_data = fetch_stock_data(
        ticker=example_ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    saved_path = save_raw_data(example_data, example_ticker)
    print(f"Fetched {len(example_data)} rows for {example_ticker} and saved to {saved_path}")
