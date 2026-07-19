"""Project-wide configuration and path constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
API_KEY_ENV_VAR = "ALPHAVANTAGE_API_KEY"
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX"]
DEFAULT_PRICE_COLUMN = "adjusted_close"
DEFAULT_TARGET_COLUMN = "close"
DEFAULT_DATE_COLUMN = "date"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

TECHNICAL_WINDOWS = {
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 100,
    "sma_very_long": 200,
    "ema_short": 20,
    "ema_medium": 50,
    "rsi": 14,
    "bollinger": 20,
    "atr": 14,
    "momentum": 10,
    "volatility": 20,
}
