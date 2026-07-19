# Exploring Stock Market Trends with Plotly

Interactive exploratory data analysis, technical indicators, machine learning, and Streamlit dashboard for Alpha Vantage stock data.

## Overview

This project is designed for BACSE301 - Exploratory Data Analysis and demonstrates a full financial analytics workflow:

1. Data collection from Alpha Vantage
2. Data cleaning and preprocessing
3. Feature engineering and technical indicators
4. Exploratory data analysis
5. Plotly visualizations
6. Machine learning models for prediction and anomaly detection
7. Streamlit dashboard for interactive analysis
8. Automated reporting support for ydata-profiling and Sweetviz

## Folder Structure

```text
StockMarketEDA/
|
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|-- src/
|   |-- api.py
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- visualization.py
|   |-- eda.py
|   |-- machine_learning.py
|   |-- dashboard.py
|   |-- utils.py
|-- models/
|-- reports/
|-- plots/
|-- app.py
|-- requirements.txt
|-- README.md
```

## Phase 1 Setup

### 1. Create a virtual environment

On Windows:

```powershell
C:/Python314/python.exe -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set the API key

Create an Alpha Vantage API key and set it in PowerShell:

```powershell
$env:ALPHAVANTAGE_API_KEY="your_api_key_here"
```

## Run the Dashboard

```powershell
streamlit run app.py
```

## Core Features

- Historical stock price collection from Alpha Vantage
- Automated cleaning and preprocessing
- Daily return, volatility, RSI, MACD, Bollinger Bands, ATR, SMA, EMA, and momentum
- Plotly candlestick, OHLC, line, area, bar, histogram, heatmap, and indicator charts
- Linear Regression, Random Forest, and Decision Tree regression models
- Isolation Forest, K-Means, and PCA utilities
- Streamlit dashboard with company selection, date range filtering, indicators, and downloads
- HTML report support with ydata-profiling and Sweetviz on compatible Python versions

## Notebook

The starter notebook is located at [notebooks/01_stock_market_eda_project.ipynb](notebooks/01_stock_market_eda_project.ipynb).

## Notes

- The Alpha Vantage free API has rate limits.
- Start with one symbol such as AAPL for initial testing.
- The dashboard automatically loads cached raw data from `data/raw/` if available.
- ydata-profiling and Sweetviz are optional on Python 3.14; use a compatible Python version if you want to generate those HTML reports locally.

## Next Development Steps

1. Add reusable notebook templates for each phase
2. Expand dashboard filters and comparison views
3. Add ARIMA, Prophet, or LSTM forecasting modules
4. Add saved model loading and prediction export
5. Generate final report and presentation assets
