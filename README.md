# Market Data Fetcher for Azure

This repository contains GitHub Actions that fetch stock market data at different intervals and store it in Azure Blob Storage.

## Features

- **Minute-level data**: Fetches 1-minute and 5-minute OHLCV data during market hours
- **Hourly and Daily data**: Fetches 1-hour and 1-day OHLCV data every hour
- **Technical Indicators**: Automatically calculates indicators after data fetch
  - Intraday indicators for 1min/5min data (EMA, SMA, RSI, VWAP, Bollinger Bands, etc.)
  - Daily indicators for hourly/daily data (20/50/100/200 SMA/EMA, MACD, ATR, etc.)
- **Advanced Pattern Detection**: Identifies trading patterns with email alerts
  - Trend continuation patterns (Golden Cross, EMA Bounce)
  - Breakout patterns (NR7, 52-week high, Gap-and-Go)
  - Mean reversion patterns (Bollinger Band pierce, Over-extension)
  - Reversal patterns (Hammer at key MAs, Engulfing patterns)
  - Volume confirmation for all patterns
- Runs automatically on schedule or can be triggered manually
- Stores data in Azure Blob Storage in Parquet format
- Handles market hours and trading days automatically
- Saves both current and historic versions of data

## Setup

1. Fork this repository

2. Add the following GitHub Secrets in your repository settings:
   - `TICKERS`: Comma-separated list of stock tickers (e.g., `AAPL,MSFT,GOOGL`)
   - `STORAGE_ACCOUNT_NAME`: Your Azure storage account name
   - `CONTAINER_NAME`: Azure container name (default: `finance`)
   - `ACCESS_KEY`: Azure storage account access key
   - `GMAIL_USER`: Gmail address for sending alerts
   - `GMAIL_APP_PWD`: Gmail app-specific password
   - `ALERT_TO`: Email address to receive alerts

3. The workflow will run automatically during market hours or can be triggered manually

## Workflows

### 1. Fetch Market Data (fetch_market_data.yml)
- **Schedule**: Every minute during market hours (9:30 AM - 4:00 PM EST)
- **Data**: 1-minute and 5-minute bars
- **Script**: `raw_data.R`

### 2. Fetch Hourly and Daily Data (fetch_hourly_daily_data.yml)
- **Schedule**: Every hour (24/7)
- **Data**: 1-hour bars (last 60 days) and daily bars (last 2 years)
- **Script**: `fetch_hourly_daily_data.py`

## Data Structure

Data is saved in the Azure container with the following structure:
```
raw_data/                          # Raw OHLCV data
├── raw_data_1min.parquet          # Current day's 1-minute data
├── raw_data_5min.parquet          # Current day's 5-minute data
├── raw_data_1h.parquet            # Current hourly data (60 days)
├── raw_data_1d.parquet            # Current daily data (2 years)
├── historic_raw_data_1min.parquet # Cumulative historic 1-minute data
├── historic_raw_data_5min.parquet # Cumulative historic 5-minute data
├── historic_raw_data_1h.parquet   # Cumulative historic hourly data
└── historic_raw_data_1d.parquet   # Cumulative historic daily data

indicators_azure/                  # Data with technical indicators
├── data_feed_1min.parquet        # Current 1-minute data with indicators
├── data_feed_5min.parquet        # Current 5-minute data with indicators
├── data_feed_1h.parquet          # Current hourly data with indicators
├── data_feed_1d.parquet          # Current daily data with indicators
├── historic_data_feed_1min.parquet # Historic 1-minute with indicators
├── historic_data_feed_5min.parquet # Historic 5-minute with indicators
├── historic_data_feed_1h.parquet   # Historic hourly with indicators
└── historic_data_feed_1d.parquet   # Historic daily with indicators

patterns/                          # Detected trading patterns
├── advanced_patterns_latest.parquet      # Latest pattern detections
└── advanced_patterns_YYYYMMDD_HHMMSS.parquet # Timestamped pattern history
```

Each Parquet file contains:
- `ticker`: Stock symbol
- `datetime`: Timestamp in EST/EDT
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

## Manual Trigger

You can manually trigger the workflow from the Actions tab in your GitHub repository.