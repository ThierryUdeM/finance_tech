# Market Data Fetcher for Azure

This GitHub Action fetches stock market data every minute during market hours and stores it in Azure Blob Storage.

## Features

- Fetches 1-minute and 5-minute OHLCV data for multiple tickers
- Runs automatically every minute during market hours (9:30 AM - 4:00 PM EST)
- Stores data in Azure Blob Storage in Parquet format
- Handles market hours and trading days automatically
- Saves both timestamped and "latest" versions of data

## Setup

1. Fork this repository

2. Add the following GitHub Secrets in your repository settings:
   - `TICKERS`: Comma-separated list of stock tickers (e.g., `AAPL,MSFT,GOOGL`)
   - `STORAGE_ACCOUNT_NAME`: Your Azure storage account name
   - `CONTAINER_NAME`: Azure container name (default: `finance`)
   - `ACCESS_KEY`: Azure storage account access key

3. The workflow will run automatically during market hours or can be triggered manually

## Data Structure

Data is saved in the Azure container with the following structure:
```
raw_data/
├── raw_data_1min.parquet          # Current day's 1-minute data (overwrites daily)
├── raw_data_5min.parquet          # Current day's 5-minute data (overwrites daily)
├── historic_raw_data_1min.parquet # Cumulative historic 1-minute data
└── historic_raw_data_5min.parquet # Cumulative historic 5-minute data
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