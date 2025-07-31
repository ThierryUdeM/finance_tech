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
   - `AZURE_STORAGE_ACCOUNT`: Your Azure storage account name
   - `AZURE_CONTAINER_NAME`: Azure container name (default: `finance`)
   - `AZURE_SAS_TOKEN`: SAS token with write permissions to the container

3. The workflow will run automatically during market hours or can be triggered manually

## Data Structure

Data is saved in the Azure container with the following structure:
```
raw_data/
├── raw_data_1min_YYYYMMDD_HHMMSS.parquet  # Timestamped 1-minute data
├── raw_data_1min_latest.parquet           # Latest 1-minute data
├── raw_data_5min_YYYYMMDD_HHMMSS.parquet  # Timestamped 5-minute data
└── raw_data_5min_latest.parquet           # Latest 5-minute data
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