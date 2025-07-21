#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download NVDA data
ticker = yf.Ticker('NVDA')

# Get 2 years of 15-minute data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

# Download data
data = ticker.history(start=start_date, end=end_date, interval='15m')

# Format for your system
data.index.name = 'timestamp'
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Save to file
data.to_csv('data/NVDA_15min_pattern_ready.csv')
print(f'Downloaded {len(data)} rows of NVDA data')
print(f'Date range: {data.index[0]} to {data.index[-1]}')