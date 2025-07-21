#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download NVDA data
ticker = yf.Ticker('NVDA')

# Yahoo Finance only provides 15-minute data for the last 60 days
end_date = datetime.now()
start_date = end_date - timedelta(days=59)  # Stay within 60-day limit

print(f"Downloading NVDA 15-minute data from {start_date.date()} to {end_date.date()}")

# Download data
data = ticker.history(start=start_date, end=end_date, interval='15m')

if len(data) > 0:
    # Format for your system
    data.index.name = 'timestamp'
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Save to file
    data.to_csv('data/NVDA_15min_pattern_ready.csv')
    print(f'Downloaded {len(data)} rows of NVDA data')
    print(f'Date range: {data.index[0]} to {data.index[-1]}')
else:
    print("No data downloaded. This might be due to:")
    print("- Market is closed and no recent data available")
    print("- Ticker symbol issue")
    print("- Network connectivity")
    
    # Create empty file to prevent workflow failure
    empty_df = pd.DataFrame(columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    empty_df.to_csv('data/NVDA_15min_pattern_ready.csv', index=False)
    print("Created empty data file to prevent workflow failure")