import yfinance as yf
import pandas as pd

# Test Yahoo Finance data fetch
ticker = "AAPL"
df = yf.download(ticker, period="5d", interval="15m", progress=False)

print(f"Downloaded {len(df)} bars")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Check structure
print("\nFirst few rows:")
print(df.head())

# Filter market hours
df_market = df.between_time("09:30", "16:00")
print(f"\nMarket hours only: {len(df_market)} bars")

# Check daily grouping
daily = df_market.groupby(df_market.index.date).size()
print(f"\nBars per day:")
print(daily)