import yfinance as yf
import pandas as pd

# Get AAPL data
df = yf.download("AAPL", period="10d", interval="15m", progress=False)

# Handle multi-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = [col.lower() for col in df.columns]
df = df.between_time("09:30", "16:00")

print(f"Total bars: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check daily grouping
daily = df.groupby(df.index.date).size()
print(f"\nBars per day:")
print(daily)

# Check a single day
for day, day_df in df.groupby(df.index.date):
    n_bars = len(day_df)
    print(f"\n{day}: {n_bars} bars")
    if n_bars >= 12:  # Enough for pattern
        print("Sample prices:", day_df['close'].values[:5])
        break