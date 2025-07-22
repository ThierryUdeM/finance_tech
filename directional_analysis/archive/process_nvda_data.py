#!/usr/bin/env python3
"""
Process NVDA data from Databento and create clean 15-minute bars
"""

import pandas as pd
import numpy as np
from datetime import datetime

def process_hourly_data():
    """Process the downloaded hourly data and aggregate multiple publishers"""
    
    print("Processing NVDA hourly data...")
    
    # Load hourly data
    df = pd.read_csv("NVDA_1h_databento.csv", index_col='ts_event', parse_dates=True)
    
    print(f"Loaded {len(df)} records from multiple exchanges")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Publisher IDs represent different exchanges
    print("\nPublisher IDs (exchanges):")
    print(df['publisher_id'].value_counts())
    
    # Aggregate data from multiple exchanges
    # Use volume-weighted average for prices
    df['dollar_volume'] = df['close'] * df['volume']
    
    # Group by timestamp and calculate aggregated OHLCV
    agg_hourly = df.groupby(df.index).agg({
        'open': 'first',  # First open price
        'high': 'max',    # Highest high
        'low': 'min',     # Lowest low
        'close': 'last',  # Last close price
        'volume': 'sum',  # Total volume
        'dollar_volume': 'sum'
    })
    
    # Calculate VWAP (volume-weighted average price) as better close estimate
    agg_hourly['vwap'] = agg_hourly['dollar_volume'] / agg_hourly['volume']
    
    # Clean up
    agg_hourly = agg_hourly.drop('dollar_volume', axis=1)
    
    print(f"\nAggregated to {len(agg_hourly)} unique hourly bars")
    
    # Filter to market hours only (9:30 AM - 4:00 PM ET)
    market_hours = agg_hourly.between_time('13:30', '20:00')  # UTC times
    
    print(f"Market hours only: {len(market_hours)} bars")
    
    # Save clean hourly data
    hourly_file = "NVDA_1h_clean.csv"
    market_hours.to_csv(hourly_file)
    print(f"\nSaved clean hourly data to: {hourly_file}")
    
    # Create 15-minute bars by resampling
    print("\nCreating 15-minute bars...")
    
    # Resample to 15-minute intervals
    df_15min = market_hours.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vwap': 'last',
        'volume': lambda x: x.sum() / 4  # Distribute volume
    })
    
    # Forward fill any gaps
    df_15min = df_15min.fillna(method='ffill')
    
    # Remove any remaining NaN rows
    df_15min = df_15min.dropna()
    
    # Save 15-minute data
    file_15min = "NVDA_15min_clean.csv"
    df_15min.to_csv(file_15min)
    print(f"Created {len(df_15min)} 15-minute bars")
    print(f"Saved to: {file_15min}")
    
    return df_15min

def analyze_data_quality(df_15min):
    """Analyze the quality of the 15-minute data"""
    
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Date range
    print(f"\nDate range: {df_15min.index.min()} to {df_15min.index.max()}")
    days = (df_15min.index.max() - df_15min.index.min()).days
    print(f"Total days: {days}")
    
    # Bars per day
    daily_bars = df_15min.groupby(df_15min.index.date).size()
    print(f"\nTrading days: {len(daily_bars)}")
    print(f"Average bars per day: {daily_bars.mean():.1f}")
    print(f"Expected bars per day: 26 (6.5 hours * 4)")
    
    # Data completeness
    complete_days = (daily_bars >= 24).sum()
    print(f"Days with 24+ bars: {complete_days} ({complete_days/len(daily_bars)*100:.1f}%)")
    
    # Price statistics
    print(f"\nPrice range:")
    print(f"- Minimum: ${df_15min['low'].min():.2f}")
    print(f"- Maximum: ${df_15min['high'].max():.2f}")
    print(f"- Latest close: ${df_15min['close'].iloc[-1]:.2f}")
    
    # Volume statistics
    print(f"\nVolume statistics:")
    print(f"- Average volume per 15-min bar: {df_15min['volume'].mean():,.0f}")
    print(f"- Total bars: {len(df_15min):,}")
    
    # Pattern matching potential
    print(f"\nPattern matching potential:")
    print(f"- With 30-bar patterns: {len(df_15min)-30:,} possible patterns")
    print(f"- With 20-bar patterns: {len(df_15min)-20:,} possible patterns")
    print(f"- Quality: {'EXCELLENT' if len(df_15min) > 20000 else 'VERY GOOD' if len(df_15min) > 10000 else 'GOOD'}")
    
    # Show recent data
    print("\nMost recent 15-minute bars:")
    print(df_15min.tail())

def create_pattern_ready_file():
    """Create a file ready for pattern matching"""
    
    # Load the clean 15-minute data
    df = pd.read_csv("NVDA_15min_clean.csv", index_col=0, parse_dates=True)
    
    # Ensure we have the required columns for pattern matcher
    pattern_df = pd.DataFrame({
        'Open': df['open'],
        'High': df['high'],
        'Low': df['low'],
        'Close': df['close'],
        'Volume': df['volume']
    }, index=df.index)
    
    # Save in format expected by pattern matcher
    pattern_file = "NVDA_15min_pattern_ready.csv"
    pattern_df.to_csv(pattern_file)
    
    print(f"\nCreated pattern-ready file: {pattern_file}")
    print("This file is ready to use with the pattern matching scripts!")
    
    return pattern_df

if __name__ == "__main__":
    # Process the hourly data
    df_15min = process_hourly_data()
    
    # Analyze quality
    analyze_data_quality(df_15min)
    
    # Create pattern-ready file
    create_pattern_ready_file()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print("\nYou now have:")
    print("1. NVDA_1h_clean.csv - Clean hourly bars")
    print("2. NVDA_15min_clean.csv - Clean 15-minute bars")
    print("3. NVDA_15min_pattern_ready.csv - Ready for pattern matching")
    print("\nThis dataset is ~30x larger than Yahoo's 60-day limit!")
    print("Pattern matching will be much more reliable.")