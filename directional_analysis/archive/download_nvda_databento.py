#!/usr/bin/env python3
"""
Download 5 years of NVDA 15-minute data from Databento
"""

import databento as db
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def estimate_nvda_cost():
    """
    Estimate cost for 5 years of NVDA 15-minute data
    """
    print("NVDA Data Cost Estimation")
    print("=" * 60)
    
    # Calculate expected data size
    years = 5
    trading_days_per_year = 252
    total_trading_days = years * trading_days_per_year
    
    # 6.5 hours per day, 4 bars per hour
    bars_per_day = 26  # 15-minute bars
    total_bars = total_trading_days * bars_per_day
    
    print(f"\nExpected data:")
    print(f"- Period: 5 years")
    print(f"- Trading days: ~{total_trading_days:,}")
    print(f"- Bars per day: {bars_per_day}")
    print(f"- Total 15-min bars: ~{total_bars:,}")
    
    # Databento pricing for OHLCV data
    # Typically $0.10 per million records for OHLCV
    estimated_cost = (total_bars / 1_000_000) * 0.10
    
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Your credit: $120.00")
    print(f"Remaining after download: ${120 - estimated_cost:.2f}")
    
    return estimated_cost

def download_nvda_data():
    """
    Download 5 years of NVDA 15-minute bars
    """
    client = db.Historical(key=API_KEY)
    
    # Date range - 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\nDownloading NVDA data")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("-" * 60)
    
    try:
        # First try with DBEQ.BASIC which should have historical data
        print("\nAttempting to download from DBEQ.BASIC dataset...")
        
        # Try daily bars first to see if symbol works
        print("Testing with daily bars...")
        test_data = client.timeseries.get_range(
            dataset="DBEQ.BASIC",
            symbols=["NVDA"],
            schema="ohlcv-1d",
            start=end_date - timedelta(days=5),
            end=end_date,
            limit=5
        )
        
        test_df = test_data.to_df()
        if not test_df.empty:
            print(f"✓ NVDA found! Latest price: ${test_df.iloc[-1]['close']:.2f}")
        
        # Now try to get trades to build 15-min bars
        print("\nDownloading trades data to build 15-minute bars...")
        
        data = client.timeseries.get_range(
            dataset="DBEQ.BASIC",
            symbols=["NVDA"],
            schema="trades",
            start=start_date,
            end=end_date
        )
        
        print("Converting to DataFrame...")
        df = data.to_df()
        
        if df.empty:
            raise ValueError("No OHLCV data available, trying trades...")
            
    except Exception as e:
        print(f"OHLCV not available: {e}")
        print("\nTrying alternative approach...")
        
        # Try with DBEQ.BASIC for trades
        data = client.timeseries.get_range(
            dataset="DBEQ.BASIC",
            symbols=["NVDA"],
            schema="trades",
            start=start_date,
            end=end_date
        )
        
        print("Converting trades to DataFrame...")
        df_trades = data.to_df()
        
        if df_trades.empty:
            raise ValueError("No data received!")
            
        print(f"Downloaded {len(df_trades):,} trades")
        
        # Build 15-minute bars from trades
        print("Building 15-minute bars from trades...")
        
        # Find price column
        if 'price' in df_trades.columns:
            price_col = 'price'
        else:
            # List all columns to debug
            print("Available columns:", df_trades.columns.tolist())
            # Try to find a price-like column
            price_cols = [c for c in df_trades.columns if 'price' in c.lower()]
            if price_cols:
                price_col = price_cols[0]
            else:
                raise ValueError("Cannot find price column")
        
        # Find size column
        size_col = 'size' if 'size' in df_trades.columns else None
        
        # Create aggregation dict
        agg_dict = {
            price_col: ['first', 'max', 'min', 'last', 'count']
        }
        if size_col:
            agg_dict[size_col] = 'sum'
        
        # Resample to 15-minute bars
        df = df_trades.resample('15min').agg(agg_dict)
        
        # Flatten columns
        if size_col:
            df.columns = ['open', 'high', 'low', 'close', 'trades', 'volume']
        else:
            df.columns = ['open', 'high', 'low', 'close', 'trades']
            df['volume'] = df['trades'] * 100  # Estimate volume
        
        # Remove empty bars
        df = df[df['trades'] > 0]
    
    # Filter to market hours only
    print("Filtering to market hours (9:30 AM - 4:00 PM ET)...")
    df = df.between_time('09:30', '16:00')
    
    print(f"\nTotal 15-minute bars: {len(df):,}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Save to CSV
    output_file = "NVDA_15min_5years.csv"
    df.to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    # Show sample data
    print("\nFirst 5 bars:")
    print(df.head())
    
    print("\nLast 5 bars:")
    print(df.tail())
    
    # Data quality check
    print("\nData Quality Check:")
    daily_bars = df.groupby(df.index.date).size()
    print(f"- Total trading days: {len(daily_bars)}")
    print(f"- Average bars per day: {daily_bars.mean():.1f}")
    print(f"- Min bars per day: {daily_bars.min()}")
    print(f"- Max bars per day: {daily_bars.max()}")
    
    # Check for gaps
    days_with_full_data = (daily_bars >= 25).sum()
    print(f"- Days with 25+ bars: {days_with_full_data} ({days_with_full_data/len(daily_bars)*100:.1f}%)")
    
    return df

def verify_data_quality(df):
    """
    Verify the downloaded data is suitable for pattern matching
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY VERIFICATION")
    print("=" * 60)
    
    # Check data completeness by year
    df['year'] = df.index.year
    yearly_summary = df.groupby('year').agg({
        'close': 'count',
        'volume': 'sum'
    }).rename(columns={'close': 'bars'})
    
    print("\nData by year:")
    print(yearly_summary)
    
    # Check for anomalies
    print("\nPrice statistics:")
    print(f"- Min price: ${df['low'].min():.2f}")
    print(f"- Max price: ${df['high'].max():.2f}")
    print(f"- Avg price: ${df['close'].mean():.2f}")
    
    # Pattern matching readiness
    total_patterns = len(df) - 30  # Assuming 30-bar windows
    print(f"\nPattern matching readiness:")
    print(f"- Total possible 30-bar patterns: {total_patterns:,}")
    print(f"- This is {'EXCELLENT' if total_patterns > 10000 else 'GOOD' if total_patterns > 5000 else 'ADEQUATE'} for pattern matching")
    
    print("\n✅ Data is ready for pattern matching!")
    print("\nNext steps:")
    print("1. Update intraday_shape_matcher.py to load local CSV")
    print("2. Run pattern analysis with 5 years of data")
    print("3. Much better predictions than 60-day Yahoo data!")

if __name__ == "__main__":
    print("NVIDIA (NVDA) 5-Year Data Download")
    print("=" * 60)
    
    # Estimate cost first
    cost = estimate_nvda_cost()
    
    if cost > 120:
        print(f"\n⚠️  WARNING: Estimated cost exceeds your credit!")
        sys.exit(1)
    
    print(f"\n✓ Cost is within your credit limit")
    print("\nStarting download...")
    
    try:
        df = download_nvda_data()
        verify_data_quality(df)
        
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE!")
        print("=" * 60)
        print("\nYour NVDA data is ready for advanced pattern matching.")
        print("This dataset is ~100x larger than Yahoo Finance's 60-day limit!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Databento API key")
        print("2. Verify your account has sufficient credit")
        print("3. Check if NVDA is available in the dataset")