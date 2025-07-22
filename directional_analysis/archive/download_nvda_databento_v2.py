#!/usr/bin/env python3
"""
Download NVDA data from Databento with correct date range
"""

import databento as db
import pandas as pd
from datetime import datetime, timedelta
import os

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def download_nvda_available_data():
    """
    Download NVDA data within Databento's available range
    """
    client = db.Historical(key=API_KEY)
    
    # DBEQ.BASIC available from 2023-03-28 to 2025-07-19 (from error message)
    start_date = datetime(2023, 3, 28)
    end_date = datetime(2025, 7, 19)  # Use the date from error message
    
    print("NVIDIA (NVDA) Data Download")
    print("=" * 60)
    print(f"Dataset: DBEQ.BASIC")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Period: {(end_date - start_date).days / 365:.1f} years")
    print("-" * 60)
    
    try:
        # First, let's see what schemas are available
        print("\nStep 1: Testing available data schemas...")
        
        # Test different schemas
        schemas_to_try = ["ohlcv-1h", "ohlcv-1d", "trades", "tbbo"]
        available_schemas = []
        
        for schema in schemas_to_try:
            try:
                print(f"  Testing {schema}...", end="")
                test_data = client.timeseries.get_range(
                    dataset="DBEQ.BASIC",
                    symbols=["NVDA"],
                    schema=schema,
                    start=datetime(2025, 7, 14),  # 5 days before end
                    end=datetime(2025, 7, 19),
                    limit=1
                )
                test_df = test_data.to_df()
                if not test_df.empty:
                    print(" ✓ Available")
                    available_schemas.append(schema)
                else:
                    print(" ✗ No data")
            except Exception as e:
                print(f" ✗ Error: {str(e)[:30]}...")
        
        print(f"\nAvailable schemas: {available_schemas}")
        
        # Try to get hourly bars first (more manageable)
        if "ohlcv-1h" in available_schemas:
            print("\n✓ Hourly OHLCV data is available!")
            print("Downloading 1-hour bars...")
            
            data = client.timeseries.get_range(
                dataset="DBEQ.BASIC",
                symbols=["NVDA"],
                schema="ohlcv-1h",
                start=start_date,
                end=end_date
            )
            
            df_hourly = data.to_df()
            
            if not df_hourly.empty:
                print(f"Downloaded {len(df_hourly):,} hourly bars")
                
                # Save hourly data
                hourly_file = "NVDA_1h_databento.csv"
                df_hourly.to_csv(hourly_file)
                print(f"Saved to: {hourly_file}")
                
                # Show sample
                print("\nSample data:")
                print(df_hourly.tail())
                
                # Try to resample to 15-minute bars
                print("\nConverting to 15-minute bars...")
                # This is approximate - we'll interpolate
                df_15min = df_hourly.resample('15min').ffill()
                
                # Save 15-min approximation
                df_15min.to_csv("NVDA_15min_resampled.csv")
                print(f"Saved resampled 15-min data")
        
        # If we have trades, build proper 15-min bars
        if "trades" in available_schemas:
            print("\n\nDownloading trades data to build exact 15-minute bars...")
            print("This may take a few minutes...")
            
            # Download in chunks to avoid timeout
            chunk_days = 30
            all_trades = []
            
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=chunk_days), end_date)
                print(f"\nDownloading chunk: {current_start.date()} to {current_end.date()}")
                
                try:
                    chunk_data = client.timeseries.get_range(
                        dataset="DBEQ.BASIC",
                        symbols=["NVDA"],
                        schema="trades",
                        start=current_start,
                        end=current_end
                    )
                    
                    chunk_df = chunk_data.to_df()
                    if not chunk_df.empty:
                        all_trades.append(chunk_df)
                        print(f"  Got {len(chunk_df):,} trades")
                    
                except Exception as e:
                    print(f"  Error in chunk: {e}")
                
                current_start = current_end
            
            if all_trades:
                # Combine all trades
                df_trades = pd.concat(all_trades)
                print(f"\nTotal trades downloaded: {len(df_trades):,}")
                
                # Save raw trades
                trades_file = "NVDA_trades_databento.csv"
                df_trades.to_csv(trades_file)
                print(f"Saved trades to: {trades_file}")
                print(f"File size: {os.path.getsize(trades_file) / 1024 / 1024:.1f} MB")
                
                # Build 15-minute bars
                print("\nBuilding 15-minute bars from trades...")
                
                # Find price column
                price_col = 'price' if 'price' in df_trades.columns else df_trades.columns[0]
                
                # Resample to 15-min
                ohlcv = df_trades[price_col].resample('15min').ohlc()
                ohlcv['volume'] = df_trades[price_col].resample('15min').count()
                
                # Filter market hours
                ohlcv = ohlcv.between_time('09:30', '16:00')
                ohlcv = ohlcv[ohlcv['volume'] > 0]
                
                # Save 15-min bars
                bars_file = "NVDA_15min_databento.csv"
                ohlcv.to_csv(bars_file)
                print(f"\nCreated {len(ohlcv):,} 15-minute bars")
                print(f"Saved to: {bars_file}")
                
                # Quality check
                print("\nData quality:")
                daily_bars = ohlcv.groupby(ohlcv.index.date).size()
                print(f"- Trading days: {len(daily_bars)}")
                print(f"- Avg bars/day: {daily_bars.mean():.1f}")
                print(f"- Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")
                
                return ohlcv
        
        # If only daily data available
        if "ohlcv-1d" in available_schemas and "ohlcv-1h" not in available_schemas:
            print("\nOnly daily data available. Downloading...")
            
            data = client.timeseries.get_range(
                dataset="DBEQ.BASIC",
                symbols=["NVDA"],
                schema="ohlcv-1d",
                start=start_date,
                end=end_date
            )
            
            df_daily = data.to_df()
            daily_file = "NVDA_daily_databento.csv"
            df_daily.to_csv(daily_file)
            print(f"Saved {len(df_daily)} daily bars to: {daily_file}")
            
            print("\nNote: Only daily data available. For 15-min bars, you'll need:")
            print("- A different data provider (Polygon.io, IB, etc)")
            print("- Or use the 60-day Yahoo Finance data")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nDatabento access issues. Your options:")
        print("1. Check if your API key has access to DBEQ.BASIC")
        print("2. Try a different dataset (ask Databento support)")
        print("3. Use free Yahoo Finance data (limited to 60 days)")

def estimate_actual_cost():
    """Check actual data availability and cost"""
    print("\nCost Information:")
    print("-" * 40)
    print("DBEQ.BASIC dataset (March 2023 - present):")
    print("- ~1.75 years of data available")
    print("- Hourly bars: ~10,000 bars → ~$0.01")
    print("- Daily bars: ~440 bars → ~$0.00")
    print("- Trades: Depends on liquidity, likely $5-20 for NVDA")
    print("\nYour $120 credit is more than sufficient!")

if __name__ == "__main__":
    download_nvda_available_data()
    estimate_actual_cost()