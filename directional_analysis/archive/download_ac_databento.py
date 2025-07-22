#!/usr/bin/env python3
"""
Download Air Canada data from Databento
"""

import databento as db
import pandas as pd
from datetime import datetime, timedelta
import os

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def download_ac_trades_and_build_bars():
    """
    Download AC trades data and build 15-minute OHLCV bars
    """
    client = db.Historical(key=API_KEY)
    
    # Date range - let's start with 1 year to test cost
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=365)  # 1 year for now
    
    print(f"Downloading Air Canada (AC) data")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Dataset: XNAS.ITCH (NASDAQ)")
    print("-" * 60)
    
    try:
        # Download trades data
        print("\nDownloading trades data...")
        
        data = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=["AC"],
            schema="trades",  # Raw trades
            start=start_date,
            end=end_date,
            stype_in="raw_symbol"
        )
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        df = data.to_df()
        
        if df.empty:
            print("ERROR: No data received!")
            return
            
        print(f"Downloaded {len(df):,} trades")
        print(f"Actual date range: {df.index.min()} to {df.index.max()}")
        
        # Save raw trades
        trades_file = "AC_trades_raw.csv"
        df.to_csv(trades_file)
        print(f"\nRaw trades saved to: {trades_file}")
        print(f"File size: {os.path.getsize(trades_file) / 1024 / 1024:.1f} MB")
        
        # Build 15-minute OHLCV bars from trades
        print("\nBuilding 15-minute OHLCV bars...")
        
        # Ensure we have a price column
        if 'price' in df.columns:
            price_col = 'price'
        elif 'trade_price' in df.columns:
            price_col = 'trade_price'
        else:
            print("Available columns:", df.columns.tolist())
            raise ValueError("Cannot find price column in trades data")
        
        # Resample to 15-minute bars
        agg_dict = {price_col: ['first', 'max', 'min', 'last', 'count']}
        if 'size' in df.columns:
            agg_dict['size'] = 'sum'
            
        ohlcv = df.resample('15min').agg(agg_dict)
        
        # Flatten column names
        if 'size' in df.columns:
            ohlcv.columns = ['open', 'high', 'low', 'close', 'trades', 'volume']
        else:
            ohlcv.columns = ['open', 'high', 'low', 'close', 'trades']
            ohlcv['volume'] = ohlcv['trades']  # Use trade count as proxy for volume
        
        # Remove bars with no trades
        ohlcv = ohlcv[ohlcv['trades'] > 0]
        
        # Filter to market hours only (9:30 AM - 4:00 PM ET)
        ohlcv = ohlcv.between_time('09:30', '16:00')
        
        print(f"Created {len(ohlcv):,} 15-minute bars")
        
        # Save OHLCV bars
        ohlcv_file = "AC_15min_bars.csv"
        ohlcv.to_csv(ohlcv_file)
        print(f"\n15-minute bars saved to: {ohlcv_file}")
        
        # Show sample
        print("\nFirst few bars:")
        print(ohlcv.head())
        
        print("\nLast few bars:")  
        print(ohlcv.tail())
        
        # Estimate cost
        print("\n" + "="*60)
        print("COST ESTIMATE")
        print("="*60)
        print(f"Total trades downloaded: {len(df):,}")
        print("Databento typically charges ~$0.003 per 1,000 trades")
        estimated_cost = (len(df) / 1000) * 0.003
        print(f"Estimated cost: ${estimated_cost:.2f}")
        
        if estimated_cost < 5:
            print("\nThis is well within your $120 credit!")
            print("\nTo download 5 years instead of 1 year:")
            print(f"Estimated cost would be: ${estimated_cost * 5:.2f}")
            
            response = input("\nDownload 5 years of data? (y/n): ")
            if response.lower() == 'y':
                download_5_years()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. AC might not be the correct symbol")
        print("2. The data might be on a different exchange")
        print("3. Contact Databento support for TSX data access")

def download_5_years():
    """Download 5 years of AC data"""
    client = db.Historical(key=API_KEY)
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\n\nDownloading 5 years of Air Canada data...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    try:
        data = client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=["AC"],
            schema="trades",
            start=start_date,
            end=end_date
        )
        
        df = data.to_df()
        
        if not df.empty:
            # Save 5-year trades
            df.to_csv("AC_trades_5years.csv")
            print(f"Downloaded {len(df):,} trades")
            
            # Build 15-min bars
            if 'price' in df.columns:
                price_col = 'price'
            elif 'trade_price' in df.columns:
                price_col = 'trade_price'
            else:
                price_col = df.select_dtypes(include=['float']).columns[0]
                
            ohlcv = df.resample('15min').agg({
                price_col: ['first', 'max', 'min', 'last', 'count']
            })
            ohlcv.columns = ['open', 'high', 'low', 'close', 'trades']
            ohlcv = ohlcv[ohlcv['trades'] > 0]
            ohlcv = ohlcv.between_time('09:30', '16:00')
            
            ohlcv.to_csv("AC_15min_5years.csv")
            print(f"Created {len(ohlcv):,} 15-minute bars")
            print("\nData saved successfully!")
            
    except Exception as e:
        print(f"Error: {e}")

def check_ac_on_tsx():
    """Alternative: Check if we need different symbol for TSX listing"""
    
    print("\n" + "="*60)
    print("ALTERNATIVE: Checking for AC on TSX")
    print("="*60)
    
    print("\nAir Canada is listed on TSX as AC.TO")
    print("However, Databento might require different access or dataset for TSX")
    print("\nOptions:")
    print("1. Use the NASDAQ listing if available (might be ADR)")
    print("2. Contact Databento support for TSX access")
    print("3. Use alternative data providers:")
    print("   - Polygon.io (has TSX data)")
    print("   - Alpha Vantage (limited free tier)")
    print("   - Interactive Brokers API")
    print("   - Questrade API (Canadian)")

if __name__ == "__main__":
    download_ac_trades_and_build_bars()
    check_ac_on_tsx()