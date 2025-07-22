#!/usr/bin/env python3
"""
Download historical 15-minute bars for AC.TO from Databento
"""

import databento as db
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# API Key
API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def estimate_cost(symbol, start_date, end_date, bar_size="15m"):
    """
    Estimate the cost of downloading data
    """
    # Databento charges ~$0.25 per million messages for OHLCV bars
    # 15-min bars: ~26 bars per day * 252 trading days * 5 years ≈ 32,760 bars
    # This is well under 1 million messages
    
    days = (end_date - start_date).days
    trading_days = int(days * 252 / 365)  # Approximate trading days
    bars_per_day = 26  # 9:30 AM to 4:00 PM = 6.5 hours * 4 bars/hour
    total_bars = trading_days * bars_per_day
    
    # Databento pricing: ~$0.25 per million for OHLCV
    estimated_cost = (total_bars / 1_000_000) * 0.25
    
    print(f"Estimated bars: {total_bars:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    return total_bars, estimated_cost

def download_ac_data():
    """
    Download 5 years of 15-minute bars for AC.TO
    """
    # Set up dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"Downloading AC.TO data from {start_date.date()} to {end_date.date()}")
    
    # Estimate cost first
    bars, cost = estimate_cost("AC.TO", start_date, end_date)
    
    if cost > 120:
        print(f"WARNING: Estimated cost ${cost:.2f} exceeds your $120 credit!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Initialize client
    client = db.Historical(key=API_KEY)
    
    try:
        # For Canadian stocks on TSX, we need to use the correct dataset
        # Databento uses different datasets for different exchanges
        # TSX data is typically in the 'XCAN' dataset
        
        print("\nDownloading data...")
        
        # Download 15-minute bars
        # Note: Databento uses specific symbology - AC.TO might be "AC.TSE" or "AC.XTSE"
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",  # Global exchanges dataset
            symbols=["AC.TO", "AC", "AC.TSE", "AC.XTSE"],  # Try multiple symbol formats
            schema="ohlcv-15m",  # 15-minute bars
            start=start_date,
            end=end_date,
            stype_in="raw_symbol"  # Use raw symbol format
        )
        
        # Convert to DataFrame
        df = data.to_df()
        
        if df.empty:
            print("No data received. Trying alternative approach...")
            # Try with different dataset
            data = client.timeseries.get_range(
                dataset="XCAN.ITCH",  # Canadian dataset
                symbols=["AC"],
                schema="ohlcv-15m",
                start=start_date,
                end=end_date
            )
            df = data.to_df()
        
        if not df.empty:
            # Save to CSV
            output_file = "AC_TO_15min_5years.csv"
            df.to_csv(output_file)
            print(f"\nData saved to {output_file}")
            print(f"Total bars downloaded: {len(df):,}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Show sample
            print("\nFirst few rows:")
            print(df.head())
            
            print("\nLast few rows:")
            print(df.tail())
            
            # Calculate actual cost (shown in account)
            print("\nCheck your Databento account for actual cost deducted from credit.")
            
        else:
            print("ERROR: No data received. Check symbol format or dataset availability.")
            print("\nYou may need to:")
            print("1. Check Databento's symbol reference for Canadian stocks")
            print("2. Verify which dataset contains TSX data")
            print("3. Use their data catalog to find the correct symbol")
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nTroubleshooting:")
        print("1. Check if AC.TO is available in Databento")
        print("2. You might need to use their symbol mapping API first")
        print("3. Consider using their support to identify the correct dataset/symbol")

def list_available_datasets():
    """
    List available datasets to find where TSX data is located
    """
    client = db.Historical(key=API_KEY)
    
    print("Checking available datasets...")
    try:
        # This would list available datasets
        # Note: Actual method might vary based on databento version
        metadata = client.metadata.list_datasets()
        print("Available datasets:", metadata)
    except Exception as e:
        print(f"Error listing datasets: {e}")

if __name__ == "__main__":
    print("Databento Data Downloader for AC.TO")
    print("=" * 50)
    
    # First, try to understand available datasets
    # list_available_datasets()
    
    # Download the data
    download_ac_data()