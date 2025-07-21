#!/usr/bin/env python3
"""
Download current intraday data for NVDA
Used by the GitHub Actions workflow
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_intraday_data():
    """Download today's NVDA intraday data"""
    
    # Create directory if needed
    os.makedirs('../directional_analysis', exist_ok=True)
    
    # Download NVDA data
    ticker = yf.Ticker("NVDA")
    
    # Get today's data (or last trading day)
    today = datetime.now()
    
    # Download 5 days to ensure we get the latest trading day
    end_date = today + timedelta(days=1)
    start_date = today - timedelta(days=5)
    
    try:
        # Download intraday data
        data = ticker.history(start=start_date, end=end_date, interval="15m")
        
        if len(data) > 0:
            # Filter to get only today's data (last trading day)
            today_data = data[data.index.date == data.index.date[-1]]
            
            # Format for the system
            today_data.index.name = 'timestamp'
            today_data = today_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save today's data
            today_data.to_csv('../directional_analysis/NVDA_intraday_current.csv')
            print(f"Downloaded {len(today_data)} bars of NVDA intraday data")
            print(f"Time range: {today_data.index[0]} to {today_data.index[-1]}")
            print(f"Current price: ${today_data['Close'].iloc[-1]:.2f}")
            
            return True
        else:
            print("No intraday data available")
            return False
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False

if __name__ == "__main__":
    success = download_intraday_data()
    exit(0 if success else 1)