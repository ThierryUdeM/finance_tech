#!/usr/bin/env python3
"""Test script to simulate download_intraday_data.py functionality"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to path
sys.path.append('/home/thierrygc/test_1/github_ready/ChartScanAI_Shiny')

def test_download_simulation():
    """Simulate the download process with dummy data"""
    
    # Create test directory
    test_dir = '/home/thierrygc/test_1/github_ready/test_workflow_env/directional_analysis'
    os.makedirs(test_dir, exist_ok=True)
    
    # Create dummy data similar to what yfinance would return
    now = datetime.now()
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Generate 15-minute intervals
    timestamps = []
    for i in range(26):  # 9:30 AM to 4:00 PM = 26 intervals
        timestamps.append(start_time + timedelta(minutes=15*i))
    
    # Create dummy OHLCV data
    data = {
        'Open': [100 + i*0.5 for i in range(26)],
        'High': [101 + i*0.5 for i in range(26)],
        'Low': [99 + i*0.5 for i in range(26)],
        'Close': [100.5 + i*0.5 for i in range(26)],
        'Volume': [1000000 + i*10000 for i in range(26)]
    }
    
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp'
    
    # Save to CSV
    output_path = os.path.join(test_dir, 'NVDA_intraday_current.csv')
    df.to_csv(output_path)
    
    print(f"Test simulation complete!")
    print(f"Created dummy data with {len(df)} bars")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    success = test_download_simulation()
    exit(0 if success else 1)