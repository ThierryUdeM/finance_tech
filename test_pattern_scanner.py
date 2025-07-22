#!/usr/bin/env python3
"""Test pattern scanner with NVDA data"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment to use scipy fallback for testing
os.environ['SKIP_TRADING_PATTERNS'] = '1'

import pattern_scanner
import yfinance as yf
from datetime import datetime

def test_pattern_scanner():
    """Test the pattern scanner with NVDA data"""
    
    print("=" * 80)
    print("PATTERN SCANNER TEST - NVDA")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Download NVDA data for the latest trading day
    ticker = "NVDA"
    print(f"\nDownloading {ticker} data (1 day, 5-minute intervals)...")
    
    data = yf.download(ticker, period='1d', interval='5m', progress=False)
    
    if data.empty:
        print("Error: No data retrieved")
        return
    
    print(f"Data retrieved: {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Latest price: ${float(data['Close'].iloc[-1]):.2f}")
    
    # Clean the data
    print("\nCleaning data...")
    cleaned_data = pattern_scanner.clean_yfinance_data(data)
    
    # Define patterns to check
    patterns = {
        "Head and Shoulders": pattern_scanner.find_head_and_shoulders,
        "Inverse Head and Shoulders": pattern_scanner.find_inverse_head_and_shoulders,
        "Double Top": pattern_scanner.find_double_top,
        "Double Bottom": pattern_scanner.find_double_bottom
    }
    
    # Check for patterns
    print("\nScanning for patterns...")
    print("-" * 80)
    
    patterns_found = 0
    
    for pattern_name, find_func in patterns.items():
        print(f"\nChecking for {pattern_name}...", end=" ")
        
        try:
            result = find_func(cleaned_data)
            
            if result is not None:
                print("FOUND!")
                patterns_found += 1
                
                # Calculate trading signals
                signals = pattern_scanner.calculate_trading_signals(cleaned_data, pattern_name, result)
                
                print(f"\n  Pattern Details:")
                print(f"    - Action: {signals['action']}")
                print(f"    - Current Price: ${signals['current_price']}")
                print(f"    - Entry Price: ${signals['entry']}")
                print(f"    - Stop Loss: ${signals['stop_loss']}")
                print(f"    - Target Price: ${signals['target']}")
                print(f"    - Risk/Reward Ratio: {signals['risk_reward']}")
                print(f"    - Distance to Entry: {signals['distance_to_entry']}%")
                print(f"    - Confidence: {signals['confidence']}%")
                
                # Show pattern points
                if pattern_name in ["Double Top", "Double Bottom"]:
                    if pattern_name == "Double Top":
                        p1, p2, valley = result
                        print(f"    - Peak 1: Index {p1} (${cleaned_data['High'].iloc[int(p1)]:.2f})")
                        print(f"    - Peak 2: Index {p2} (${cleaned_data['High'].iloc[int(p2)]:.2f})")
                        print(f"    - Valley: Index {valley} (${cleaned_data['Low'].iloc[int(valley)]:.2f})")
                    else:
                        t1, t2, peak = result
                        print(f"    - Trough 1: Index {t1} (${cleaned_data['Low'].iloc[int(t1)]:.2f})")
                        print(f"    - Trough 2: Index {t2} (${cleaned_data['Low'].iloc[int(t2)]:.2f})")
                        print(f"    - Peak: Index {peak} (${cleaned_data['High'].iloc[int(peak)]:.2f})")
            else:
                print("Not found")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {patterns_found} pattern(s)")
    print("=" * 80)
    
    # Test with different time periods
    print("\n\nTesting with longer period (5 days)...")
    data_5d = yf.download(ticker, period='5d', interval='5m', progress=False)
    
    if not data_5d.empty:
        print(f"Data retrieved: {len(data_5d)} candles")
        cleaned_data_5d = pattern_scanner.clean_yfinance_data(data_5d)
        
        patterns_found_5d = 0
        for pattern_name, find_func in patterns.items():
            result = find_func(cleaned_data_5d)
            if result is not None:
                patterns_found_5d += 1
                print(f"  - Found {pattern_name}")
        
        print(f"\nFound {patterns_found_5d} pattern(s) in 5-day data")

if __name__ == "__main__":
    test_pattern_scanner()