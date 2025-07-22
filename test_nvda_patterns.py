#!/usr/bin/env python3
"""Test pattern scanner specifically for NVDA latest trading day"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pattern_scanner
import yfinance as yf
from datetime import datetime

# Download NVDA data
ticker = "NVDA"
print(f"\n{'='*60}")
print(f"NVDA Pattern Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}\n")

# Get intraday data
data = yf.download(ticker, period='1d', interval='5m', progress=False)

if not data.empty:
    print(f"Trading Day: {data.index[0].strftime('%Y-%m-%d')}")
    print(f"Data Points: {len(data)} (5-minute candles)")
    print(f"Price Range: ${float(data['Low'].min()):.2f} - ${float(data['High'].max()):.2f}")
    print(f"Current Price: ${float(data['Close'].iloc[-1]):.2f}\n")
    
    # Clean and scan
    cleaned_data = pattern_scanner.clean_yfinance_data(data)
    
    patterns = {
        "Head and Shoulders": pattern_scanner.find_head_and_shoulders,
        "Inverse Head and Shoulders": pattern_scanner.find_inverse_head_and_shoulders,
        "Double Top": pattern_scanner.find_double_top,
        "Double Bottom": pattern_scanner.find_double_bottom
    }
    
    print("Patterns Detected:")
    print("-" * 60)
    
    found_patterns = []
    
    for pattern_name, find_func in patterns.items():
        result = find_func(cleaned_data)
        if result is not None:
            signals = pattern_scanner.calculate_trading_signals(cleaned_data, pattern_name, result)
            found_patterns.append((pattern_name, signals))
            
            print(f"\n✓ {pattern_name}")
            print(f"  • Action: {signals['action']} @ ${signals['entry']}")
            print(f"  • Stop Loss: ${signals['stop_loss']}")
            print(f"  • Target: ${signals['target']}")
            print(f"  • Risk/Reward: {signals['risk_reward']}")
            print(f"  • Confidence: {signals['confidence']}%")
    
    if not found_patterns:
        print("\nNo patterns detected in today's trading session.")
    else:
        print(f"\n{'='*60}")
        print(f"Summary: {len(found_patterns)} pattern(s) found")
        
        # Check if using TradingPatternScanner
        if pattern_scanner.HAS_TRADING_PATTERNS:
            print("Detection Method: TradingPatternScanner (84.5% accuracy)")
        else:
            print("Detection Method: Basic scipy implementation")
        print(f"{'='*60}")

else:
    print("Error: Unable to download data")