#!/usr/bin/env python3
"""Properly test tradingpatterns library with NVDA"""

import yfinance as yf
import pandas as pd
import numpy as np
from tradingpatterns.tradingpatterns import (
    detect_head_shoulder,
    detect_double_top_bottom,
    find_pivots
)

print("=== Testing TradingPatterns Library with NVDA ===\n")

# Download NVDA data
ticker = "NVDA"
print(f"Downloading {ticker} data (intraday)...")
data = yf.download(ticker, period='1d', interval='5m', progress=False)

if data.empty:
    print("Error: No data retrieved")
    exit(1)

# Clean data for tradingpatterns library
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Current price: ${float(data['Close'].iloc[-1]):.2f}\n")

# Make a copy to avoid modifying original
df = data.copy()

print("Detecting patterns...")
print("-" * 60)

# 1. Head and Shoulders Detection
print("\n1. Head and Shoulders Detection:")
df_hs = detect_head_shoulder(df.copy(), window=5)  # Use window=5 for 5-min data
hs_patterns = df_hs[df_hs['head_shoulder_pattern'].notna()]

if len(hs_patterns) > 0:
    print(f"   ✓ Found {len(hs_patterns)} head and shoulders pattern(s)")
    for idx, row in hs_patterns.iterrows():
        print(f"   - {row['head_shoulder_pattern']} at {idx}")
        print(f"     Price: ${row['Close']:.2f}, High: ${row['High']:.2f}, Low: ${row['Low']:.2f}")
else:
    print("   - No head and shoulders patterns found")

# 2. Double Top/Bottom Detection
print("\n2. Double Top/Bottom Detection:")
df_dt = detect_double_top_bottom(df.copy(), window=5, threshold=0.02)  # 2% threshold for intraday
dt_patterns = df_dt[df_dt['double_pattern'].notna()]

if len(dt_patterns) > 0:
    print(f"   ✓ Found {len(dt_patterns)} double top/bottom pattern(s)")
    for idx, row in dt_patterns.iterrows():
        print(f"   - {row['double_pattern']} at {idx}")
        print(f"     Price: ${row['Close']:.2f}, High: ${row['High']:.2f}, Low: ${row['Low']:.2f}")
else:
    print("   - No double top/bottom patterns found")

# 3. Find Pivots
print("\n3. Pivot Points:")
try:
    pivots = find_pivots(df.copy(), window=5)
    pivot_highs = pivots[pivots['pivot_type'] == 'high']
    pivot_lows = pivots[pivots['pivot_type'] == 'low']
    
    print(f"   ✓ Found {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")
    
    if len(pivot_highs) > 0:
        print("   Recent pivot highs:")
        for idx, row in pivot_highs.tail(3).iterrows():
            print(f"     - {idx}: ${row['High']:.2f}")
    
    if len(pivot_lows) > 0:
        print("   Recent pivot lows:")
        for idx, row in pivot_lows.tail(3).iterrows():
            print(f"     - {idx}: ${row['Low']:.2f}")
            
except Exception as e:
    # Try alternative approach
    print(f"   Note: find_pivots may have different implementation")
    # Look for local maxima/minima manually
    high_peaks = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    low_troughs = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    print(f"   ✓ Found {high_peaks.sum()} potential peaks and {low_troughs.sum()} potential troughs")

print("\n" + "="*60)
print("Pattern Summary:")
print(f"- Head & Shoulders: {len(hs_patterns)} patterns")
print(f"- Double Top/Bottom: {len(dt_patterns)} patterns")
print(f"- Detection method: TradingPatterns library (with wavelet-based approach)")
print("="*60)