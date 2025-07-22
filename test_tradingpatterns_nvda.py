#!/usr/bin/env python3
"""Test tradingpatterns library with NVDA data"""

import yfinance as yf
import pandas as pd
from tradingpatterns.tradingpatterns import (
    detect_head_shoulder,
    detect_double_top_bottom,
    detect_triangle_pattern,
    detect_wedge,
    find_pivots
)

print("=== Testing TradingPatterns Library with NVDA ===\n")

# Download NVDA data
ticker = "NVDA"
print(f"Downloading {ticker} data...")
data = yf.download(ticker, period='5d', interval='5m', progress=False)

if data.empty:
    print("Error: No data retrieved")
    exit(1)

# Clean data for tradingpatterns library
# The library expects simple column names
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Current price: ${float(data['Close'].iloc[-1]):.2f}\n")

# Test pattern detection
print("Detecting patterns...")
print("-" * 60)

# 1. Head and Shoulders
print("\n1. Head and Shoulders Detection:")
try:
    hs_patterns = detect_head_shoulder(data)
    if hs_patterns and len(hs_patterns) > 0:
        print(f"   ✓ Found {len(hs_patterns)} head and shoulders pattern(s)")
        for i, pattern in enumerate(hs_patterns):
            print(f"   Pattern {i+1}: {pattern}")
    else:
        print("   - No head and shoulders patterns found")
except Exception as e:
    print(f"   Error: {e}")

# 2. Double Top/Bottom
print("\n2. Double Top/Bottom Detection:")
try:
    dt_patterns = detect_double_top_bottom(data)
    if dt_patterns and len(dt_patterns) > 0:
        print(f"   ✓ Found {len(dt_patterns)} double top/bottom pattern(s)")
        for i, pattern in enumerate(dt_patterns):
            print(f"   Pattern {i+1}: {pattern}")
    else:
        print("   - No double top/bottom patterns found")
except Exception as e:
    print(f"   Error: {e}")

# 3. Triangle Patterns
print("\n3. Triangle Pattern Detection:")
try:
    triangle_patterns = detect_triangle_pattern(data)
    if triangle_patterns and len(triangle_patterns) > 0:
        print(f"   ✓ Found {len(triangle_patterns)} triangle pattern(s)")
        for i, pattern in enumerate(triangle_patterns):
            print(f"   Pattern {i+1}: {pattern}")
    else:
        print("   - No triangle patterns found")
except Exception as e:
    print(f"   Error: {e}")

# 4. Wedge Patterns
print("\n4. Wedge Pattern Detection:")
try:
    wedge_patterns = detect_wedge(data)
    if wedge_patterns and len(wedge_patterns) > 0:
        print(f"   ✓ Found {len(wedge_patterns)} wedge pattern(s)")
        for i, pattern in enumerate(wedge_patterns):
            print(f"   Pattern {i+1}: {pattern}")
    else:
        print("   - No wedge patterns found")
except Exception as e:
    print(f"   Error: {e}")

# 5. Find Pivots (useful for pattern detection)
print("\n5. Pivot Points Detection:")
try:
    pivots = find_pivots(data)
    print(f"   ✓ Found pivot points")
    print(f"   Shape: {pivots.shape if hasattr(pivots, 'shape') else 'N/A'}")
    if hasattr(pivots, 'head'):
        print(f"   First few pivots:\n{pivots.head()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)
print("Test completed!")