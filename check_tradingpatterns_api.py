#!/usr/bin/env python3
"""Check the actual API of tradingpatterns functions"""

from tradingpatterns.tradingpatterns import detect_head_shoulder, detect_double_top_bottom
import inspect

print("=== TradingPatterns API Check ===\n")

# Check detect_head_shoulder
print("1. detect_head_shoulder:")
print(f"   Signature: {inspect.signature(detect_head_shoulder)}")
print(f"   Docstring: {detect_head_shoulder.__doc__}")

print("\n2. detect_double_top_bottom:")
print(f"   Signature: {inspect.signature(detect_double_top_bottom)}")
print(f"   Docstring: {detect_double_top_bottom.__doc__}")

# Let's also check the source to understand the expected input
print("\n3. Source code snippet of detect_head_shoulder:")
try:
    source = inspect.getsource(detect_head_shoulder)
    lines = source.split('\n')[:20]  # First 20 lines
    for line in lines:
        print(f"   {line}")
except Exception as e:
    print(f"   Error getting source: {e}")