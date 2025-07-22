#!/usr/bin/env python3
"""Test script to verify deployment compatibility"""

import sys
print(f"Python version: {sys.version}")

# Test imports
print("\nTesting imports...")

try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except ImportError as e:
    print(f"✗ yfinance import failed: {e}")

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    from tradingpatterns.tradingpatterns import detect_head_shoulder
    print("✓ tradingpattern imported successfully")
except ImportError as e:
    print(f"✗ tradingpattern import failed: {e}")

try:
    import talib
    print(f"✓ TA-Lib imported successfully (version {talib.__version__})")
    
    # Test a simple TA-Lib function
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = talib.SMA(test_data, timeperiod=3)
    print(f"  SMA test result: {sma}")
except ImportError as e:
    print(f"✗ TA-Lib import failed: {e}")
except Exception as e:
    print(f"✗ TA-Lib error: {e}")

try:
    from azure.storage.blob import BlobServiceClient
    print("✓ Azure storage imported successfully")
except ImportError as e:
    print(f"✗ Azure storage import failed: {e}")

print("\nDeployment test completed!")