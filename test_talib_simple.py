#!/usr/bin/env python3
"""Simple TA-Lib test to verify installation"""

import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    # Try the pattern_detection venv which has working TA-Lib
    sys.path.insert(0, '/home/thierrygc/script/pattern_detection/venv/lib/python3.12/site-packages')
    import talib
    print(f"✓ TA-Lib imported successfully!")
    print(f"  Version: {talib.__version__}")
    print(f"  Location: {talib.__file__}")
    
    # Test a simple function
    import numpy as np
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = talib.SMA(close, timeperiod=3)
    print(f"  SMA test: {sma}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")