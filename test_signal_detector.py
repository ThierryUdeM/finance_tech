#!/usr/bin/env python3
"""Test the signal detector with market closed handling"""

import os
import sys

# Set test environment
os.environ['TEST_MODE'] = 'true'
os.environ['TICKERS'] = 'NVDA,MSFT,TSLA'

# Import and run the signal detector
print("=== Testing Signal Detector ===")
print("\nThis will test the market closed handling by analyzing the last trading day's data.\n")

try:
    import signal_detector
    signal_detector.main()
    print("\n✓ Test completed successfully!")
except Exception as e:
    print(f"\n✗ Test failed: {str(e)}")
    sys.exit(1)