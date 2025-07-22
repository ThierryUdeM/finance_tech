#!/usr/bin/env python3
"""Test the actual tradingpatterns library"""

try:
    from tradingpatterns import tradingpatterns
    print("✓ Successfully imported tradingpatterns.tradingpatterns")
    
    print("\nAvailable functions:")
    for attr in dir(tradingpatterns):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            
except ImportError as e:
    print(f"Import error: {e}")

# Try alternative imports
print("\nTrying alternative imports:")

try:
    from tradingpatterns.tradingpatterns import *
    print("✓ Successfully imported from tradingpatterns.tradingpatterns")
    
    # List what was imported
    import sys
    current_module = sys.modules[__name__]
    for name in dir(current_module):
        if not name.startswith('_') and name not in ['sys', 'tradingpatterns']:
            print(f"  - {name}")
            
except ImportError as e:
    print(f"Import error: {e}")

# Try to check the main tradingpatterns file
print("\nChecking tradingpatterns.py content:")
try:
    import tradingpatterns.tradingpatterns as tp
    print("Functions in tradingpatterns.tradingpatterns:")
    for attr in dir(tp):
        if not attr.startswith('_'):
            obj = getattr(tp, attr)
            print(f"  - {attr} ({type(obj).__name__})")
except Exception as e:
    print(f"Error: {e}")