#!/usr/bin/env python3
"""Explore the tradingpattern library API"""

import tradingpatterns

print("=== TradingPattern Library Exploration ===\n")

# List all available attributes
print("Available attributes in tradingpatterns:")
for attr in dir(tradingpatterns):
    if not attr.startswith('_'):
        print(f"  - {attr}")
        obj = getattr(tradingpatterns, attr)
        print(f"    Type: {type(obj).__name__}")
        if hasattr(obj, '__doc__') and obj.__doc__:
            doc = obj.__doc__.split('\n')[0][:60]
            print(f"    Doc: {doc}...")

print("\n" + "="*50)

# Check for specific pattern detection functions
pattern_names = ['head_shoulder', 'double_top', 'double_bottom', 'wedge', 'triangle']
print("\nChecking for pattern detection methods:")
for pattern in pattern_names:
    for prefix in ['detect_', 'find_', 'scan_', '']:
        method_name = f"{prefix}{pattern}"
        if hasattr(tradingpatterns, method_name):
            print(f"  ✓ Found: {method_name}")

# Check if there's a Scanner or PatternScanner class
print("\nChecking for scanner classes:")
for class_name in ['Scanner', 'PatternScanner', 'TradingPatternScanner', 'PatternDetector']:
    if hasattr(tradingpatterns, class_name):
        print(f"  ✓ Found class: {class_name}")
        cls = getattr(tradingpatterns, class_name)
        # List methods of the class
        print("    Methods:")
        for method in dir(cls):
            if not method.startswith('_'):
                print(f"      - {method}")

# Try to import submodules
print("\nChecking for submodules:")
try:
    import tradingpattern.patterns
    print("  ✓ Found: tradingpattern.patterns")
except ImportError:
    pass

try:
    import tradingpattern.scanner
    print("  ✓ Found: tradingpattern.scanner")
except ImportError:
    pass