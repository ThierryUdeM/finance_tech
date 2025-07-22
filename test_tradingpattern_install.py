#!/usr/bin/env python3
"""Test if tradingpattern can be installed and imported"""

import subprocess
import sys

print("Testing tradingpattern installation...")

try:
    # Try to import
    import tradingpattern
    print("✓ tradingpattern is already installed")
    print(f"  Version: {tradingpattern.__version__ if hasattr(tradingpattern, '__version__') else 'unknown'}")
    
    # List available attributes
    print("\nAvailable attributes:")
    for attr in dir(tradingpattern):
        if not attr.startswith('_'):
            print(f"  - {attr}")
            
except ImportError:
    print("✗ tradingpattern not installed")
    print("\nAttempting to install...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tradingpattern"])
        print("\n✓ Installation successful!")
        
        # Try importing again
        import tradingpattern
        print("\nAvailable attributes:")
        for attr in dir(tradingpattern):
            if not attr.startswith('_'):
                print(f"  - {attr}")
                
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nTo install manually, try:")
        print("  pip install tradingpattern")
        print("  or")
        print("  pip install git+https://github.com/white07S/TradingPatternScanner.git")