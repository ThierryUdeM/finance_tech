#!/usr/bin/env python3
"""
Test script for Simple Technical Scanner
Run this locally to verify the scanner works before deploying
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock Azure storage for testing
import unittest.mock as mock
sys.modules['azure'] = mock.MagicMock()
sys.modules['azure.storage'] = mock.MagicMock()
sys.modules['azure.storage.blob'] = mock.MagicMock()

from simple_technical_scanner import SimpleTechnicalScanner

def main():
    """Test the simple technical scanner"""
    print("Testing Simple Technical Scanner...")
    
    # Create scanner instance
    scanner = SimpleTechnicalScanner()
    
    # Test scanning NVDA
    print("\nScanning NVDA...")
    signal = scanner.scan_ticker('NVDA', interval='15m')
    
    if signal:
        print(f"\nSignal Generated:")
        print(f"  Signal: {signal['signal']}")
        print(f"  Confidence: {signal['confidence']*100:.1f}%")
        print(f"  Price: ${signal['price']:.2f}")
        print(f"  Stop Loss: ${signal['stop_loss']:.2f}" if signal['stop_loss'] else "  Stop Loss: N/A")
        print(f"  Take Profit: ${signal['take_profit']:.2f}" if signal['take_profit'] else "  Take Profit: N/A")
        print(f"\nIndicators:")
        for key, value in signal['indicators'].items():
            if value is not None:
                print(f"  {key}: {value}")
        print(f"\nComponents:")
        for key, value in signal['components'].items():
            print(f"  {key}: {value}")
    else:
        print("No signal generated - check error messages above")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()