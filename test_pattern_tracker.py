#!/usr/bin/env python3
"""
Test script for pattern signal tracker
"""

import os
import sys

# For local testing, add the script path
sys.path.insert(0, '/home/thierrygc/script/')

from pattern_signal_tracker import PatternSignalTracker
import logging

logging.basicConfig(level=logging.INFO)

def test_tracker():
    """Test the pattern tracker locally"""
    
    print("Testing Pattern Signal Tracker...")
    
    try:
        # Initialize tracker
        tracker = PatternSignalTracker()
        print("✓ Successfully initialized tracker with Azure credentials")
        
        # Test pattern scanning
        print("\nScanning for patterns in NVDA...")
        signals = tracker.scan_patterns('NVDA')
        
        if signals:
            print(f"✓ Found {len(signals)} patterns:")
            for signal in signals:
                print(f"  - {signal['pattern']}: {signal['action']} at ${signal['entry_price']}")
                print(f"    Risk/Reward: 1:{signal['risk_reward']}")
            
            # Test saving to Azure
            print("\nSaving signals to Azure...")
            tracker.save_signals_to_azure(signals, 'NVDA')
            print("✓ Successfully saved signals to Azure")
        else:
            print("No patterns found (this is normal)")
        
        # Test evaluation
        print("\nEvaluating past signals...")
        stats = tracker.evaluate_signals('NVDA', hours=24)
        
        if stats:
            print(f"✓ Evaluation complete:")
            print(f"  - Total signals: {stats['total_signals']}")
            print(f"  - Win rate: {stats['win_rate']}%")
        else:
            print("No signals to evaluate yet")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Created config/.env with Azure credentials")
        print("2. Set up the Azure container")
        print("3. Installed required packages")

if __name__ == "__main__":
    test_tracker()