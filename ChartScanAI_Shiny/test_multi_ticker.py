#!/usr/bin/env python3
"""
Test script for multi-ticker predictor
Tests basic functionality without running full predictions
"""

import os
import sys
import yfinance as yf
from datetime import datetime

# Configuration
TICKERS = ['BTC-USD', 'NVDA', 'AC.TO']

def test_ticker_access():
    """Test if we can access data for all tickers"""
    print("Testing ticker data access...")
    print("-" * 50)
    
    for ticker in TICKERS:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            
            if price:
                print(f"✅ {ticker}: Current price = ${price:.2f}")
            else:
                print(f"❌ {ticker}: Could not get current price")
                
            # Test data fetch for 1d interval
            data = stock.history(period='5d', interval='1d')
            if not data.empty:
                print(f"   Data points available: {len(data)}")
            else:
                print(f"   WARNING: No historical data available")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {str(e)}")
            
    print()

def test_intervals():
    """Test data availability for different intervals"""
    print("Testing interval data availability...")
    print("-" * 50)
    
    intervals = ['15m', '1h', '4h', '1d']
    period_map = {
        '15m': '5d',
        '1h': '2wk',
        '4h': '3mo',
        '1d': '6mo'
    }
    
    for ticker in TICKERS[:1]:  # Test just BTC for brevity
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        
        for interval in intervals:
            try:
                period = period_map[interval]
                data = stock.history(period=period, interval=interval)
                
                if not data.empty:
                    print(f"  {interval}: {len(data)} data points")
                else:
                    print(f"  {interval}: No data available")
                    
            except Exception as e:
                print(f"  {interval}: Error - {str(e)}")

def test_azure_env():
    """Test if Azure environment variables are set"""
    print("\nTesting Azure configuration...")
    print("-" * 50)
    
    required_vars = ['STORAGE_ACCOUNT_NAME', 'ACCESS_KEY', 'CONTAINER_NAME']
    
    # Check if .env file exists
    env_path = 'config/.env'
    if os.path.exists(env_path):
        print(f"✅ Environment file found: {env_path}")
        from dotenv import load_dotenv
        load_dotenv(env_path)
    else:
        print(f"❌ Environment file not found: {env_path}")
        print("   You need to create this file with your Azure credentials")
        return False
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Set (length: {len(value)})")
        else:
            print(f"❌ {var}: Not set")
            all_set = False
            
    return all_set

def test_model_path():
    """Test if YOLO model exists"""
    print("\nTesting YOLO model path...")
    print("-" * 50)
    
    model_path = '../ChartScanAI/weights/custom_yolov8.pt'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        print("   Make sure the model weights are in the correct location")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("Multi-Ticker Predictor Test Suite")
    print("="*50 + "\n")
    
    # Test ticker access
    test_ticker_access()
    
    # Test intervals
    test_intervals()
    
    # Test Azure config
    azure_ok = test_azure_env()
    
    # Test model
    model_ok = test_model_path()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    if azure_ok and model_ok:
        print("✅ All critical components are ready!")
        print("\nYou can now run:")
        print("  python multi_ticker_predictor_azure.py")
    else:
        print("❌ Some components need configuration")
        print("\nPlease fix the issues above before running predictions")

if __name__ == "__main__":
    main()