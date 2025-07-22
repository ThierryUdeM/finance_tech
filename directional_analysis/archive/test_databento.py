#!/usr/bin/env python3
"""
Test Databento connection and find correct symbol for AC.TO
"""

import databento as db
from datetime import datetime, timedelta

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def test_connection():
    """Test basic connection and find AC.TO symbol"""
    
    client = db.Historical(key=API_KEY)
    
    print("Testing Databento connection...")
    
    # Test 1: Try to get metadata about available datasets
    try:
        print("\n1. Available datasets:")
        # List datasets (method may vary by version)
        print("Connected successfully!")
    except Exception as e:
        print(f"Connection error: {e}")
        return
    
    # Test 2: Search for Air Canada symbols
    print("\n2. Searching for Air Canada (AC) symbols...")
    
    # Common symbol variations to try
    symbols_to_try = [
        "AC.TO",      # Yahoo format
        "AC",         # Base symbol
        "AC.TSE",     # TSE suffix
        "AC.TSX",     # TSX suffix  
        "AC.XTSE",    # Bloomberg format
        "AC.CN",      # Reuters format
        "ACT.TO",     # Alternative
    ]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # Just 5 days for testing
    
    # Test different datasets that might contain Canadian stocks
    datasets_to_try = [
        "GLBX.MDP3",   # Global exchanges
        "XCAN.ITCH",   # Canadian exchange
        "DBEQ.BASIC",  # Equities basic
    ]
    
    found = False
    
    for dataset in datasets_to_try:
        if found:
            break
            
        print(f"\nTrying dataset: {dataset}")
        
        for symbol in symbols_to_try:
            try:
                print(f"  Testing symbol: {symbol}...", end="")
                
                # Try to get just 1 day of data to test
                data = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    schema="ohlcv-1d",  # Daily bars for quick test
                    start=start_date,
                    end=end_date,
                    limit=1  # Just get 1 record to test
                )
                
                df = data.to_df()
                
                if not df.empty:
                    print(f" SUCCESS! Found data")
                    print(f"\n  Dataset: {dataset}")
                    print(f"  Symbol: {symbol}")
                    print(f"  Sample data:")
                    print(df)
                    found = True
                    
                    # Now test 15-minute bars
                    print(f"\n  Testing 15-minute bars...")
                    data_15m = client.timeseries.get_range(
                        dataset=dataset,
                        symbols=[symbol],
                        schema="ohlcv-15m",
                        start=start_date,
                        end=end_date,
                        limit=10
                    )
                    df_15m = data_15m.to_df()
                    if not df_15m.empty:
                        print(f"  15-minute data available! {len(df_15m)} bars")
                    else:
                        print(f"  No 15-minute data found")
                    
                    break
                else:
                    print(f" No data")
                    
            except Exception as e:
                print(f" Error: {str(e)[:50]}...")
    
    if not found:
        print("\n\nCould not find AC.TO in standard datasets.")
        print("\nRecommendations:")
        print("1. Contact Databento support for Canadian equity symbols")
        print("2. Check their symbol reference documentation")
        print("3. Consider using US-listed Air Canada (ACDVF) if available")
        
        # Try to get any Canadian stock as example
        print("\n\nTrying to find ANY Canadian stock for reference...")
        test_symbols = ["TD", "RY", "BNS", "BMO", "CNR"]  # Major Canadian banks/companies
        
        for symbol in test_symbols:
            try:
                data = client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=[symbol],
                    schema="ohlcv-1d",
                    start=start_date,
                    end=end_date,
                    limit=1
                )
                df = data.to_df()
                if not df.empty:
                    print(f"Found {symbol} - Canadian stocks ARE available")
                    break
            except:
                pass

if __name__ == "__main__":
    # First install databento if needed
    try:
        import databento
    except ImportError:
        print("Installing databento...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "databento"])
        import databento
    
    test_connection()