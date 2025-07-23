#!/usr/bin/env python3
"""
Check if Databento has TSX data
"""

import databento as db
from datetime import datetime, timedelta

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def check_tsx_availability():
    """Check if Databento has TSX data"""
    
    client = db.Historical(key=API_KEY)
    
    print("Checking for TSX (Toronto Stock Exchange) data in Databento...")
    print("=" * 60)
    
    # Common TSX dataset names to try
    tsx_datasets = [
        "XTSE",           # TSX standard code
        "TSE",            # Alternative
        "XTSE.ITCH",      # TSX ITCH feed
        "XTSE.TRADES",    # TSX trades
        "XTSE.MBO",       # TSX market by order
        "XCAN",           # Canada generic
        "XCAN.ITCH",      # Canada ITCH
        "CA",             # Canada short
        "TSX",            # Direct TSX
        "TSXV",           # TSX Venture
        "DBEQ.TSX",       # Databento Equities TSX
        "GLBX.TSX",       # Global Exchange TSX
    ]
    
    # Test date range
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=5)
    
    print("\nTesting TSX dataset availability:\n")
    
    valid_datasets = []
    
    for dataset in tsx_datasets:
        try:
            print(f"Testing: {dataset:<15}", end="")
            
            # Try a simple metadata query first
            # This should fail differently if dataset exists vs not
            data = client.timeseries.get_range(
                dataset=dataset,
                symbols=["AC.TO", "AC", "AC.TSE"],  # Try different symbol formats
                schema="trades",
                start=start_date,
                end=end_date,
                limit=1
            )
            
            print(" ✓ Dataset exists!")
            valid_datasets.append(dataset)
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid `dataset`" in error_msg or "400 validation_failed" in error_msg:
                print(" ✗ Not found")
            elif "422" in error_msg:
                print(" ? Exists but error")
                valid_datasets.append(f"{dataset} (exists but access issue)")
            else:
                print(f" ! Error: {error_msg[:30]}...")
    
    print("\n" + "=" * 60)
    
    # Try to list all available datasets
    print("\nTrying to get dataset information...")
    
    try:
        # Get account info which might show available datasets
        print("\nChecking account/dataset access...")
        
        # Try common TSX symbols with most likely datasets
        test_symbols = [
            ("AC.TO", "Air Canada"),
            ("AC", "Air Canada alt"),
            ("TD.TO", "TD Bank"),
            ("TD", "TD Bank alt"),
            ("RY.TO", "Royal Bank"),
            ("BCE.TO", "BCE Inc"),
        ]
        
        # Focus on most likely datasets
        likely_datasets = ["DBEQ.BASIC", "DBEQ.PLUS", "DBEQ.MAX"]
        
        print("\nTesting TSX symbols on Databento Equities datasets:")
        
        for dataset in likely_datasets:
            print(f"\nDataset: {dataset}")
            print("-" * 40)
            
            for symbol, name in test_symbols:
                try:
                    print(f"  {symbol:<8} ({name})...", end="")
                    
                    data = client.timeseries.get_range(
                        dataset=dataset,
                        symbols=[symbol],
                        schema="ohlcv-1d",  # Try daily bars first
                        start=start_date,
                        end=end_date,
                        limit=1
                    )
                    
                    df = data.to_df()
                    
                    if not df.empty:
                        print(f" ✓ Found! Price: ${df.iloc[0]['close']:.2f}")
                        
                        # Now try 1-minute bars (15m is not supported, need to resample)
                        try:
                            data_1m = client.timeseries.get_range(
                                dataset=dataset,
                                symbols=[symbol],
                                schema="ohlcv-1m",
                                start=start_date,
                                end=end_date,
                                limit=5
                            )
                            df_1m = data_1m.to_df()
                            if not df_1m.empty:
                                print(f"       └─ 1-min bars available (can resample to 15m)!")
                        except:
                            print(f"       └─ No 1-min bars")
                            
                        break  # Found this symbol
                    else:
                        print(" No data")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "data_end_after_available" in error_msg:
                        print(" (dataset ends before date)")
                    elif "symbology_invalid" in error_msg:
                        print(" (symbol not found)")
                    else:
                        print(f" Error: {error_msg[:30]}...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if valid_datasets:
        print(f"\nPotential TSX datasets found: {valid_datasets}")
    else:
        print("\nNo obvious TSX datasets found.")
    
    print("\nRecommendations:")
    print("1. Contact Databento support directly about TSX data")
    print("2. Their documentation should list available exchanges")
    print("3. TSX data might require additional subscription")
    print("4. Check https://databento.com/docs/datasets")
    
    print("\nNote: Major exchanges often require separate licensing:")
    print("- NYSE, NASDAQ: Usually included in US equities")
    print("- TSX, LSE, etc: Often require additional fees")

if __name__ == "__main__":
    check_tsx_availability()