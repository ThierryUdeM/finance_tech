#!/usr/bin/env python3
"""
Test Databento - check available schemas and datasets
"""

import databento as db
from datetime import datetime, timedelta

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def check_available_data():
    """Check what's actually available in Databento"""
    
    client = db.Historical(key=API_KEY)
    
    print("Checking Databento available data...\n")
    
    # Test with known liquid stocks that should be available
    test_symbols = [
        # Canadian on TSX
        ("AC", "Air Canada"),
        ("TD", "TD Bank"),
        ("RY", "Royal Bank"),
        ("BCE", "BCE Inc"),
        
        # Try with exchange suffixes
        ("AC.XTSE", "Air Canada TSX"),
        ("AC:TSE", "Air Canada TSX format 2"),
        
        # US listings that might work
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
    ]
    
    # Different schemas to try
    schemas_to_try = [
        "trades",      # Raw trades
        "tbbo",        # Top of book quotes
        "ohlcv-1d",    # Daily bars
        "ohlcv-1h",    # Hourly bars
        "ohlcv-15m",   # 15-minute bars
        "mbp-1",       # Market by price
    ]
    
    # Test date range (just 1 week for testing)
    end_date = datetime(2024, 1, 15)  # Use a fixed recent date
    start_date = end_date - timedelta(days=7)
    
    # Known working datasets based on Databento docs
    datasets = [
        "GLBX.MDP3",    # CME Globex
        "XNAS.ITCH",    # Nasdaq
        "XNYS.TRADES",  # NYSE trades
        "OPRA.TRADES",  # Options
        "DBEQ.BASIC",   # Databento Equities Basic
        "DBEQ.PLUS",    # Databento Equities Plus  
        "DBEQ.MAX",     # Databento Equities Max
    ]
    
    found_combinations = []
    
    # First, let's check TSX specifically
    print("Checking for TSX/Canadian exchange data...\n")
    
    # TSX might be under these datasets
    tsx_datasets = [
        "XTSE.TRADES",   # TSX trades
        "XTSE",          # TSX generic
        "TSE",           # Alternative
        "TSX",           # Alternative
        "DBEQ.BASIC",    # Might include Canadian
    ]
    
    for dataset in tsx_datasets:
        try:
            print(f"Testing dataset: {dataset}")
            # Just try to query - if dataset exists, we'll get a different error
            data = client.timeseries.get_range(
                dataset=dataset,
                symbols=["AC"],
                schema="trades",
                start=start_date,
                end=end_date,
                limit=1
            )
            print(f"  ✓ Dataset {dataset} exists!")
        except Exception as e:
            error_msg = str(e)
            if "Invalid `dataset`" in error_msg:
                print(f"  ✗ Dataset {dataset} not found")
            else:
                print(f"  ? Dataset {dataset} exists but error: {error_msg[:60]}...")
    
    print("\n" + "="*60 + "\n")
    
    # Now test standard datasets with different symbols
    for dataset in datasets[:3]:  # Test first 3 to save API calls
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        
        for symbol, name in test_symbols[:4]:  # Test first 4 symbols
            for schema in schemas_to_try[:3]:  # Test first 3 schemas
                try:
                    data = client.timeseries.get_range(
                        dataset=dataset,
                        symbols=[symbol],
                        schema=schema,
                        start=start_date,
                        end=end_date,
                        limit=1
                    )
                    
                    df = data.to_df()
                    
                    if not df.empty:
                        print(f"  ✓ Found: {symbol} ({name}) - {schema}")
                        found_combinations.append((dataset, symbol, schema))
                        break  # Found this symbol, skip other schemas
                        
                except Exception as e:
                    # Only print if it's not a "not found" error
                    error_msg = str(e)
                    if "data_unavailable" not in error_msg and "not_found" not in error_msg:
                        if "data_schema_not_fully_available" in error_msg:
                            pass  # Schema not available, normal
                        else:
                            print(f"  ! {symbol} - {schema}: {error_msg[:50]}...")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if found_combinations:
        print("\nWorking combinations found:")
        for dataset, symbol, schema in found_combinations:
            print(f"  - {dataset}: {symbol} ({schema})")
    else:
        print("\nNo working combinations found in quick test.")
    
    print("\nFor Canadian equities (AC.TO), you likely need to:")
    print("1. Check Databento's data coverage documentation")
    print("2. Contact support to confirm TSX availability")
    print("3. Find the correct dataset name for TSX")
    print("4. Or use alternative data sources for Canadian stocks")
    
    # Alternative: Check if we can at least get US airline stocks
    print("\n\nAlternative: Checking US airline stocks...")
    us_airlines = ["AAL", "DAL", "UAL", "LUV", "ALK"]
    
    for symbol in us_airlines:
        try:
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[symbol],
                schema="trades",
                start=start_date,
                end=end_date,
                limit=1
            )
            df = data.to_df()
            if not df.empty:
                print(f"  ✓ Found US airline: {symbol}")
                break
        except:
            pass

if __name__ == "__main__":
    check_available_data()