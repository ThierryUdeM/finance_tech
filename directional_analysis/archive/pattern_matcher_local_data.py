#!/usr/bin/env python3
"""
Pattern matcher that can use local CSV data or Yahoo Finance
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import yfinance as yf
from sklearn.metrics.pairwise import euclidean_distances

def load_local_data(csv_file):
    """Load 15-minute OHLCV data from local CSV"""
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have columns: {required_cols}")
    
    # Filter to market hours if not already done
    if hasattr(df.index, 'time'):
        df = df.between_time('09:30', '16:00')
    
    return df

def check_data_quality(df, min_bars_per_day=20, min_days=100):
    """Check if data has enough liquidity for pattern matching"""
    
    # Group by date
    daily_bars = df.groupby(df.index.date).size()
    
    # Calculate statistics
    total_days = len(daily_bars)
    avg_bars_per_day = daily_bars.mean()
    days_with_enough_bars = (daily_bars >= min_bars_per_day).sum()
    
    quality_score = days_with_enough_bars / total_days if total_days > 0 else 0
    
    print(f"\nData Quality Check:")
    print(f"  Total days: {total_days}")
    print(f"  Average bars per day: {avg_bars_per_day:.1f}")
    print(f"  Days with >={min_bars_per_day} bars: {days_with_enough_bars} ({quality_score*100:.1f}%)")
    
    if quality_score < 0.5:
        print("\n⚠️  WARNING: Data quality is poor!")
        print("  This ticker appears to be thinly traded.")
        print("  Pattern matching may not be reliable.\n")
        return False
    
    return True

def build_library_from_local(df, query_length):
    """Build pattern library from local data"""
    query_length = int(query_length)
    
    # For 15-minute bars
    bars_per_hour = 4
    bars_per_3h = 12
    
    lib = []
    
    # Group by date
    for day, day_df in df.groupby(df.index.date):
        prices = day_df["close"].values
        n_prices = len(prices)
        
        if n_prices < query_length + bars_per_3h:
            continue
        
        # Extract multiple windows per day
        for start_idx in range(0, n_prices - query_length - bars_per_3h):
            window = prices[start_idx:start_idx + query_length]
            
            if len(window) != query_length:
                continue
            
            # Normalize
            if window.std() > 0:
                normed = (window - window.mean()) / window.std()
            else:
                normed = window - window.mean()
            
            base_price = window[-1]
            out = {}
            
            # Future returns
            end_idx = start_idx + query_length
            
            # 1h ahead (4 bars)
            if end_idx + bars_per_hour <= n_prices:
                p1 = prices[end_idx + bars_per_hour - 1]
                out["1h"] = (p1 / base_price) - 1
            
            # 3h ahead (12 bars)
            if end_idx + bars_per_3h <= n_prices:
                p3 = prices[end_idx + bars_per_3h - 1]
                out["3h"] = (p3 / base_price) - 1
            
            # EOD
            p_eod = prices[-1]
            out["eod"] = (p_eod / base_price) - 1
            
            if out:
                lib.append((normed, out))
    
    return lib

def analyze_ticker_with_local_data(ticker, csv_file=None, interval="15m", 
                                  query_length=8, K=5):
    """
    Analyze ticker using local CSV data if available, otherwise Yahoo Finance
    """
    
    if csv_file and os.path.exists(csv_file):
        print(f"\nUsing local data from: {csv_file}")
        
        # Load local data
        df = load_local_data(csv_file)
        print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
        # Check quality
        is_quality = check_data_quality(df)
        
        if not is_quality:
            response = input("\nContinue with poor quality data? (y/n): ")
            if response.lower() != 'y':
                print("\nSuggestions:")
                print("1. Use a more liquid ticker")
                print("2. Download data from the primary exchange (TSX)")
                print("3. Use daily bars instead of intraday")
                return None
        
        # Build library
        print(f"\nBuilding pattern library...")
        library = build_library_from_local(df[:-query_length], query_length)
        print(f"Built library with {len(library)} patterns")
        
        if len(library) < 50:
            print("⚠️  WARNING: Very small pattern library!")
        
        # Get current pattern
        current_window = df["close"].values[-query_length:]
        
    else:
        print(f"\nNo local data found. Using Yahoo Finance for {ticker}")
        print("Note: Yahoo Finance limits intraday data to 60 days")
        
        # Fall back to original Yahoo Finance approach
        # Import the original functions
        from intraday_shape_matcher import fetch_intraday, build_library, predict_returns
        
        hist = fetch_intraday(ticker, interval, "60d")
        library = build_library(hist[:-query_length], query_length)
        current_window = hist["close"].values[-query_length:]
    
    # Make predictions
    if len(library) > 0:
        curr = np.array(current_window)
        if curr.std() > 0:
            curr_norm = (curr - curr.mean()) / curr.std()
        else:
            curr_norm = curr - curr.mean()
        
        # Calculate distances
        distances = []
        for hist_pattern, _ in library:
            dist = np.sqrt(np.sum((curr_norm - hist_pattern) ** 2))
            distances.append(dist)
        
        distances = np.array(distances)
        k_actual = min(K, len(library))
        nearest_indices = np.argsort(distances)[:k_actual]
        
        # Aggregate predictions
        agg = {"1h": [], "3h": [], "eod": []}
        for idx in nearest_indices:
            _, returns_dict = library[idx]
            for horizon in agg:
                if horizon in returns_dict:
                    agg[horizon].append(returns_dict[horizon])
        
        # Calculate mean predictions
        result = {}
        for horizon in ["1h", "3h", "eod"]:
            if agg[horizon]:
                result[horizon] = float(np.mean(agg[horizon]))
            else:
                result[horizon] = np.nan
        
        return result
    
    return None

def suggest_alternatives():
    """Suggest alternative approaches for Canadian stocks"""
    
    print("\n" + "="*60)
    print("ALTERNATIVE APPROACHES FOR CANADIAN STOCKS")
    print("="*60)
    
    print("\n1. Use more liquid Canadian tickers:")
    print("   - TD, RY, BNS, BMO (Big banks)")
    print("   - CNR, CP (Railways)")
    print("   - SU, CNQ (Energy)")
    
    print("\n2. Use US-listed Canadian stocks:")
    print("   - Many have better liquidity on NYSE/NASDAQ")
    
    print("\n3. Alternative data sources for TSX:")
    print("   - Polygon.io: ~$300/month for TSX data")
    print("   - Interactive Brokers API: Need account")
    print("   - Questrade API: Canadian broker")
    print("   - TMX Datalinx: Official TSX data")
    
    print("\n4. Modified approach:")
    print("   - Use daily bars (years of history)")
    print("   - Combine with technical indicators")
    print("   - Focus on sector/market regime patterns")

if __name__ == "__main__":
    # Example usage
    print("Pattern Matcher with Local Data Support")
    print("="*40)
    
    # Check if we have AC data
    if os.path.exists("AC_15min_bars.csv"):
        result = analyze_ticker_with_local_data(
            "AC",
            csv_file="AC_15min_bars.csv",
            query_length=8,  # 2 hours of 15-min bars
            K=5
        )
        
        if result:
            print("\nPredictions:")
            print(f"  1-hour return: {result['1h']*100:+.2f}%")
            print(f"  3-hour return: {result['3h']*100:+.2f}%")
            print(f"  EOD return: {result['eod']*100:+.2f}%")
    
    suggest_alternatives()