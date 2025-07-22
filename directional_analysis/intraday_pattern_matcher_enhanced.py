#!/usr/bin/env python3
"""
Enhanced pattern matcher that uses local Databento data when available
Falls back to Yahoo Finance for other tickers
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import yfinance as yf
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# Check for local data files
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_DATA_FILES = {
    'NVDA': os.path.join(SCRIPT_DIR, 'NVDA_15min_pattern_ready.csv'),
    # Add more as you download them
}

def load_local_or_yahoo(ticker, interval="15m", period="60d"):
    """
    Load data from local file if available, otherwise use Yahoo Finance
    """
    if ticker in LOCAL_DATA_FILES and os.path.exists(LOCAL_DATA_FILES[ticker]):
        print(f"Using local Databento data for {ticker}")
        df = pd.read_csv(LOCAL_DATA_FILES[ticker], index_col=0, parse_dates=True)
        
        # Ensure column names match expected format
        df.columns = [col.lower() for col in df.columns]
        
        print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        return df, "local"
    else:
        print(f"Using Yahoo Finance data for {ticker} (limited to {period})")
        return fetch_intraday_yahoo(ticker, interval, period), "yahoo"

def fetch_intraday_yahoo(ticker, interval="15m", period="60d"):
    """
    Fetch intraday data from Yahoo Finance
    """
    # For 15-minute bars, Yahoo limits to 60 days
    # For smaller intervals, even less
    if interval == "15m" and period == "60d":
        # Use a safer period that works
        actual_period = "30d"
        print(f"  Note: Using {actual_period} for 15m bars due to Yahoo limits")
    else:
        actual_period = period
        
    # Use download instead of history for better consistency
    df = yf.download(ticker, 
                     period=actual_period, 
                     interval=interval, 
                     progress=False,
                     prepost=False)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column names
    df.columns = [col.lower() for col in df.columns]
    
    # Keep only market hours
    df = df.between_time("09:30", "16:00")
    
    return df

def build_pattern_library(df, query_length, data_source="yahoo", interval="15m"):
    """
    Build library of historical patterns
    """
    query_length = int(query_length)
    
    # Calculate bars per hour based on interval
    if interval == "5m":
        bars_per_hour = 12
    elif interval == "15m":
        bars_per_hour = 4
    elif interval == "30m":
        bars_per_hour = 2
    elif interval == "1h":
        bars_per_hour = 1
    else:
        bars_per_hour = 4  # Default to 15m
        
    bars_per_3h = bars_per_hour * 3
    
    lib = []
    
    # For local data, we can use more patterns
    if data_source == "local":
        step_size = 1  # Use every possible pattern
        min_patterns_per_day = 10
    else:
        step_size = 2  # Skip every other pattern to reduce correlation
        min_patterns_per_day = 5
    
    # Group by date
    for day, day_df in df.groupby(df.index.date):
        prices = day_df["close"].values
        n_prices = len(prices)
        
        # For Yahoo data, be more flexible with requirements
        if data_source == "yahoo":
            min_required = query_length + bars_per_hour  # At least 1 hour lookahead
        else:
            min_required = query_length + bars_per_3h  # Full 3 hour lookahead
            
        if n_prices < min_required:
            continue
        
        # Extract patterns with step size
        max_lookahead = bars_per_3h if n_prices >= query_length + bars_per_3h else bars_per_hour
        max_start = n_prices - query_length - max_lookahead
        
        if max_start <= 0:
            continue
            
        for start_idx in range(0, max_start, step_size):
            window = prices[start_idx:start_idx + query_length]
            
            if len(window) != query_length:
                continue
            
            # Normalize
            if window.std() > 0:
                normed = (window - window.mean()) / window.std()
            else:
                continue  # Skip flat patterns
            
            base_price = window[-1]
            out = {}
            
            # Calculate future returns
            end_idx = start_idx + query_length
            
            # 1h ahead
            if end_idx + bars_per_hour <= n_prices:
                p1 = prices[end_idx + bars_per_hour - 1]
                out["1h"] = (p1 / base_price) - 1
            
            # 3h ahead
            if end_idx + bars_per_3h <= n_prices:
                p3 = prices[end_idx + bars_per_3h - 1]
                out["3h"] = (p3 / base_price) - 1
            
            # EOD
            p_eod = prices[-1]
            out["eod"] = (p_eod / base_price) - 1
            
            if out:
                lib.append((normed, out))
    
    return lib

def predict_returns(current_prices, library, K=10):
    """
    Predict returns using K nearest patterns
    """
    if len(library) == 0:
        return {"1h": np.nan, "3h": np.nan, "eod": np.nan}
    
    curr = np.array(current_prices)
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
    
    # Get K nearest neighbors
    k_actual = min(K, len(library))
    nearest_indices = np.argsort(distances)[:k_actual]
    
    # Get distance weights (closer = higher weight)
    nearest_distances = distances[nearest_indices]
    weights = 1 / (1 + nearest_distances)  # Inverse distance weighting
    weights = weights / weights.sum()  # Normalize
    
    # Aggregate predictions with weights
    weighted_preds = {"1h": 0, "3h": 0, "eod": 0}
    counts = {"1h": 0, "3h": 0, "eod": 0}
    
    for idx, weight in zip(nearest_indices, weights):
        _, returns_dict = library[idx]
        for horizon in weighted_preds:
            if horizon in returns_dict:
                weighted_preds[horizon] += returns_dict[horizon] * weight
                counts[horizon] += weight
    
    # Calculate final predictions
    result = {}
    for horizon in ["1h", "3h", "eod"]:
        if counts[horizon] > 0:
            result[horizon] = float(weighted_preds[horizon])
        else:
            result[horizon] = np.nan
    
    return result

def forecast_shape(ticker, interval="15m", period="60d", query_length=20, K=10):
    """
    Main function to forecast based on pattern matching
    """
    # Ensure integer types
    query_length = int(query_length)
    K = int(K)
    
    # Load data
    hist, data_source = load_local_or_yahoo(ticker, interval, period)
    
    print(f"Total bars available: {len(hist)}")
    
    if len(hist) < query_length * 2:
        raise RuntimeError(f"Not enough data: have {len(hist)} bars, need at least {query_length * 2}")
    
    # Build library (exclude most recent query_length bars)
    library_data = hist.iloc[:-query_length]
    print(f"Building pattern library from {len(library_data)} historical bars")
    
    library = build_pattern_library(library_data, query_length, data_source, interval)
    print(f"Built library with {len(library)} patterns")
    
    if data_source == "local":
        print(f"Pattern density: {len(library)/len(library_data)*100:.1f}% of possible patterns")
    
    if not library:
        raise RuntimeError("Could not build pattern library")
    
    # Use most recent bars for prediction
    current_window = hist["close"].values[-query_length:]
    print(f"Using most recent {len(current_window)} bars for prediction")
    
    # Make prediction
    preds = predict_returns(current_window, library, K=K)
    
    # Add confidence based on library size
    if len(library) > 10000:
        confidence = "HIGH"
    elif len(library) > 1000:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    preds['confidence'] = confidence
    preds['library_size'] = len(library)
    preds['data_source'] = data_source
    
    return preds

# Module is ready for import - no test code when imported