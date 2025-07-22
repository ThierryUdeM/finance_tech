# intraday_shape_matcher_v2.py

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Use sklearn's euclidean distance as a simpler alternative to DTW for now
from sklearn.metrics.pairwise import euclidean_distances

def fetch_intraday(ticker, interval="10m", period="60d"):
    """
    Fetches intraday bars for the given ticker.
    """
    df = yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        prepost=False
    )
    if df.empty:
        raise ValueError(f"No intraday data for {ticker}")
    # keep only regular-market hours (09:30–16:00)
    return df.between_time("09:30", "16:00")


def build_library(df, query_length):
    query_length = int(query_length)
    """
    Builds a library of (normalized_window, future_1h, future_3h, future_eod).
    """
    # Calculate bars per hour based on interval
    if len(df) < 2:
        raise ValueError("Not enough data to infer interval")
    
    delta = df.index[1] - df.index[0]
    delta_minutes = delta.total_seconds() / 60
    bars_per_hour = max(1, int(round(60 / delta_minutes)))
    bars_per_3h = bars_per_hour * 3

    lib = []
    # Group by date
    grouped = df.groupby(df.index.date)
    
    for day, day_df in grouped:
        prices = day_df["Close"].values
        n_prices = len(prices)
        
        # Need at least query_length bars plus some for future returns
        if n_prices < query_length + 1:
            continue

        # For each possible window in this day
        for start_idx in range(n_prices - query_length):
            end_idx = start_idx + query_length
            
            # Extract window
            window = prices[start_idx:end_idx]
            if len(window) != query_length:
                continue
                
            # Normalize
            if window.std() > 0:
                normed = (window - window.mean()) / window.std()
            else:
                normed = window - window.mean()

            base_price = window[-1]
            out = {}

            # Calculate future returns if data available
            # 1h ahead
            future_1h_idx = end_idx + bars_per_hour - 1
            if future_1h_idx < n_prices:
                p1 = prices[future_1h_idx]
                out["1h"] = (p1 / base_price) - 1

            # 3h ahead
            future_3h_idx = end_idx + bars_per_3h - 1
            if future_3h_idx < n_prices:
                p3 = prices[future_3h_idx]
                out["3h"] = (p3 / base_price) - 1

            # EOD (last price of the day)
            p_eod = prices[-1]
            out["eod"] = (p_eod / base_price) - 1

            # Only add if we have at least some future returns
            if out:
                lib.append((normed, out))

    return lib


def predict_returns(current_prices, library, K=10):
    """
    Given current prices, find K nearest patterns and predict returns.
    """
    if len(library) == 0:
        return {"1h": np.nan, "3h": np.nan, "eod": np.nan}
    
    curr = np.array(current_prices)
    if curr.std() > 0:
        curr_norm = (curr - curr.mean()) / curr.std()
    else:
        curr_norm = curr - curr.mean()

    # Calculate distances to all patterns in library
    distances = []
    for hist_pattern, _ in library:
        # Simple Euclidean distance
        dist = np.sqrt(np.sum((curr_norm - hist_pattern) ** 2))
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Get K nearest neighbors
    k_actual = min(K, len(library))
    nearest_indices = np.argsort(distances)[:k_actual]

    # Aggregate predictions
    agg = {"1h": [], "3h": [], "eod": []}
    for idx in nearest_indices:
        _, returns_dict = library[idx]
        for horizon in agg:
            if horizon in returns_dict:
                agg[horizon].append(returns_dict[horizon])

    # Return mean of predictions
    result = {}
    for horizon in ["1h", "3h", "eod"]:
        if agg[horizon]:
            result[horizon] = float(np.mean(agg[horizon]))
        else:
            result[horizon] = np.nan
    
    return result


def forecast_shape(ticker, interval="10m", period="60d", query_length=30, K=10):
    # Ensure integer types
    query_length = int(query_length)
    K = int(K)
    """
    Main function to forecast based on shape matching.
    """
    print(f"Fetching {ticker} data: interval={interval}, period={period}")
    
    # Fetch all available data
    hist = fetch_intraday(ticker, interval, period)
    print(f"Fetched {len(hist)} total bars")
    
    if len(hist) < query_length * 2:
        raise RuntimeError(f"Not enough data: have {len(hist)} bars, need at least {query_length * 2}")
    
    # Build library from historical data (exclude most recent query_length bars)
    library_data = hist.iloc[:-query_length]
    print(f"Building library from {len(library_data)} historical bars")
    
    library = build_library(library_data, query_length)
    print(f"Built library with {len(library)} patterns")
    
    if not library:
        raise RuntimeError("Could not build pattern library")
    
    # Use the most recent query_length bars for prediction
    current_window = hist["Close"].values[-query_length:]
    print(f"Using most recent {len(current_window)} bars for prediction")
    
    # Make prediction
    preds = predict_returns(current_window, library, K=K)
    
    return preds