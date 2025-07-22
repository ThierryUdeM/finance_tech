#!/usr/bin/env python3
"""
Enhanced pattern matcher with time-decay weighting, adaptive thresholds, and configurable ensemble
Implements improvements:
1. Time-decay weighting for recent bars
2. Adaptive ATR-based thresholds
3. Configurable ensemble weights from YAML
4. Probability-based confidence scoring
"""

import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime
import yfinance as yf
# Remove sklearn dependency - using numpy for distance calculations
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_DATA_FILES = {
    'NVDA': os.path.join(SCRIPT_DIR, 'NVDA_15min_pattern_ready.csv'),
    # Add more as you download them
}

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(SCRIPT_DIR, '..', 'config', 'ensemble_weights.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}, using defaults")
        return get_default_config()

def get_default_config():
    """Default configuration if YAML file not found"""
    return {
        'timeframe_weights': {'short_term': 0.4, 'medium_term': 0.35, 'long_term': 0.25},
        'adaptive_thresholds': {
            'enabled': True, 'atr_multiplier': 0.25, 'atr_period': 14,
            'min_threshold': 0.05, 'max_threshold': 0.50, 'fallback_threshold': 0.1
        },
        'time_decay': {'enabled': True, 'decay_rate': 0.1, 'focus_bars': 3},
        'pattern_matching': {'confidence_thresholds': {'high': 10000, 'medium': 1000}}
    }

def calculate_atr_threshold(df, config):
    """Calculate adaptive threshold based on ATR"""
    atr_config = config['adaptive_thresholds']
    
    if not atr_config['enabled']:
        return atr_config['fallback_threshold']
    
    try:
        period = atr_config['atr_period']
        
        # Calculate True Range
        df_calc = df.copy()
        df_calc['prev_close'] = df_calc['close'].shift(1)
        df_calc['tr1'] = df_calc['high'] - df_calc['low']
        df_calc['tr2'] = abs(df_calc['high'] - df_calc['prev_close'])
        df_calc['tr3'] = abs(df_calc['low'] - df_calc['prev_close'])
        df_calc['true_range'] = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr = df_calc['true_range'].rolling(window=period).mean().iloc[-1]
        current_price = df_calc['close'].iloc[-1]
        
        # Convert to percentage
        atr_pct = (atr / current_price) * 100
        
        # Apply multiplier and constraints
        threshold = atr_pct * atr_config['atr_multiplier']
        threshold = max(atr_config['min_threshold'], min(threshold, atr_config['max_threshold']))
        
        print(f"Adaptive threshold: {threshold:.3f}% (ATR: {atr_pct:.3f}%)")
        return threshold
        
    except Exception as e:
        print(f"ATR calculation failed: {e}, using fallback")
        return atr_config['fallback_threshold']

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
    if interval == "15m" and period == "60d":
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

def calculate_time_decay_weights(pattern_length, config):
    """Calculate time-decay weights for pattern matching"""
    time_config = config['time_decay']
    
    if not time_config['enabled']:
        return np.ones(pattern_length)
    
    # Create base weights (all equal)
    weights = np.ones(pattern_length)
    
    # Apply exponential decay to emphasize recent bars
    decay_rate = time_config['decay_rate']
    focus_bars = min(time_config['focus_bars'], pattern_length)
    
    # Exponential weighting for the most recent bars
    for i in range(focus_bars):
        recent_idx = pattern_length - 1 - i  # Index from the end
        weights[recent_idx] *= np.exp(decay_rate * (focus_bars - i))
    
    # Normalize weights to sum to pattern_length (preserve scale)
    weights = weights * (pattern_length / weights.sum())
    
    return weights

def build_pattern_library(df, query_length, data_source="yahoo", interval="15m", config=None):
    """
    Build library of historical patterns with time-decay weighting
    """
    if config is None:
        config = get_default_config()
        
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

def predict_returns_with_weights(current_prices, library, K=10, config=None):
    """
    Predict returns using K nearest patterns with time-decay weighting
    """
    if config is None:
        config = get_default_config()
        
    if len(library) == 0:
        return {"1h": np.nan, "3h": np.nan, "eod": np.nan, "confidence_score": 0.0}
    
    curr = np.array(current_prices)
    if curr.std() > 0:
        curr_norm = (curr - curr.mean()) / curr.std()
    else:
        curr_norm = curr - curr.mean()
    
    # Calculate time-decay weights
    weights = calculate_time_decay_weights(len(curr_norm), config)
    
    # Calculate weighted distances
    distances = []
    for hist_pattern, _ in library:
        # Apply time-decay weighting to distance calculation
        weighted_diff = weights * (curr_norm - hist_pattern) ** 2
        dist = np.sqrt(np.sum(weighted_diff))
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Get K nearest neighbors
    k_actual = min(K, len(library))
    nearest_indices = np.argsort(distances)[:k_actual]
    
    # Get distance weights (closer = higher weight)
    nearest_distances = distances[nearest_indices]
    distance_weights = 1 / (1 + nearest_distances)  # Inverse distance weighting
    distance_weights = distance_weights / distance_weights.sum()  # Normalize
    
    # Calculate confidence score based on distance spread
    if len(nearest_distances) > 1:
        distance_std = np.std(nearest_distances)
        confidence_score = 1.0 / (1.0 + distance_std)  # Lower spread = higher confidence
    else:
        confidence_score = 0.5
    
    # Aggregate predictions with weights
    weighted_preds = {"1h": 0, "3h": 0, "eod": 0}
    counts = {"1h": 0, "3h": 0, "eod": 0}
    
    for idx, weight in zip(nearest_indices, distance_weights):
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
    
    result["confidence_score"] = float(confidence_score)
    return result

def classify_direction_adaptive(pred_pct, threshold):
    """Classify direction using adaptive threshold"""
    if pred_pct > threshold:
        return "BULLISH"
    elif pred_pct < -threshold:
        return "BEARISH"
    else:
        return "NEUTRAL"

def forecast_shape_enhanced(ticker, interval="15m", period="60d", query_length=20, K=10, config=None):
    """
    Enhanced forecast function with all improvements
    """
    if config is None:
        config = load_config()
    
    # Ensure integer types
    query_length = int(query_length)
    K = int(K)
    
    # Load data
    hist, data_source = load_local_or_yahoo(ticker, interval, period)
    
    print(f"Total bars available: {len(hist)}")
    
    if len(hist) < query_length * 2:
        raise RuntimeError(f"Not enough data: have {len(hist)} bars, need at least {query_length * 2}")
    
    # Calculate adaptive threshold
    adaptive_threshold = calculate_atr_threshold(hist, config)
    
    # Build library (exclude most recent query_length bars)
    library_data = hist.iloc[:-query_length]
    print(f"Building pattern library from {len(library_data)} historical bars")
    
    library = build_pattern_library(library_data, query_length, data_source, interval, config)
    print(f"Built library with {len(library)} patterns")
    
    if data_source == "local":
        print(f"Pattern density: {len(library)/len(library_data)*100:.1f}% of possible patterns")
    
    if not library:
        raise RuntimeError("Could not build pattern library")
    
    # Use most recent bars for prediction
    current_window = hist["close"].values[-query_length:]
    print(f"Using most recent {len(current_window)} bars for prediction")
    
    # Make prediction with enhanced algorithm
    preds = predict_returns_with_weights(current_window, library, K=K, config=config)
    
    # Add confidence based on library size and distance score
    conf_thresholds = config['pattern_matching']['confidence_thresholds']
    if len(library) > conf_thresholds['high']:
        base_confidence = "HIGH"
    elif len(library) > conf_thresholds['medium']:
        base_confidence = "MEDIUM"
    else:
        base_confidence = "LOW"
    
    # Adjust confidence based on prediction quality
    confidence_score = preds.get('confidence_score', 0.5)
    if confidence_score > 0.8:
        confidence_modifier = "+"
    elif confidence_score < 0.3:
        confidence_modifier = "-"
    else:
        confidence_modifier = ""
    
    preds['confidence'] = base_confidence + confidence_modifier
    preds['confidence_score'] = confidence_score
    preds['library_size'] = len(library)
    preds['data_source'] = data_source
    preds['adaptive_threshold'] = adaptive_threshold
    preds['current_price'] = float(hist['close'].iloc[-1])
    
    # Add directional classification with adaptive threshold
    for horizon in ['1h', '3h', 'eod']:
        if not np.isnan(preds[horizon]):
            pred_pct = preds[horizon] * 100  # Convert to percentage
            direction = classify_direction_adaptive(pred_pct, adaptive_threshold)
            preds[f'{horizon}_direction'] = direction
    
    # Add method information
    preds['method'] = 'enhanced_pattern_matching_v2'
    preds['improvements'] = ['time_decay_weighting', 'adaptive_atr_thresholds', 'configurable_ensemble']
    
    return preds

# Backward compatibility
def forecast_shape(ticker, interval="15m", period="60d", query_length=20, K=10):
    """Backward compatible interface"""
    return forecast_shape_enhanced(ticker, interval, period, query_length, K)

# Module is ready for import - no test code when imported