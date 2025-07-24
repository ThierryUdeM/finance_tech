#!/usr/bin/env python3
"""
TSLA V1 Model - Adapted Momentum + Shape Matching
Optimized for TSLA's high volatility and extreme moves
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def v1_tsla_model(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """TSLA-optimized momentum + shape matching model"""
    
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Import required libraries
    import talib
    
    # Prepare training data
    train_df = train_data.copy()
    train_df.columns = [col.lower() for col in train_df.columns]
    
    # Calculate simple features
    train_df['return'] = train_df['close'].pct_change()
    train_df['momentum_2'] = train_df['close'].pct_change(2)  # 30 min momentum (faster for TSLA)
    train_df['momentum_4'] = train_df['close'].pct_change(4)  # 1 hour momentum
    train_df['momentum_8'] = train_df['close'].pct_change(8)  # 2 hour momentum
    train_df['volume_ratio'] = train_df['volume'] / train_df['volume'].rolling(15).mean()  # Shorter window
    train_df['atr'] = talib.ATR(train_df['high'].values, train_df['low'].values, 
                                train_df['close'].values, timeperiod=10)  # Faster ATR
    train_df['volatility'] = train_df['return'].rolling(15).std()  # Shorter window for TSLA
    
    # Add extreme move detection for TSLA
    train_df['extreme_move'] = abs(train_df['return']) > 0.01  # 1% move in 15 min
    train_df['range_pct'] = (train_df['high'] - train_df['low']) / train_df['close']
    
    # Build simple pattern library
    pattern_window = 8  # 2 hours of data (shorter for fast-moving TSLA)
    pattern_library = []
    
    # Collect patterns with their future returns
    for i in range(pattern_window, len(train_df) - 8):
        # Skip if we don't have forward returns
        if i + 8 >= len(train_df):
            continue
            
        # Pattern: recent returns normalized by volatility
        returns = train_df['return'].iloc[i-pattern_window:i].values
        vol = train_df['volatility'].iloc[i]
        
        if vol > 0 and not np.isnan(vol):
            pattern = returns / vol
            
            # Future return (shorter horizon for TSLA - 30 min to 1 hour)
            future_ret = (train_df['close'].iloc[i+2] / train_df['close'].iloc[i]) - 1
            
            # Store pattern with metadata
            pattern_library.append({
                'pattern': pattern,
                'future_return': future_ret,
                'momentum': train_df['momentum_2'].iloc[i],  # Use short momentum
                'extreme': train_df['extreme_move'].iloc[i],
                'volume_ratio': train_df['volume_ratio'].iloc[i],
                'range_pct': train_df['range_pct'].iloc[i]
            })
    
    print(f"  Built pattern library with {len(pattern_library)} patterns")
    
    # Prepare test data
    test_df = test_data.copy()
    test_df.columns = [col.lower() for col in test_df.columns]
    
    # Calculate features for test data
    test_df['return'] = test_df['close'].pct_change()
    test_df['momentum_1'] = test_df['close'].pct_change(1)
    test_df['momentum_2'] = test_df['close'].pct_change(2)
    test_df['momentum_4'] = test_df['close'].pct_change(4)
    test_df['momentum_8'] = test_df['close'].pct_change(8)
    test_df['volume_ratio'] = test_df['volume'] / test_df['volume'].rolling(15).mean()
    test_df['atr'] = talib.ATR(test_df['high'].values, test_df['low'].values,
                              test_df['close'].values, timeperiod=10)
    test_df['volatility'] = test_df['return'].rolling(15).std()
    test_df['extreme_move'] = abs(test_df['return']) > 0.01
    test_df['range_pct'] = (test_df['high'] - test_df['low']) / test_df['close']
    
    # Combine for continuous data
    combined_df = pd.concat([train_df[-pattern_window:], test_df])
    
    # Generate signals
    daily_signals = {}  # Track signals per day
    all_candidates = []  # Store all candidates for ranking
    
    for i in range(pattern_window, len(test_df)):
        # Get current pattern
        returns = combined_df['return'].iloc[i:i+pattern_window].values
        vol = test_df['volatility'].iloc[i]
        
        if vol <= 0 or np.isnan(vol):
            continue
            
        current_pattern = returns / vol
        
        # Find similar patterns (simple distance)
        distances = []
        for p in pattern_library:
            # Euclidean distance
            dist = np.sqrt(np.sum((current_pattern - p['pattern'])**2))
            distances.append((dist, p))
        
        # Sort by distance and take top 15 (fewer for fast-moving TSLA)
        distances.sort(key=lambda x: x[0])
        nearest = distances[:15]
        
        if len(nearest) == 0:
            continue
        
        # Calculate expected return from similar patterns
        weights = [1.0 / (d[0] + 0.1) for d in nearest]  # Inverse distance weighting
        total_weight = sum(weights)
        
        expected_returns = [w * p[1]['future_return'] for w, p in zip(weights, nearest)]
        shape_return = sum(expected_returns) / total_weight if total_weight > 0 else 0
        
        # Get current indicators
        momentum = test_df['momentum_2'].iloc[i]  # Use short-term momentum
        if pd.isna(momentum):
            momentum = 0
            
        extreme = test_df['extreme_move'].iloc[i]
        range_pct = test_df['range_pct'].iloc[i]
        if pd.isna(range_pct):
            range_pct = 0
        
        # Combine signals with TSLA-specific weights
        # 1. Shape-based expected return (weight: 1.5) - lower for volatile TSLA
        # 2. Short momentum (weight: 3.5) - highest weight for momentum
        # 3. Extreme move fade (weight: 2.0) - important for TSLA
        
        signal_strength = 0
        
        # Shape component (reduced for TSLA)
        if abs(shape_return) > 0.001:  # 10 bps minimum (higher for TSLA)
            signal_strength += np.sign(shape_return) * min(abs(shape_return) * 50, 1) * 1.5
        
        # Momentum component (strongest signal for TSLA)
        if abs(momentum) > 0.002:  # 20 bps minimum
            signal_strength += np.sign(momentum) * min(abs(momentum) * 25, 1) * 3.5
        
        # Extreme move fade (TSLA specific)
        if extreme and i > 0:
            prev_return = test_df['return'].iloc[i-1]
            if abs(prev_return) > 0.01:  # Previous bar was extreme
                # Fade the extreme move
                signal_strength -= np.sign(prev_return) * 2.0
        
        # Range-based adjustment
        if range_pct > 0.015:  # High range (1.5%+)
            signal_strength *= 1.3  # Boost signals in volatile periods
        
        # Volume confirmation (critical for TSLA)
        vol_ratio = test_df['volume_ratio'].iloc[i]
        if not pd.isna(vol_ratio):
            if vol_ratio > 2.0:  # High volume spike
                signal_strength *= 1.5
            elif vol_ratio < 0.5:  # Very low volume
                signal_strength *= 0.5
        
        # Store candidate
        date = test_df.index[i].date() if hasattr(test_df.index[i], 'date') else i // 26
        
        all_candidates.append({
            'index': test_df.index[i],
            'date': date,
            'signal_strength': signal_strength,
            'shape_return': shape_return,
            'momentum': momentum,
            'extreme': extreme,
            'range_pct': range_pct,
            'price': test_df['close'].iloc[i]
        })
    
    # Rank and select top signals
    # Group by day and take top 4 signals (more for volatile TSLA)
    from collections import defaultdict
    daily_candidates = defaultdict(list)
    
    for cand in all_candidates:
        daily_candidates[cand['date']].append(cand)
    
    signal_count = 0
    min_strength = 0.8  # Higher minimum for TSLA quality control
    
    for date, candidates in daily_candidates.items():
        # Sort by absolute signal strength
        candidates.sort(key=lambda x: abs(x['signal_strength']), reverse=True)
        
        # Take top 4 signals per day (more opportunities with TSLA)
        for j, cand in enumerate(candidates[:4]):
            if abs(cand['signal_strength']) >= min_strength:
                signals.loc[cand['index'], 'signal'] = np.sign(cand['signal_strength'])
                signal_count += 1
    
    print(f"  Generated {signal_count} signals ({signal_count / len(daily_candidates):.1f} per day avg)")
    
    return signals