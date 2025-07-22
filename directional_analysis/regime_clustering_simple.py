#!/usr/bin/env python3
"""
Simplified Regime Clustering for NVDA Pattern Matching
Focus on volatility regimes as the main clustering feature
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_volatility_regimes(df, lookback_days=30, n_regimes=3):
    """
    Simple volatility-based regime classification
    
    Regimes:
    0: Low Volatility (bottom 33%)
    1: Medium Volatility (middle 33%) 
    2: High Volatility (top 33%)
    """
    
    # Calculate rolling 30-day realized volatility
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(window=lookback_days * 26).std() * np.sqrt(252 * 26)  # Annualized
    
    # Remove NaN values
    vol_data = rolling_vol.dropna()
    
    if len(vol_data) == 0:
        print("Insufficient data for regime calculation")
        return None, None
    
    # Define regime thresholds (33rd and 67th percentiles)
    low_threshold = vol_data.quantile(0.33)
    high_threshold = vol_data.quantile(0.67)
    
    # Classify regimes
    regimes = pd.Series(index=vol_data.index, dtype=int)
    regimes[vol_data <= low_threshold] = 0  # Low vol
    regimes[(vol_data > low_threshold) & (vol_data <= high_threshold)] = 1  # Medium vol
    regimes[vol_data > high_threshold] = 2  # High vol
    
    regime_summary = {
        0: f"Low Vol (≤{low_threshold:.1%})",
        1: f"Medium Vol ({low_threshold:.1%}-{high_threshold:.1%})", 
        2: f"High Vol (>{high_threshold:.1%})"
    }
    
    current_vol = vol_data.iloc[-1]
    if current_vol <= low_threshold:
        current_regime = 0
    elif current_vol <= high_threshold:
        current_regime = 1
    else:
        current_regime = 2
    
    print(f"Volatility Regimes:")
    for regime, desc in regime_summary.items():
        count = (regimes == regime).sum()
        print(f"  Regime {regime}: {desc} ({count:,} periods)")
    
    print(f"Current volatility: {current_vol:.1%}")
    print(f"Current regime: {current_regime} ({regime_summary[current_regime]})")
    
    return regimes, current_regime

def regime_pattern_matching(df, query_length=20, K=10, regime_weight=0.7):
    """
    Pattern matching with volatility regime awareness
    """
    print("Building regime-aware pattern library...")
    
    # Calculate regimes
    regimes, current_regime = calculate_volatility_regimes(df)
    
    if regimes is None or current_regime is None:
        print("Failed to calculate regimes, falling back to standard matching")
        return None
    
    # Build pattern library
    patterns = []
    bars_per_hour = 4
    bars_per_3h = 12
    
    for i in range(query_length, len(df) - bars_per_3h):
        # Skip if we don't have regime data for this period
        if df.index[i] not in regimes.index:
            continue
            
        pattern_regime = regimes.loc[df.index[i]]
        
        # Extract pattern
        pattern_prices = df['close'].iloc[i-query_length:i].values
        
        if len(pattern_prices) != query_length or pattern_prices.std() == 0:
            continue
            
        # Normalize pattern
        normalized_pattern = (pattern_prices - pattern_prices.mean()) / pattern_prices.std()
        
        # Calculate future returns
        base_price = pattern_prices[-1]
        future_returns = {}
        
        # 1h prediction
        if i + bars_per_hour < len(df):
            future_price_1h = df['close'].iloc[i + bars_per_hour]
            future_returns['1h'] = (future_price_1h / base_price) - 1
        
        # 3h prediction
        if i + bars_per_3h < len(df):
            future_price_3h = df['close'].iloc[i + bars_per_3h]
            future_returns['3h'] = (future_price_3h / base_price) - 1
        
        # EOD prediction (simplified - use 6h ahead or end of data)
        eod_idx = min(i + 6*bars_per_hour, len(df) - 1)
        future_price_eod = df['close'].iloc[eod_idx]
        future_returns['eod'] = (future_price_eod / base_price) - 1
        
        # Regime similarity weight
        if pattern_regime == current_regime:
            regime_similarity = 1.0
        else:
            # Different regimes get reduced weight
            regime_similarity = 1.0 - regime_weight
        
        patterns.append({
            'pattern': normalized_pattern,
            'returns': future_returns,
            'regime': pattern_regime,
            'regime_weight': regime_similarity,
            'timestamp': df.index[i]
        })
    
    print(f"Built library with {len(patterns)} patterns")
    
    # Current pattern
    current_pattern = df['close'].iloc[-query_length:].values
    if current_pattern.std() == 0:
        print("Current pattern has zero variance")
        return None
        
    current_normalized = (current_pattern - current_pattern.mean()) / current_pattern.std()
    
    # Calculate weighted distances
    distances = []
    regime_weights = []
    
    for p in patterns:
        euclidean_dist = np.sqrt(np.sum((current_normalized - p['pattern']) ** 2))
        distances.append(euclidean_dist)
        regime_weights.append(p['regime_weight'])
    
    distances = np.array(distances)
    regime_weights = np.array(regime_weights)
    
    # Apply regime weighting to distances (lower weight = higher effective distance)
    weighted_distances = distances / regime_weights
    
    # Get K nearest neighbors
    k_actual = min(K, len(patterns))
    nearest_indices = np.argsort(weighted_distances)[:k_actual]
    
    # Calculate regime composition of nearest neighbors
    same_regime_count = sum(1 for i in nearest_indices if patterns[i]['regime'] == current_regime)
    regime_match_ratio = same_regime_count / k_actual
    
    print(f"Using {k_actual} nearest neighbors")
    print(f"Same regime matches: {same_regime_count}/{k_actual} ({regime_match_ratio:.1%})")
    
    # Weighted prediction
    predictions = {'1h': 0, '3h': 0, 'eod': 0}
    total_weight = {'1h': 0, '3h': 0, 'eod': 0}
    
    for i, idx in enumerate(nearest_indices):
        pattern = patterns[idx]
        
        # Combined weight: regime similarity * inverse distance
        distance_weight = 1 / (1 + weighted_distances[idx])
        combined_weight = pattern['regime_weight'] * distance_weight
        
        for horizon in predictions:
            if horizon in pattern['returns']:
                predictions[horizon] += pattern['returns'][horizon] * combined_weight
                total_weight[horizon] += combined_weight
    
    # Normalize predictions
    final_predictions = {}
    for horizon in predictions:
        if total_weight[horizon] > 0:
            final_predictions[horizon] = predictions[horizon] / total_weight[horizon]
        else:
            final_predictions[horizon] = 0
    
    # Calculate confidence based on regime consistency
    base_confidence = min(1.0, regime_match_ratio + 0.3)  # Boost for regime consistency
    
    final_predictions.update({
        'current_regime': int(current_regime),
        'regime_match_ratio': regime_match_ratio,
        'confidence_score': base_confidence,
        'method': 'volatility_regime_aware',
        'patterns_analyzed': len(patterns),
        'same_regime_patterns': same_regime_count
    })
    
    return final_predictions

if __name__ == "__main__":
    # Test with NVDA data
    import os
    data_path = 'NVDA_15min_pattern_ready.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        
        print(f"Testing regime-aware pattern matching on {len(df):,} bars of NVDA data")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        results = regime_pattern_matching(df, query_length=20, K=20, regime_weight=0.7)
        
        if results:
            print(f"\n🎯 Regime-Aware Predictions:")
            print(f"Current Regime: {results['current_regime']}")
            print(f"Same Regime Patterns: {results['same_regime_patterns']}/{results['patterns_analyzed']} ({results['regime_match_ratio']:.1%})")
            print(f"Enhanced Confidence: {results['confidence_score']:.3f}")
            
            for horizon in ['1h', '3h', 'eod']:
                pred_pct = results[horizon] * 100
                print(f"{horizon.upper()}: {pred_pct:+.3f}%")
        else:
            print("Regime-aware pattern matching failed")
    else:
        print("NVDA data file not found")