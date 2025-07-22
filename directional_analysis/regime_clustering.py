#!/usr/bin/env python3
"""
Regime Clustering for NVDA Pattern Matching
Clusters historical patterns by market regime characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def calculate_regime_features(df, lookback_days=30):
    """Calculate regime features for each time period"""
    features = []
    
    for i in range(lookback_days * 26, len(df)):  # 26 bars per day (15min)
        window = df.iloc[i-lookback_days*26:i]
        
        if len(window) < lookback_days * 20:  # Minimum data requirement
            continue
            
        # Feature 1: Realized Volatility (30-day)
        returns = window['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252 * 26)  # Annualized
        
        # Feature 2: Trend Strength (30-day)
        price_change = (window['close'].iloc[-1] / window['close'].iloc[0] - 1)
        trend_strength = abs(price_change)
        
        # Feature 3: Volume Regime (relative to 90-day average)
        if i >= 90 * 26:
            vol_window_90d = df.iloc[i-90*26:i]
            avg_volume_90d = vol_window_90d['volume'].mean()
            current_volume_regime = window['volume'].mean() / avg_volume_90d
        else:
            current_volume_regime = 1.0
            
        # Feature 4: Intraday Range Expansion
        daily_ranges = []
        for day_start in range(0, len(window), 26):  # Each day
            day_data = window.iloc[day_start:day_start+26]
            if len(day_data) > 0:
                day_range = (day_data['high'].max() - day_data['low'].min()) / day_data['close'].iloc[-1]
                daily_ranges.append(day_range)
        
        avg_daily_range = np.mean(daily_ranges) if daily_ranges else 0.01
        
        # Feature 5: Time-based features (earnings proximity, option expiry)
        timestamp = df.index[i]
        
        # Quarterly earnings proximity (rough estimate)
        month = timestamp.month
        days_to_earnings = min(abs(month - 2), abs(month - 5), abs(month - 8), abs(month - 11)) * 30
        earnings_proximity = np.exp(-days_to_earnings / 45)  # Decay function
        
        # Option expiry proximity (3rd Friday of month)
        third_friday = pd.Timestamp(timestamp.year, timestamp.month, 15)
        while third_friday.dayofweek != 4:  # Friday
            third_friday += timedelta(days=1)
            
        days_to_expiry = abs((timestamp - third_friday).days)
        expiry_proximity = np.exp(-days_to_expiry / 7)  # Weekly decay
        
        features.append({
            'timestamp': timestamp,
            'realized_vol': realized_vol,
            'trend_strength': trend_strength,
            'volume_regime': current_volume_regime,
            'daily_range': avg_daily_range,
            'earnings_proximity': earnings_proximity,
            'expiry_proximity': expiry_proximity,
            'index': i
        })
    
    return pd.DataFrame(features)

def identify_regimes(features_df, n_clusters=5):
    """Cluster time periods into market regimes"""
    
    # Feature matrix for clustering
    feature_matrix = features_df[[
        'realized_vol', 'trend_strength', 'volume_regime', 
        'daily_range', 'earnings_proximity', 'expiry_proximity'
    ]].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix_scaled)
    
    features_df = features_df.copy()
    features_df['regime'] = clusters
    
    # Analyze regime characteristics
    regime_summary = []
    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['regime'] == cluster]
        
        regime_summary.append({
            'regime': cluster,
            'count': len(cluster_data),
            'avg_vol': cluster_data['realized_vol'].mean(),
            'avg_trend': cluster_data['trend_strength'].mean(),
            'avg_volume_regime': cluster_data['volume_regime'].mean(),
            'avg_range': cluster_data['daily_range'].mean(),
            'description': classify_regime(cluster_data)
        })
    
    regime_df = pd.DataFrame(regime_summary)
    
    return features_df, regime_df, scaler, kmeans

def classify_regime(cluster_data):
    """Generate human-readable regime description"""
    vol = cluster_data['realized_vol'].mean()
    trend = cluster_data['trend_strength'].mean()
    volume = cluster_data['volume_regime'].mean()
    
    # Volatility classification
    if vol > 0.4:
        vol_desc = "High Vol"
    elif vol > 0.25:
        vol_desc = "Medium Vol"
    else:
        vol_desc = "Low Vol"
    
    # Trend classification
    if trend > 0.15:
        trend_desc = "Strong Trend"
    elif trend > 0.05:
        trend_desc = "Moderate Trend"
    else:
        trend_desc = "Range-bound"
    
    # Volume classification
    if volume > 1.3:
        vol_desc_str = "High Volume"
    elif volume > 0.7:
        vol_desc_str = "Normal Volume"
    else:
        vol_desc_str = "Low Volume"
    
    return f"{vol_desc}, {trend_desc}, {vol_desc_str}"

def get_current_regime(df, features_df, scaler, kmeans):
    """Determine current market regime"""
    current_features = calculate_regime_features(df, lookback_days=30)
    
    if len(current_features) == 0:
        return None, "Unknown"
    
    latest_features = current_features.iloc[-1]
    
    feature_vector = [[
        latest_features['realized_vol'],
        latest_features['trend_strength'], 
        latest_features['volume_regime'],
        latest_features['daily_range'],
        latest_features['earnings_proximity'],
        latest_features['expiry_proximity']
    ]]
    
    feature_vector_scaled = scaler.transform(feature_vector)
    current_regime = kmeans.predict(feature_vector_scaled)[0]
    
    return current_regime, latest_features

def regime_aware_pattern_matching(df, query_length=20, K=10, regime_weight=0.7):
    """
    Enhanced pattern matching with regime awareness
    
    regime_weight: 0.0 = ignore regimes, 1.0 = only use same regime
    """
    print("Building regime-aware pattern library...")
    
    # Calculate regime features for all historical data
    features_df = calculate_regime_features(df, lookback_days=30)
    
    if len(features_df) < 100:
        print("Insufficient data for regime clustering, falling back to standard matching")
        return None
    
    # Identify regimes
    features_df, regime_summary, scaler, kmeans = identify_regimes(features_df, n_clusters=5)
    
    print(f"Identified {len(regime_summary)} market regimes:")
    for _, regime in regime_summary.iterrows():
        print(f"  Regime {regime['regime']}: {regime['description']} ({regime['count']} periods)")
    
    # Get current regime
    current_regime, current_features = get_current_regime(df, features_df, scaler, kmeans)
    
    if current_regime is None:
        print("Could not determine current regime")
        return None
        
    print(f"Current regime: {current_regime}")
    
    # Build pattern library with regime weighting
    patterns = []
    
    # Get patterns from all regimes, but weight by regime similarity
    for _, row in features_df.iterrows():
        idx = row['index']
        pattern_regime = row['regime']
        
        # Skip if not enough lookahead data
        if idx + query_length + 12 >= len(df):  # Need 3h lookahead (12 bars)
            continue
            
        # Extract pattern
        pattern_prices = df['close'].iloc[idx-query_length:idx].values
        
        if len(pattern_prices) != query_length:
            continue
            
        # Normalize pattern
        if pattern_prices.std() > 0:
            normalized_pattern = (pattern_prices - pattern_prices.mean()) / pattern_prices.std()
        else:
            continue
            
        # Calculate future returns
        base_price = pattern_prices[-1]
        future_returns = {}
        
        # 1h, 3h predictions
        for h, bars in [(1, 4), (3, 12)]:
            if idx + bars < len(df):
                future_price = df['close'].iloc[idx + bars]
                future_returns[f'{h}h'] = (future_price / base_price) - 1
        
        # EOD prediction
        # Find end of same trading day
        pattern_time = df.index[idx]
        eod_idx = idx
        while eod_idx < len(df) - 1 and df.index[eod_idx].date() == pattern_time.date():
            eod_idx += 1
        eod_idx -= 1  # Last bar of same day
        
        if eod_idx > idx:
            eod_price = df['close'].iloc[eod_idx]
            future_returns['eod'] = (eod_price / base_price) - 1
        
        # Regime similarity weight
        if pattern_regime == current_regime:
            regime_sim_weight = 1.0
        else:
            # Calculate regime distance (simplified)
            regime_sim_weight = 1.0 - regime_weight  # Reduced weight for different regimes
        
        patterns.append({
            'pattern': normalized_pattern,
            'returns': future_returns,
            'regime': pattern_regime,
            'regime_weight': regime_sim_weight,
            'timestamp': df.index[idx]
        })
    
    print(f"Built pattern library with {len(patterns)} patterns")
    
    # Current pattern matching
    current_pattern = df['close'].iloc[-query_length:].values
    if current_pattern.std() > 0:
        current_normalized = (current_pattern - current_pattern.mean()) / current_pattern.std()
    else:
        return None
    
    # Calculate distances with regime weighting
    distances = []
    for p in patterns:
        euclidean_dist = np.sqrt(np.sum((current_normalized - p['pattern']) ** 2))
        regime_weighted_dist = euclidean_dist / p['regime_weight']  # Lower weight = higher distance
        distances.append(regime_weighted_dist)
    
    distances = np.array(distances)
    
    # Get K nearest neighbors
    k_actual = min(K, len(patterns))
    nearest_indices = np.argsort(distances)[:k_actual]
    
    # Weighted prediction
    predictions = {'1h': 0, '3h': 0, 'eod': 0}
    weights_sum = {'1h': 0, '3h': 0, 'eod': 0}
    
    for idx in nearest_indices:
        pattern = patterns[idx]
        weight = pattern['regime_weight'] / (1 + distances[idx])  # Combined regime + distance weight
        
        for horizon in predictions:
            if horizon in pattern['returns']:
                predictions[horizon] += pattern['returns'][horizon] * weight
                weights_sum[horizon] += weight
    
    # Normalize predictions
    final_predictions = {}
    for horizon in predictions:
        if weights_sum[horizon] > 0:
            final_predictions[horizon] = predictions[horizon] / weights_sum[horizon]
        else:
            final_predictions[horizon] = 0
    
    # Add regime metadata
    same_regime_count = sum(1 for i in nearest_indices if patterns[i]['regime'] == current_regime)
    
    final_predictions.update({
        'current_regime': int(current_regime),
        'regime_match_ratio': same_regime_count / k_actual,
        'regime_summary': regime_summary.to_dict('records'),
        'method': 'regime_aware_pattern_matching'
    })
    
    return final_predictions

if __name__ == "__main__":
    # Test with NVDA data
    import os
    data_path = 'NVDA_15min_pattern_ready.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df.columns = [col.lower() for col in df.columns]
        
        results = regime_aware_pattern_matching(df, query_length=20, K=20, regime_weight=0.7)
        
        if results:
            print(f"\nRegime-Aware Predictions:")
            print(f"Current Regime: {results['current_regime']}")
            print(f"Regime Match Ratio: {results['regime_match_ratio']:.2%}")
            for horizon in ['1h', '3h', 'eod']:
                print(f"{horizon.upper()}: {results[horizon]*100:+.3f}%")
    else:
        print("NVDA data file not found")