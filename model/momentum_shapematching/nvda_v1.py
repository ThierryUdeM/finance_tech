#!/usr/bin/env python3
"""
Simplified Momentum + Shape Matching Model
Based on diagnostic findings - focus on what works
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

def momentum_shape_model(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Simple momentum + shape matching model"""
    
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Import required libraries
    import talib
    
    # Prepare training data
    train_df = train_data.copy()
    train_df.columns = [col.lower() for col in train_df.columns]
    
    # Calculate simple features
    train_df['return'] = train_df['close'].pct_change()
    train_df['momentum_4'] = train_df['close'].pct_change(4)  # 1 hour momentum
    train_df['momentum_12'] = train_df['close'].pct_change(12)  # 3 hour momentum
    train_df['volume_ratio'] = train_df['volume'] / train_df['volume'].rolling(20).mean()
    train_df['atr'] = talib.ATR(train_df['high'].values, train_df['low'].values, 
                                train_df['close'].values, timeperiod=14)
    train_df['volatility'] = train_df['return'].rolling(20).std()
    
    # Build simple pattern library
    pattern_window = 12  # 3 hours of data
    pattern_library = []
    
    # Collect patterns with their future returns
    for i in range(pattern_window, len(train_df) - 12):
        # Skip if we don't have forward returns
        if i + 12 >= len(train_df):
            continue
            
        # Pattern: recent returns normalized by volatility
        returns = train_df['return'].iloc[i-pattern_window:i].values
        vol = train_df['volatility'].iloc[i]
        
        if vol > 0 and not np.isnan(vol):
            pattern = returns / vol
            
            # Future return (1 hour ahead)
            future_ret = (train_df['close'].iloc[i+4] / train_df['close'].iloc[i]) - 1
            
            # Store pattern with metadata
            pattern_library.append({
                'pattern': pattern,
                'future_return': future_ret,
                'momentum': train_df['momentum_4'].iloc[i],
                'volume_ratio': train_df['volume_ratio'].iloc[i]
            })
    
    print(f"  Built pattern library with {len(pattern_library)} patterns")
    
    # Prepare test data
    test_df = test_data.copy()
    test_df.columns = [col.lower() for col in test_df.columns]
    
    # Calculate features for test data
    test_df['return'] = test_df['close'].pct_change()
    test_df['momentum_1'] = test_df['close'].pct_change(1)
    test_df['momentum_4'] = test_df['close'].pct_change(4)
    test_df['momentum_12'] = test_df['close'].pct_change(12)
    test_df['volume_ratio'] = test_df['volume'] / test_df['volume'].rolling(20).mean()
    test_df['atr'] = talib.ATR(test_df['high'].values, test_df['low'].values,
                              test_df['close'].values, timeperiod=14)
    test_df['volatility'] = test_df['return'].rolling(20).std()
    
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
        
        # Sort by distance and take top 20
        distances.sort(key=lambda x: x[0])
        nearest = distances[:20]
        
        if len(nearest) == 0:
            continue
        
        # Calculate expected return from similar patterns
        weights = [1.0 / (d[0] + 0.1) for d in nearest]  # Inverse distance weighting
        total_weight = sum(weights)
        
        expected_returns = [w * p[1]['future_return'] for w, p in zip(weights, nearest)]
        shape_return = sum(expected_returns) / total_weight if total_weight > 0 else 0
        
        # Get current momentum
        momentum = test_df['momentum_4'].iloc[i]
        if pd.isna(momentum):
            momentum = 0
        
        # Combine signals with weights
        # 1. Shape-based expected return (weight: 2.0)
        # 2. Recent momentum (weight: 3.0) - higher weight based on diagnostics
        # 3. Volume confirmation (weight: 1.0)
        
        signal_strength = 0
        
        # Shape component
        if abs(shape_return) > 0.0005:  # 5 bps minimum
            signal_strength += np.sign(shape_return) * min(abs(shape_return) * 100, 1) * 2.0
        
        # Momentum component (strongest signal)
        if abs(momentum) > 0.001:  # 10 bps minimum
            signal_strength += np.sign(momentum) * min(abs(momentum) * 50, 1) * 3.0
        
        # Volume confirmation
        vol_ratio = test_df['volume_ratio'].iloc[i]
        if not pd.isna(vol_ratio) and vol_ratio > 1.5:
            signal_strength *= 1.2
        
        # Store candidate
        date = test_df.index[i].date() if hasattr(test_df.index[i], 'date') else i // 26
        
        all_candidates.append({
            'index': test_df.index[i],
            'date': date,
            'signal_strength': signal_strength,
            'shape_return': shape_return,
            'momentum': momentum,
            'price': test_df['close'].iloc[i]
        })
    
    # Rank and select top signals
    # Group by day and take top 3 signals
    from collections import defaultdict
    daily_candidates = defaultdict(list)
    
    for cand in all_candidates:
        daily_candidates[cand['date']].append(cand)
    
    signal_count = 0
    min_strength = 0.5  # Minimum signal strength
    
    for date, candidates in daily_candidates.items():
        # Sort by absolute signal strength
        candidates.sort(key=lambda x: abs(x['signal_strength']), reverse=True)
        
        # Take top 3 signals per day
        for j, cand in enumerate(candidates[:3]):
            if abs(cand['signal_strength']) >= min_strength:
                signals.loc[cand['index'], 'signal'] = np.sign(cand['signal_strength'])
                signal_count += 1
    
    print(f"  Generated {signal_count} signals ({signal_count / len(daily_candidates):.1f} per day avg)")
    
    # If still no signals, use pure momentum fallback
    if signal_count == 0:
        print("  Using momentum fallback strategy...")
        momentum_threshold = 0.002  # 20 bps
        
        for i in range(12, len(test_df)):
            if i % 13 == 0:  # Roughly 2 signals per day
                mom = test_df['momentum_4'].iloc[i]
                if not pd.isna(mom) and abs(mom) > momentum_threshold:
                    signals.iloc[i, 0] = np.sign(mom)
                    signal_count += 1
        
        print(f"  Generated {signal_count} momentum signals")
    
    return signals


def test_momentum_shape_model():
    """Test the momentum + shape model using walk-forward framework"""
    
    print("\n" + "="*80)
    print("TESTING MOMENTUM + SHAPE MATCHING MODEL")
    print("="*80)
    
    from walk_forward_framework import WalkForwardTester
    
    # Configure test
    tester = WalkForwardTester(
        lookback_days=90,   # 3 months training
        test_days=20,       # 20 days test  
        min_train_days=60,  # Minimum 2 months
        transaction_cost=0.0
    )
    
    # Run test
    result = tester.run_test(
        model_name='Momentum_Shape',
        model_func=momentum_shape_model,
        data_path='data/NVDA_15min_clean.csv'
    )
    
    if result:
        print("\n" + "="*60)
        print("MOMENTUM + SHAPE MODEL RESULTS")
        print("="*60)
        print(f"Total Return: {result['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {result['win_rate']*100:.1f}%")
        print(f"Number of Trades: {result['num_trades']}")
        print(f"Profit Factor: {result.get('profit_factor', 'N/A')}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'results/MOMENTUM_SHAPE_{timestamp}.txt', 'w') as f:
            f.write("Momentum + Shape Matching Model Results\n")
            f.write("="*50 + "\n\n")
            f.write("Model Design:\n")
            f.write("- Simple pattern matching (12-bar window)\n")
            f.write("- Momentum signals (1h, 3h)\n")
            f.write("- Volume confirmation\n")
            f.write("- Top 3 signals per day\n\n")
            f.write("Results:\n")
            f.write(f"Total Return: {result['total_return']*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {result['max_drawdown']*100:.2f}%\n")
            f.write(f"Win Rate: {result['win_rate']*100:.1f}%\n")
            f.write(f"Number of Trades: {result['num_trades']}\n")
            f.write(f"Avg Trades/Day: {result['num_trades'] / (result['num_splits'] * 20):.1f}\n\n")
            f.write("Comparison:\n")
            f.write("Daily Pattern Scanner: 18.82% return, 2.53 Sharpe (daily data)\n")
            f.write("Simple Technical: 0.91% return, 1.54 Sharpe\n")
            f.write("Hybrid Model: 0.00% return (too complex)\n")
            f.write(f"Momentum+Shape: {result['total_return']*100:.2f}% return, {result['sharpe_ratio']:.2f} Sharpe\n")
        
        print(f"\nResults saved to results/MOMENTUM_SHAPE_{timestamp}.txt")
        
        # Compare with other models  
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print("Model                    Return    Sharpe   Data")
        print("-" * 50)
        print(f"Daily Pattern Scanner    18.82%    2.53     daily")
        print(f"Simple Technical         0.91%     1.54     15-min")
        print(f"Signal Model V2         -1.18%    -2.31     15-min")
        print(f"Hybrid Model            0.00%     0.00      15-min")
        print(f"Momentum + Shape        {result['total_return']*100:6.2f}%    {result['sharpe_ratio']:.2f}     15-min")
    
    return result


if __name__ == "__main__":
    test_momentum_shape_model()