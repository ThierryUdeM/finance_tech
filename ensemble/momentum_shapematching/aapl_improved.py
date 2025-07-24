#!/usr/bin/env python3
"""
AAPL Improved Model - Quality over Quantity
Based on analysis showing:
- Mean reversion works best at 2% threshold
- Volume spikes mark reversals
- Low volatility regimes best
- Focus on hours 14, 15, 20
"""

import sys
import os
import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

def aapl_improved_model(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """AAPL improved model - patient mean reversion with volume confirmation"""
    
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Prepare training data
    train_df = train_data.copy()
    train_df.columns = [col.lower() for col in train_df.columns]
    
    # Core features
    train_df['returns'] = train_df['close'].pct_change()
    train_df['sma20'] = train_df['close'].rolling(20).mean()
    train_df['sma50'] = train_df['close'].rolling(50).mean()
    train_df['dist_sma20'] = (train_df['close'] - train_df['sma20']) / train_df['sma20']
    
    # Volatility regime
    train_df['volatility'] = train_df['returns'].rolling(20).std()
    train_df['vol_percentile'] = train_df['volatility'].rolling(100).rank(pct=True)
    
    # Volume analysis
    train_df['volume_ma'] = train_df['volume'].rolling(20).mean()
    train_df['volume_ratio'] = train_df['volume'] / train_df['volume_ma']
    train_df['volume_spike'] = train_df['volume_ratio'] > 1.5
    
    # Technical indicators
    train_df['rsi'] = talib.RSI(train_df['close'].values, timeperiod=14)
    train_df['bb_upper'], train_df['bb_middle'], train_df['bb_lower'] = talib.BBANDS(
        train_df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    train_df['bb_width'] = (train_df['bb_upper'] - train_df['bb_lower']) / train_df['bb_middle']
    
    # ATR for position sizing
    train_df['atr'] = talib.ATR(train_df['high'].values, train_df['low'].values, 
                                train_df['close'].values, timeperiod=14)
    
    # Multi-timeframe confirmation
    train_df['sma5'] = train_df['close'].rolling(5).mean()
    train_df['sma10'] = train_df['close'].rolling(10).mean()
    train_df['trend_align'] = (
        (train_df['sma5'] > train_df['sma10']) & 
        (train_df['sma10'] > train_df['sma20'])
    ).astype(int) - (
        (train_df['sma5'] < train_df['sma10']) & 
        (train_df['sma10'] < train_df['sma20'])
    ).astype(int)
    
    # Calculate mean reversion edges from training data
    reversion_stats = []
    for threshold in [0.01, 0.015, 0.02, 0.025]:
        # Above SMA
        above_mask = train_df['dist_sma20'] > threshold
        if above_mask.sum() > 20:
            above_returns = []
            for idx in train_df[above_mask].index:
                try:
                    idx_pos = train_df.index.get_loc(idx)
                    if idx_pos + 4 < len(train_df):
                        ret = (train_df.iloc[idx_pos+4]['close'] / train_df.iloc[idx_pos]['close']) - 1
                        above_returns.append(ret)
                except:
                    pass
            if above_returns:
                avg_ret = np.mean(above_returns)
                win_rate = sum(1 for r in above_returns if r < 0) / len(above_returns)
                reversion_stats.append(('above', threshold, avg_ret, win_rate))
    
    print(f"  Reversion edges calculated from {len(reversion_stats)} scenarios")
    
    # Prepare test data
    test_df = test_data.copy()
    test_df.columns = [col.lower() for col in test_df.columns]
    
    # Calculate all features
    test_df['returns'] = test_df['close'].pct_change()
    test_df['sma20'] = test_df['close'].rolling(20).mean()
    test_df['sma50'] = test_df['close'].rolling(50).mean()
    test_df['dist_sma20'] = (test_df['close'] - test_df['sma20']) / test_df['sma20']
    
    test_df['volatility'] = test_df['returns'].rolling(20).std()
    test_df['vol_percentile'] = test_df['volatility'].rolling(100).rank(pct=True)
    
    test_df['volume_ma'] = test_df['volume'].rolling(20).mean()
    test_df['volume_ratio'] = test_df['volume'] / test_df['volume_ma']
    test_df['volume_spike'] = test_df['volume_ratio'] > 1.5
    
    test_df['rsi'] = talib.RSI(test_df['close'].values, timeperiod=14)
    test_df['bb_upper'], test_df['bb_middle'], test_df['bb_lower'] = talib.BBANDS(
        test_df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    test_df['bb_width'] = (test_df['bb_upper'] - test_df['bb_lower']) / test_df['bb_middle']
    
    test_df['atr'] = talib.ATR(test_df['high'].values, test_df['low'].values,
                              test_df['close'].values, timeperiod=14)
    
    test_df['sma5'] = test_df['close'].rolling(5).mean()
    test_df['sma10'] = test_df['close'].rolling(10).mean()
    test_df['trend_align'] = (
        (test_df['sma5'] > test_df['sma10']) & 
        (test_df['sma10'] > test_df['sma20'])
    ).astype(int) - (
        (test_df['sma5'] < test_df['sma10']) & 
        (test_df['sma10'] < test_df['sma20'])
    ).astype(int)
    
    # Extract hour
    if hasattr(test_df.index, 'hour'):
        test_df['hour'] = test_df.index.hour
    else:
        test_df['hour'] = 14  # Default to best hour
    
    # Generate signals - QUALITY OVER QUANTITY
    signal_count = 0
    daily_signals = {}
    consecutive_losses = 0
    
    for i in range(100, len(test_df)):  # Need history for indicators
        
        # Get current values
        dist_sma = test_df.iloc[i]['dist_sma20']
        vol_pct = test_df.iloc[i]['vol_percentile']
        volume_spike = test_df.iloc[i]['volume_spike']
        rsi = test_df.iloc[i]['rsi']
        hour = test_df.iloc[i]['hour']
        trend = test_df.iloc[i]['trend_align']
        close = test_df.iloc[i]['close']
        bb_upper = test_df.iloc[i]['bb_upper']
        bb_lower = test_df.iloc[i]['bb_lower']
        
        # Skip if indicators not ready
        if pd.isna(dist_sma) or pd.isna(rsi) or pd.isna(vol_pct):
            continue
        
        # Date for daily limit
        date = test_df.index[i].date() if hasattr(test_df.index[i], 'date') else i // 26
        if date not in daily_signals:
            daily_signals[date] = 0
        
        # STRICT LIMIT: Max 1 signal per day for AAPL
        if daily_signals[date] >= 1:
            continue
        
        # Skip if recent losses (risk management)
        if consecutive_losses >= 2:
            consecutive_losses -= 1
            continue
        
        signal = 0
        confidence = 0
        
        # STRATEGY 1: High-confidence mean reversion (2%+ from SMA)
        if abs(dist_sma) > 0.02:  # 2% threshold based on analysis
            
            # Additional filters for quality
            if vol_pct < 0.7:  # Low volatility regime preferred
                
                if dist_sma > 0.02:  # Above SMA
                    # Short conditions
                    if rsi > 65 and trend <= 0:  # Overbought + not trending up
                        if hour in [14, 15, 20]:  # Best hours
                            signal = -1
                            confidence = min(abs(dist_sma) * 50, 1.0)
                            
                elif dist_sma < -0.02:  # Below SMA
                    # Long conditions
                    if rsi < 35 and trend >= 0:  # Oversold + not trending down
                        if hour in [14, 15, 20]:  # Best hours
                            signal = 1
                            confidence = min(abs(dist_sma) * 50, 1.0)
        
        # STRATEGY 2: Volume spike reversal
        elif volume_spike and i > 0:
            prev_return = test_df.iloc[i-1]['returns']
            
            # Volume spike after big move = reversal
            if abs(prev_return) > 0.005:  # 50 bps move
                if close > bb_upper and rsi > 70:
                    signal = -1
                    confidence = 0.7
                elif close < bb_lower and rsi < 30:
                    signal = 1
                    confidence = 0.7
        
        # STRATEGY 3: Bollinger Band squeeze breakout fade
        elif test_df.iloc[i]['bb_width'] < 0.01:  # Tight bands
            if close > bb_upper:
                signal = -1
                confidence = 0.5
            elif close < bb_lower:
                signal = 1
                confidence = 0.5
        
        # Final quality check
        if signal != 0 and confidence >= 0.5:
            # Additional volatility filter
            current_vol = test_df.iloc[i]['volatility']
            if current_vol > test_df['volatility'].rolling(50).mean().iloc[i] * 1.5:
                signal = 0  # Skip high volatility periods
        
        # Set signal
        if signal != 0:
            signals.iloc[i, 0] = signal
            signal_count += 1
            daily_signals[date] += 1
    
    print(f"  Generated {signal_count} HIGH-QUALITY signals ({signal_count / len(daily_signals):.1f} per day avg)")
    
    return signals