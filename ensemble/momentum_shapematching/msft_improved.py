#!/usr/bin/env python3
"""
MSFT Improved Model - Extreme Patience Strategy
Based on analysis showing:
- Only 2%+ moves from SMA have edge
- Best in high volatility regimes for mean reversion
- Very tight range requires extreme selectivity
- Multi-timeframe confirmation crucial
"""

import sys
import os
import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

def msft_improved_model(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """MSFT improved model - extreme patience for high-probability setups"""
    
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Prepare training data
    train_df = train_data.copy()
    train_df.columns = [col.lower() for col in train_df.columns]
    
    # Core features
    train_df['returns'] = train_df['close'].pct_change()
    
    # Multiple timeframe SMAs
    train_df['sma10'] = train_df['close'].rolling(10).mean()
    train_df['sma20'] = train_df['close'].rolling(20).mean()
    train_df['sma50'] = train_df['close'].rolling(50).mean()
    train_df['sma100'] = train_df['close'].rolling(100).mean()
    
    # Distance metrics
    train_df['dist_sma20'] = (train_df['close'] - train_df['sma20']) / train_df['sma20']
    train_df['dist_sma50'] = (train_df['close'] - train_df['sma50']) / train_df['sma50']
    
    # Volatility analysis
    train_df['volatility'] = train_df['returns'].rolling(20).std()
    train_df['vol_rank'] = train_df['volatility'].rolling(252).rank(pct=True)  # Yearly rank
    
    # Range detection
    train_df['high_20'] = train_df['high'].rolling(20).max()
    train_df['low_20'] = train_df['low'].rolling(20).min()
    train_df['range_20'] = (train_df['high_20'] - train_df['low_20']) / train_df['close']
    train_df['range_position'] = (train_df['close'] - train_df['low_20']) / (train_df['high_20'] - train_df['low_20'])
    
    # Volume patterns
    train_df['volume_ma'] = train_df['volume'].rolling(50).mean()  # Longer MA for MSFT
    train_df['volume_ratio'] = train_df['volume'] / train_df['volume_ma']
    
    # Advanced indicators
    train_df['rsi'] = talib.RSI(train_df['close'].values, timeperiod=14)
    train_df['cci'] = talib.CCI(train_df['high'].values, train_df['low'].values, 
                                train_df['close'].values, timeperiod=20)
    
    # Bollinger Bands with multiple deviations
    train_df['bb_upper_2'], train_df['bb_middle'], train_df['bb_lower_2'] = talib.BBANDS(
        train_df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    train_df['bb_upper_3'], _, train_df['bb_lower_3'] = talib.BBANDS(
        train_df['close'].values, timeperiod=20, nbdevup=3, nbdevdn=3)
    
    # VWAP approximation
    train_df['vwap'] = (train_df['close'] * train_df['volume']).rolling(20).sum() / train_df['volume'].rolling(20).sum()
    train_df['dist_vwap'] = (train_df['close'] - train_df['vwap']) / train_df['vwap']
    
    # Microstructure
    train_df['spread'] = (train_df['high'] - train_df['low']) / train_df['close']
    train_df['close_position'] = (train_df['close'] - train_df['low']) / (train_df['high'] - train_df['low'])
    
    # Calculate extreme move statistics
    extreme_threshold = 0.015  # 1.5% for MSFT
    extreme_moves = []
    
    for i in range(100, len(train_df) - 4):
        if abs(train_df.iloc[i]['dist_sma20']) > extreme_threshold:
            future_ret = (train_df.iloc[i+4]['close'] / train_df.iloc[i]['close']) - 1
            vol_rank = train_df.iloc[i]['vol_rank']
            extreme_moves.append({
                'distance': train_df.iloc[i]['dist_sma20'],
                'future_return': future_ret,
                'vol_rank': vol_rank,
                'profitable': abs(future_ret) > 0.002  # 20 bps profit threshold
            })
    
    if extreme_moves:
        profitable_rate = sum(1 for m in extreme_moves if m['profitable']) / len(extreme_moves)
        print(f"  Extreme move profitability: {profitable_rate:.1%} from {len(extreme_moves)} samples")
    
    # Prepare test data
    test_df = test_data.copy()
    test_df.columns = [col.lower() for col in test_df.columns]
    
    # Calculate all features for test data
    test_df['returns'] = test_df['close'].pct_change()
    
    test_df['sma10'] = test_df['close'].rolling(10).mean()
    test_df['sma20'] = test_df['close'].rolling(20).mean()
    test_df['sma50'] = test_df['close'].rolling(50).mean()
    test_df['sma100'] = test_df['close'].rolling(100).mean()
    
    test_df['dist_sma20'] = (test_df['close'] - test_df['sma20']) / test_df['sma20']
    test_df['dist_sma50'] = (test_df['close'] - test_df['sma50']) / test_df['sma50']
    
    test_df['volatility'] = test_df['returns'].rolling(20).std()
    test_df['vol_rank'] = test_df['volatility'].rolling(100).rank(pct=True)
    
    test_df['high_20'] = test_df['high'].rolling(20).max()
    test_df['low_20'] = test_df['low'].rolling(20).min()
    test_df['range_20'] = (test_df['high_20'] - test_df['low_20']) / test_df['close']
    test_df['range_position'] = (test_df['close'] - test_df['low_20']) / (test_df['high_20'] - test_df['low_20'])
    
    test_df['volume_ma'] = test_df['volume'].rolling(50).mean()
    test_df['volume_ratio'] = test_df['volume'] / test_df['volume_ma']
    
    test_df['rsi'] = talib.RSI(test_df['close'].values, timeperiod=14)
    test_df['cci'] = talib.CCI(test_df['high'].values, test_df['low'].values,
                              test_df['close'].values, timeperiod=20)
    
    test_df['bb_upper_2'], test_df['bb_middle'], test_df['bb_lower_2'] = talib.BBANDS(
        test_df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
    test_df['bb_upper_3'], _, test_df['bb_lower_3'] = talib.BBANDS(
        test_df['close'].values, timeperiod=20, nbdevup=3, nbdevdn=3)
    
    test_df['vwap'] = (test_df['close'] * test_df['volume']).rolling(20).sum() / test_df['volume'].rolling(20).sum()
    test_df['dist_vwap'] = (test_df['close'] - test_df['vwap']) / test_df['vwap']
    
    test_df['spread'] = (test_df['high'] - test_df['low']) / test_df['close']
    test_df['close_position'] = (test_df['close'] - test_df['low']) / (test_df['high'] - test_df['low'])
    
    # Extract hour
    if hasattr(test_df.index, 'hour'):
        test_df['hour'] = test_df.index.hour
    else:
        test_df['hour'] = 14
    
    # Generate signals - EXTREME SELECTIVITY
    signal_count = 0
    daily_signals = {}
    weekly_signals = {}
    
    for i in range(100, len(test_df)):
        
        # Get current values
        dist_sma20 = test_df.iloc[i]['dist_sma20']
        dist_sma50 = test_df.iloc[i]['dist_sma50']
        vol_rank = test_df.iloc[i]['vol_rank']
        range_pos = test_df.iloc[i]['range_position']
        rsi = test_df.iloc[i]['rsi']
        cci = test_df.iloc[i]['cci']
        close = test_df.iloc[i]['close']
        bb_upper_2 = test_df.iloc[i]['bb_upper_2']
        bb_lower_2 = test_df.iloc[i]['bb_lower_2']
        bb_upper_3 = test_df.iloc[i]['bb_upper_3']
        bb_lower_3 = test_df.iloc[i]['bb_lower_3']
        volume_ratio = test_df.iloc[i]['volume_ratio']
        spread = test_df.iloc[i]['spread']
        
        # Skip if indicators not ready
        if pd.isna(dist_sma20) or pd.isna(rsi) or pd.isna(cci):
            continue
        
        # Date tracking
        date = test_df.index[i].date() if hasattr(test_df.index[i], 'date') else i // 26
        week = date.isocalendar()[1] if hasattr(date, 'isocalendar') else i // 130
        
        if date not in daily_signals:
            daily_signals[date] = 0
        if week not in weekly_signals:
            weekly_signals[week] = 0
        
        # ULTRA-STRICT LIMITS: Max 1 per day, 3 per week
        if daily_signals[date] >= 1 or weekly_signals[week] >= 3:
            continue
        
        signal = 0
        confidence = 0
        
        # STRATEGY 1: Extreme distance from mean (primary)
        if abs(dist_sma20) > 0.02:  # 2%+ only based on analysis
            
            # Multi-timeframe confirmation
            sma_aligned = (
                (dist_sma20 > 0 and dist_sma50 > 0) or 
                (dist_sma20 < 0 and dist_sma50 < 0)
            )
            
            if sma_aligned:
                
                if dist_sma20 > 0.02:  # Extreme overbought
                    # Multiple confirmations required
                    conditions = [
                        rsi > 70,
                        cci > 100,
                        close > bb_upper_2,
                        range_pos > 0.9,  # Near range high
                        spread < 0.003   # Not too volatile
                    ]
                    
                    if sum(conditions) >= 4:  # Need 4/5 conditions
                        signal = -1
                        confidence = min(abs(dist_sma20) * 40 + sum(conditions) * 0.1, 1.0)
                        
                elif dist_sma20 < -0.02:  # Extreme oversold
                    conditions = [
                        rsi < 30,
                        cci < -100,
                        close < bb_lower_2,
                        range_pos < 0.1,  # Near range low
                        spread < 0.003    # Not too volatile
                    ]
                    
                    if sum(conditions) >= 4:  # Need 4/5 conditions
                        signal = 1
                        confidence = min(abs(dist_sma20) * 40 + sum(conditions) * 0.1, 1.0)
        
        # STRATEGY 2: 3-sigma Bollinger Band touch (rare events)
        elif close > bb_upper_3 or close < bb_lower_3:
            
            if close > bb_upper_3 and vol_rank > 0.5:  # High vol regime
                if rsi > 75 and volume_ratio < 1.5:  # No volume confirmation
                    signal = -1
                    confidence = 0.8
                    
            elif close < bb_lower_3 and vol_rank > 0.5:  # High vol regime
                if rsi < 25 and volume_ratio < 1.5:  # No panic selling
                    signal = 1
                    confidence = 0.8
        
        # STRATEGY 3: Range breakout fade (only at extremes)
        elif test_df.iloc[i]['range_20'] > 0.03:  # Wide range (3%+)
            
            if range_pos > 0.95 and i > 0:  # At range high
                if test_df.iloc[i-1]['close_position'] < 0.9:  # Just reached
                    if volume_ratio > 1.5 and rsi > 65:
                        signal = -1
                        confidence = 0.6
                        
            elif range_pos < 0.05 and i > 0:  # At range low
                if test_df.iloc[i-1]['close_position'] > 0.1:  # Just reached
                    if volume_ratio > 1.5 and rsi < 35:
                        signal = 1
                        confidence = 0.6
        
        # Final quality filter - only take highest confidence trades
        if signal != 0 and confidence < 0.7:
            signal = 0
        
        # Risk check - avoid if recent volatility spike
        if signal != 0 and i > 5:
            recent_vol = test_df['volatility'].iloc[i-5:i].max()
            avg_vol = test_df['volatility'].iloc[i-50:i-5].mean()
            if recent_vol > avg_vol * 2:
                signal = 0  # Skip volatile periods
        
        # Set signal
        if signal != 0:
            signals.iloc[i, 0] = signal
            signal_count += 1
            daily_signals[date] += 1
            weekly_signals[week] += 1
    
    print(f"  Generated {signal_count} ULTRA-HIGH-QUALITY signals")
    print(f"  Average per day: {signal_count / len(daily_signals):.2f}")
    print(f"  Average per week: {signal_count / len(weekly_signals):.2f}")
    
    return signals