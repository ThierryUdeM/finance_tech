#!/usr/bin/env python3
"""
Simple Technical Scanner - TSLA Using Original Parameters
Using the original ML-optimized parameters that worked well
Converted to walk-forward compatible format
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def simple_technical_tsla(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    TSLA technical scanner using original ML-optimized parameters
    
    Using the same parameters that worked well for MSFT
    since the custom TSLA model overtraded
    """
    
    # Original ML-optimized parameters
    params = {
        'sma_fast': 8,
        'sma_slow': 24,
        'rsi_period': 12,
        'rsi_oversold': 32,
        'rsi_overbought': 68,
        'stoch_k_period': 12,
        'stoch_d_period': 3,
        'stoch_oversold': 25,
        'stoch_overbought': 75,
        'volume_sma': 16,
        'atr_period': 12,
        'bb_period': 18,
        'bb_std': 2.1,
        'macd_fast': 10,
        'macd_slow': 24,
        'macd_signal': 8
    }
    
    # Original ML-optimized weights
    weights = {
        'ma_cross': 1.8,
        'rsi': 2.2,
        'stoch': 1.2,
        'bb': 1.4,
        'macd': 1.8,
        'volume': 0.8,
        'momentum': 1.6,
        'volatility': 1.0
    }
    
    # Original thresholds
    buy_threshold = 5.2
    sell_threshold = -5.2
    
    # Combine data for indicator calculation
    combined_data = pd.concat([train_data, test_data])
    
    # Calculate indicators
    df = calculate_tsla_indicators(combined_data, params)
    
    # Get dynamic thresholds
    thresholds = get_tsla_dynamic_thresholds(df, params)
    
    # Generate signals on test data
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Get test portion
    test_start_idx = len(train_data)
    test_df = df.iloc[test_start_idx:].copy()
    test_df.index = test_data.index
    
    # Generate signals
    for i in range(max(params.values()) + 15, len(test_df)):
        signal_score = calculate_tsla_signal_score(
            df, test_start_idx + i, params, weights, thresholds
        )
        
        # Apply signal multiplier
        signal_score *= thresholds['signal_multiplier']
        
        if signal_score >= buy_threshold:
            signals.iloc[i, 0] = 1
        elif signal_score <= sell_threshold:
            signals.iloc[i, 0] = -1
    
    return signals


def calculate_tsla_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate technical indicators (original optimized version)"""
    
    # Price-based indicators with EMA
    df['SMA_fast'] = df['close'].rolling(window=params['sma_fast']).mean()
    df['SMA_slow'] = df['close'].rolling(window=params['sma_slow']).mean()
    df['EMA_fast'] = df['close'].ewm(span=params['sma_fast']).mean()
    df['EMA_slow'] = df['close'].ewm(span=params['sma_slow']).mean()
    
    # RSI with EMA smoothing
    df['RSI'] = calculate_rsi_ema(df['close'], params['rsi_period'])
    df['RSI_momentum'] = df['RSI'].diff(3)
    
    # Stochastic
    df['STOCH_K'] = calculate_stochastic_k(
        df['high'], df['low'], df['close'], params['stoch_k_period']
    )
    df['STOCH_D'] = df['STOCH_K'].rolling(window=params['stoch_d_period']).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=params['bb_period']).mean()
    bb_std = df['close'].rolling(window=params['bb_period']).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * params['bb_std'])
    df['BB_lower'] = df['BB_middle'] - (bb_std * params['bb_std'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume analysis
    df['Volume_SMA'] = df['volume'].rolling(window=params['volume_sma']).mean()
    df['Volume_EMA'] = df['volume'].ewm(span=params['volume_sma']).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_EMA']
    df['Volume_momentum'] = df['volume'].pct_change(periods=3)
    
    # ATR with normalization
    df['ATR'] = calculate_atr_ema(df, params['atr_period'])
    df['ATR_normalized'] = df['ATR'] / df['close']
    
    # MACD
    ema_fast = df['close'].ewm(span=params['macd_fast']).mean()
    ema_slow = df['close'].ewm(span=params['macd_slow']).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=params['macd_signal']).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['MACD_momentum'] = df['MACD'].diff(2)
    
    # Momentum indicators
    df['ROC'] = df['close'].pct_change(periods=8) * 100
    df['Price_momentum'] = df['close'] / df['close'].shift(6) - 1
    
    # Volatility indicators
    df['Price_volatility'] = df['close'].pct_change().rolling(window=16).std()
    df['Volatility_ratio'] = df['Price_volatility'] / df['Price_volatility'].rolling(window=48).mean()
    
    # Trend strength
    df['Higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(window=4).sum()
    df['Lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(window=4).sum()
    df['Trend_strength'] = (df['Higher_highs'] - df['Lower_lows']) / 4
    
    # Additional columns needed for dynamic thresholds
    df['Returns'] = df['close'].pct_change()
    
    return df


def get_tsla_dynamic_thresholds(df: pd.DataFrame, params: Dict) -> Dict:
    """ML-enhanced dynamic threshold calculation"""
    current_volatility = df['Volatility_ratio'].iloc[-10:].mean() if 'Volatility_ratio' in df else 1.0
    current_trend = df['Trend_strength'].iloc[-8:].mean() if 'Trend_strength' in df else 0.0
    rsi_level = df['RSI'].iloc[-1] if 'RSI' in df else 50.0
    bb_width = df['BB_width'].iloc[-1] if 'BB_width' in df else 0.02
    
    # Base thresholds
    base_oversold = params['rsi_oversold']
    base_overbought = params['rsi_overbought']
    
    # Volatility adjustment
    if current_volatility > 1.4:
        vol_adj = 8
    elif current_volatility > 1.1:
        vol_adj = 4
    elif current_volatility < 0.8:
        vol_adj = -6
    else:
        vol_adj = 0
    
    # Trend adjustment
    if current_trend > 0.4:
        trend_adj = -3
    elif current_trend < -0.4:
        trend_adj = 3
    else:
        trend_adj = 0
    
    # Bollinger Band width adjustment
    bb_adj = max(-8, min(8, (bb_width - 0.02) * 200))
    
    rsi_oversold = base_oversold + vol_adj + trend_adj + bb_adj
    rsi_overbought = base_overbought - vol_adj - trend_adj - bb_adj
    
    # Ensure reasonable bounds
    rsi_oversold = max(15, min(45, rsi_oversold))
    rsi_overbought = max(55, min(85, rsi_overbought))
    
    # Signal sensitivity multiplier
    if current_volatility > 1.3:
        signal_multiplier = 0.85
    elif current_volatility < 0.7:
        signal_multiplier = 1.15
    else:
        signal_multiplier = 1.0
    
    return {
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'volatility_ratio': current_volatility,
        'trend_strength': current_trend,
        'signal_multiplier': signal_multiplier,
        'bb_width': bb_width
    }


def calculate_tsla_signal_score(df: pd.DataFrame, idx: int, params: Dict, 
                               weights: Dict, thresholds: Dict) -> float:
    """Calculate signal score for TSLA"""
    
    if idx < 1:
        return 0
    
    score = 0
    
    # 1. Moving Average signals
    sma_signal = get_ma_signal(df, idx, 'SMA_fast', 'SMA_slow')
    ema_signal = get_ma_signal(df, idx, 'EMA_fast', 'EMA_slow')
    score += max(sma_signal, ema_signal) * weights['ma_cross']
    
    # 2. RSI signals
    rsi_signal = get_rsi_signal(df, idx, thresholds)
    rsi_momentum_signal = get_rsi_momentum_signal(df, idx)
    score += (rsi_signal + rsi_momentum_signal * 0.5) * weights['rsi']
    
    # 3. Stochastic signals
    score += get_stochastic_signal(df, idx, params) * weights['stoch']
    
    # 4. Bollinger Band signals
    score += get_bb_signal(df, idx) * weights['bb']
    
    # 5. MACD signals
    macd_signal = get_macd_signal(df, idx)
    macd_momentum = get_macd_momentum_signal(df, idx)
    score += (macd_signal + macd_momentum * 0.3) * weights['macd']
    
    # 6. Volume signals
    score += get_volume_signal(df, idx) * weights['volume']
    
    # 7. Momentum signals
    score += get_momentum_signal(df, idx) * weights['momentum']
    
    # 8. Volatility signal
    score += get_volatility_signal(df, idx, thresholds) * weights['volatility']
    
    return score


# Helper functions
def get_ma_signal(df, idx, fast_col, slow_col):
    if idx < 1:
        return 0
    
    fast_now = df[fast_col].iloc[idx]
    slow_now = df[slow_col].iloc[idx]
    fast_prev = df[fast_col].iloc[idx-1]
    slow_prev = df[slow_col].iloc[idx-1]
    
    if pd.isna(fast_now) or pd.isna(slow_now):
        return 0
    
    if fast_now > slow_now and fast_prev <= slow_prev:
        return 2.5
    elif fast_now < slow_now and fast_prev >= slow_prev:
        return -2.5
    elif fast_now > slow_now:
        return 1
    elif fast_now < slow_now:
        return -1
    else:
        return 0


def get_rsi_signal(df, idx, thresholds):
    rsi = df['RSI'].iloc[idx]
    if pd.isna(rsi):
        return 0
    
    if rsi < thresholds['rsi_oversold']:
        if rsi < thresholds['rsi_oversold'] - 10:
            return 2
        else:
            return 1
    elif rsi > thresholds['rsi_overbought']:
        if rsi > thresholds['rsi_overbought'] + 10:
            return -2
        else:
            return -1
    else:
        return 0


def get_rsi_momentum_signal(df, idx):
    rsi_momentum = df['RSI_momentum'].iloc[idx]
    if pd.isna(rsi_momentum):
        return 0
    
    if rsi_momentum > 2:
        return 1
    elif rsi_momentum < -2:
        return -1
    else:
        return 0


def get_stochastic_signal(df, idx, params):
    stoch_k = df['STOCH_K'].iloc[idx]
    stoch_d = df['STOCH_D'].iloc[idx]
    
    if pd.isna(stoch_k) or pd.isna(stoch_d):
        return 0
    
    if stoch_k < params['stoch_oversold'] and stoch_k > stoch_d:
        return 1.5
    elif stoch_k > params['stoch_overbought'] and stoch_k < stoch_d:
        return -1.5
    elif stoch_k < 15:
        return 1
    elif stoch_k > 85:
        return -1
    else:
        return 0


def get_bb_signal(df, idx):
    bb_position = df['BB_position'].iloc[idx]
    bb_width = df['BB_width'].iloc[idx]
    
    if pd.isna(bb_position) or pd.isna(bb_width):
        return 0
    
    if bb_position <= 0.05:
        if bb_width > 0.03:
            return 2
        else:
            return 1
    elif bb_position >= 0.95:
        if bb_width > 0.03:
            return -2
        else:
            return -1
    else:
        return 0


def get_macd_signal(df, idx):
    if idx < 1:
        return 0
    
    macd_hist = df['MACD_histogram'].iloc[idx]
    macd_hist_prev = df['MACD_histogram'].iloc[idx-1]
    
    if pd.isna(macd_hist) or pd.isna(macd_hist_prev):
        return 0
    
    if macd_hist > 0 and macd_hist_prev <= 0:
        return 1.5
    elif macd_hist < 0 and macd_hist_prev >= 0:
        return -1.5
    elif macd_hist > 0:
        return 0.5
    elif macd_hist < 0:
        return -0.5
    else:
        return 0


def get_macd_momentum_signal(df, idx):
    macd_momentum = df['MACD_momentum'].iloc[idx]
    if pd.isna(macd_momentum):
        return 0
    
    if macd_momentum > 0.001:
        return 1
    elif macd_momentum < -0.001:
        return -1
    else:
        return 0


def get_volume_signal(df, idx):
    volume_ratio = df['Volume_ratio'].iloc[idx]
    volume_momentum = df['Volume_momentum'].iloc[idx]
    
    if pd.isna(volume_ratio):
        return 0
    
    volume_signal = 0
    if volume_ratio > 1.8:
        volume_signal = 1.5
    elif volume_ratio > 1.3:
        volume_signal = 1
    elif volume_ratio < 0.4:
        volume_signal = -0.5
    
    if not pd.isna(volume_momentum):
        if volume_momentum > 0.3:
            volume_signal += 0.5
        elif volume_momentum < -0.3:
            volume_signal -= 0.5
    
    return volume_signal


def get_momentum_signal(df, idx):
    roc = df['ROC'].iloc[idx]
    price_momentum = df['Price_momentum'].iloc[idx]
    
    momentum_signal = 0
    
    if not pd.isna(roc):
        if roc > 2.0:
            momentum_signal += 1.5
        elif roc > 0.8:
            momentum_signal += 0.8
        elif roc < -2.0:
            momentum_signal -= 1.5
        elif roc < -0.8:
            momentum_signal -= 0.8
    
    if not pd.isna(price_momentum):
        if price_momentum > 0.015:
            momentum_signal += 0.5
        elif price_momentum < -0.015:
            momentum_signal -= 0.5
    
    return momentum_signal


def get_volatility_signal(df, idx, thresholds):
    volatility = thresholds['volatility_ratio']
    atr_norm = df['ATR_normalized'].iloc[idx]
    
    if pd.isna(atr_norm):
        return 0
    
    if volatility > 1.6:
        if atr_norm > 0.025:
            return 0.8
        else:
            return 0.4
    elif volatility < 0.6:
        return -0.3
    else:
        return 0


def calculate_rsi_ema(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic_k(high, low, close, period=14):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k_percent


def calculate_atr_ema(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr