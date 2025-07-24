#!/usr/bin/env python3
"""
Simple Technical Scanner - NVDA Specific
Optimized for high-momentum growth stock characteristics
Based on walk-forward testing showing NVDA needs momentum-focused approach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def simple_technical_nvda(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    NVDA-specific technical scanner optimized for high momentum
    
    Key adjustments for NVDA:
    - Shorter MA periods for faster signals
    - Higher weight on momentum indicators
    - Wider RSI bands for trending market
    - Reduced mean reversion signals
    """
    
    # NVDA-optimized parameters (high momentum focus)
    params = {
        'sma_fast': 5,           # Very fast for momentum
        'sma_slow': 15,          # Shorter slow MA
        'rsi_period': 10,        # Shorter RSI
        'rsi_oversold': 25,      # Lower oversold for trending
        'rsi_overbought': 75,    # Higher overbought for trending
        'stoch_k_period': 10,
        'stoch_d_period': 3,
        'stoch_oversold': 20,
        'stoch_overbought': 80,
        'volume_sma': 12,
        'atr_period': 10,
        'bb_period': 15,
        'bb_std': 2.5,           # Wider bands for volatility
        'macd_fast': 8,
        'macd_slow': 17,
        'macd_signal': 6
    }
    
    # NVDA-optimized weights (momentum-heavy)
    weights = {
        'ma_cross': 2.5,         # Higher weight for trend
        'rsi': 1.5,              # Lower weight on mean reversion
        'stoch': 1.0,
        'bb': 0.8,               # Lower BB weight
        'macd': 2.2,             # Higher MACD weight
        'volume': 1.2,
        'momentum': 2.8,         # Highest weight on momentum
        'volatility': 1.5        # Higher volatility weight
    }
    
    # Higher thresholds for fewer, higher conviction trades
    buy_threshold = 6.5
    sell_threshold = -6.5
    
    # Combine data for indicator calculation
    combined_data = pd.concat([train_data, test_data])
    
    # Calculate indicators
    df = calculate_nvda_indicators(combined_data, params)
    
    # Generate signals on test data
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Get test portion
    test_start_idx = len(train_data)
    test_df = df.iloc[test_start_idx:].copy()
    test_df.index = test_data.index
    
    # Generate signals
    for i in range(20, len(test_df)):  # Need history for indicators
        signal_score = calculate_nvda_signal_score(
            df, test_start_idx + i, params, weights
        )
        
        if signal_score >= buy_threshold:
            signals.iloc[i, 0] = 1
        elif signal_score <= sell_threshold:
            signals.iloc[i, 0] = -1
    
    return signals


def calculate_nvda_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate NVDA-optimized indicators"""
    
    # Price-based indicators
    df['SMA_fast'] = df['close'].rolling(window=params['sma_fast']).mean()
    df['SMA_slow'] = df['close'].rolling(window=params['sma_slow']).mean()
    df['EMA_fast'] = df['close'].ewm(span=params['sma_fast']).mean()
    df['EMA_slow'] = df['close'].ewm(span=params['sma_slow']).mean()
    
    # RSI with momentum bias
    df['RSI'] = calculate_rsi_momentum(df['close'], params['rsi_period'])
    df['RSI_slope'] = df['RSI'].diff(3)  # RSI momentum
    
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
    df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
    df['Volume_surge'] = (df['Volume_ratio'] > 1.5).astype(int)
    
    # ATR for volatility
    df['ATR'] = calculate_atr(df, params['atr_period'])
    df['ATR_pct'] = df['ATR'] / df['close']
    
    # MACD with momentum focus
    ema_fast = df['close'].ewm(span=params['macd_fast']).mean()
    ema_slow = df['close'].ewm(span=params['macd_slow']).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=params['macd_signal']).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['MACD_slope'] = df['MACD'].diff(2)
    
    # Enhanced momentum indicators for NVDA
    df['ROC_5'] = df['close'].pct_change(periods=5) * 100
    df['ROC_10'] = df['close'].pct_change(periods=10) * 100
    df['Momentum_strength'] = (df['ROC_5'] + df['ROC_10']) / 2
    
    # Trend strength
    df['Higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(window=5).sum()
    df['Higher_lows'] = (df['low'] > df['low'].shift(1)).rolling(window=5).sum()
    df['Trend_score'] = (df['Higher_highs'] + df['Higher_lows']) / 10
    
    # Volatility expansion
    df['Volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['Vol_expansion'] = df['Volatility'] / df['Volatility'].rolling(window=50).mean()
    
    return df


def calculate_nvda_signal_score(df: pd.DataFrame, idx: int, 
                                params: Dict, weights: Dict) -> float:
    """Calculate NVDA-specific signal score"""
    
    if idx < 1:
        return 0
    
    score = 0
    current_idx = idx
    
    # 1. MA Cross (trend following)
    ma_fast = df['EMA_fast'].iloc[idx]
    ma_slow = df['EMA_slow'].iloc[idx]
    ma_fast_prev = df['EMA_fast'].iloc[idx-1]
    ma_slow_prev = df['EMA_slow'].iloc[idx-1]
    
    if not pd.isna(ma_fast) and not pd.isna(ma_slow):
        # Strong weight on fresh crosses
        if ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev:
            score += 3.0 * weights['ma_cross']
        elif ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev:
            score -= 3.0 * weights['ma_cross']
        elif ma_fast > ma_slow:
            # Trend continuation bonus
            score += 1.5 * weights['ma_cross']
        else:
            score -= 1.5 * weights['ma_cross']
    
    # 2. RSI (reduced mean reversion for trending stock)
    rsi = df['RSI'].iloc[idx]
    rsi_slope = df['RSI_slope'].iloc[idx]
    
    if not pd.isna(rsi):
        # Only extreme oversold/overbought
        if rsi < params['rsi_oversold'] and rsi_slope > 0:
            score += 1.5 * weights['rsi']
        elif rsi > params['rsi_overbought'] and rsi_slope < 0:
            score -= 1.5 * weights['rsi']
        # Momentum continuation
        elif 40 < rsi < 60 and rsi_slope > 1:
            score += 0.5 * weights['rsi']
    
    # 3. Stochastic (momentum focused)
    stoch_k = df['STOCH_K'].iloc[idx]
    stoch_d = df['STOCH_D'].iloc[idx]
    
    if not pd.isna(stoch_k) and not pd.isna(stoch_d):
        if stoch_k > stoch_d and stoch_k < 30:
            score += 1.0 * weights['stoch']
        elif stoch_k < stoch_d and stoch_k > 70:
            score -= 1.0 * weights['stoch']
    
    # 4. Bollinger Bands (breakout focused)
    bb_position = df['BB_position'].iloc[idx]
    bb_width = df['BB_width'].iloc[idx]
    
    if not pd.isna(bb_position) and not pd.isna(bb_width):
        # Breakout signals
        if bb_position > 1.0:  # Above upper band
            score += 1.5 * weights['bb']
        elif bb_position < 0.0:  # Below lower band
            score -= 1.5 * weights['bb']
        # Squeeze play
        elif bb_width < 0.02 and 0.4 < bb_position < 0.6:
            score += 0.5 * weights['bb']
    
    # 5. MACD (momentum)
    macd_hist = df['MACD_histogram'].iloc[idx]
    macd_hist_prev = df['MACD_histogram'].iloc[idx-1]
    macd_slope = df['MACD_slope'].iloc[idx]
    
    if not pd.isna(macd_hist):
        # Histogram crossover
        if macd_hist > 0 and macd_hist_prev <= 0:
            score += 2.0 * weights['macd']
        elif macd_hist < 0 and macd_hist_prev >= 0:
            score -= 2.0 * weights['macd']
        # Momentum acceleration
        elif macd_slope > 0.001:
            score += 1.0 * weights['macd']
        elif macd_slope < -0.001:
            score -= 1.0 * weights['macd']
    
    # 6. Volume confirmation
    volume_ratio = df['Volume_ratio'].iloc[idx]
    volume_surge = df['Volume_surge'].iloc[idx]
    
    if not pd.isna(volume_ratio):
        if volume_surge and score > 0:
            score += 1.5 * weights['volume']
        elif volume_ratio < 0.5:
            score *= 0.7  # Reduce score on low volume
    
    # 7. Momentum (highest weight for NVDA)
    momentum = df['Momentum_strength'].iloc[idx]
    trend_score = df['Trend_score'].iloc[idx]
    
    if not pd.isna(momentum):
        if momentum > 3.0:
            score += 2.5 * weights['momentum']
        elif momentum > 1.5:
            score += 1.5 * weights['momentum']
        elif momentum < -3.0:
            score -= 2.5 * weights['momentum']
        elif momentum < -1.5:
            score -= 1.5 * weights['momentum']
        
        # Trend bonus
        if not pd.isna(trend_score):
            if trend_score > 0.7:
                score += 1.0 * weights['momentum']
            elif trend_score < 0.3:
                score -= 1.0 * weights['momentum']
    
    # 8. Volatility (expansion favored)
    vol_expansion = df['Vol_expansion'].iloc[idx]
    
    if not pd.isna(vol_expansion):
        if vol_expansion > 1.3:
            score += 1.0 * weights['volatility']
        elif vol_expansion > 1.1:
            score += 0.5 * weights['volatility']
    
    return score


def calculate_rsi_momentum(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI calculation with momentum bias"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use EMA for more responsive RSI
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic_k(high: pd.Series, low: pd.Series, 
                           close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Stochastic %K"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return k_percent


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr