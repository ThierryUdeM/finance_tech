#!/usr/bin/env python3
"""
Simple Technical Scanner - AAPL Specific
Optimized for stable, quality stock characteristics
Based on walk-forward testing showing AAPL needs conservative approach
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def simple_technical_aapl(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    AAPL-specific technical scanner optimized for stable quality stock
    
    Key adjustments for AAPL:
    - Longer MA periods for stability
    - Higher weight on mean reversion
    - Stricter RSI bands for quality signals
    - Focus on low volatility environments
    """
    
    # AAPL-optimized parameters (conservative approach)
    params = {
        'sma_fast': 12,          # Longer than momentum stocks
        'sma_slow': 26,          # Standard slow MA
        'rsi_period': 14,        # Standard RSI
        'rsi_oversold': 35,      # Tighter bands
        'rsi_overbought': 65,    # Tighter bands
        'stoch_k_period': 14,
        'stoch_d_period': 3,
        'stoch_oversold': 30,
        'stoch_overbought': 70,
        'volume_sma': 20,
        'atr_period': 14,
        'bb_period': 20,         # Standard BB
        'bb_std': 2.0,           # Standard deviation
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }
    
    # AAPL-optimized weights (mean reversion focus)
    weights = {
        'ma_cross': 1.5,         # Lower weight on trend
        'rsi': 2.5,              # Higher weight on mean reversion
        'stoch': 1.8,            # Higher stochastic weight
        'bb': 2.0,               # Higher BB weight for ranges
        'macd': 1.5,
        'volume': 0.6,           # Lower volume weight
        'momentum': 1.0,         # Lower momentum weight
        'volatility': -0.5       # Negative weight - prefer low vol
    }
    
    # Conservative thresholds for quality over quantity
    buy_threshold = 6.0
    sell_threshold = -6.0
    
    # Combine data for indicator calculation
    combined_data = pd.concat([train_data, test_data])
    
    # Calculate indicators
    df = calculate_aapl_indicators(combined_data, params)
    
    # Generate signals on test data
    signals = pd.DataFrame(index=test_data.index)
    signals['signal'] = 0
    
    # Get test portion
    test_start_idx = len(train_data)
    test_df = df.iloc[test_start_idx:].copy()
    test_df.index = test_data.index
    
    # Track daily signals for AAPL (max 1 per day)
    daily_signals = {}
    
    # Generate signals
    for i in range(20, len(test_df)):  # Need history for indicators
        current_date = test_df.index[i].date()
        
        # Skip if already have signal today
        if current_date in daily_signals and daily_signals[current_date] >= 1:
            continue
        
        signal_score = calculate_aapl_signal_score(
            df, test_start_idx + i, params, weights
        )
        
        # Additional filters for AAPL
        if is_quality_setup(df, test_start_idx + i, params):
            if signal_score >= buy_threshold:
                signals.iloc[i, 0] = 1
                daily_signals[current_date] = daily_signals.get(current_date, 0) + 1
            elif signal_score <= sell_threshold:
                signals.iloc[i, 0] = -1
                daily_signals[current_date] = daily_signals.get(current_date, 0) + 1
    
    return signals


def calculate_aapl_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate AAPL-optimized indicators"""
    
    # Price-based indicators
    df['SMA_fast'] = df['close'].rolling(window=params['sma_fast']).mean()
    df['SMA_slow'] = df['close'].rolling(window=params['sma_slow']).mean()
    df['EMA_fast'] = df['close'].ewm(span=params['sma_fast']).mean()
    df['EMA_slow'] = df['close'].ewm(span=params['sma_slow']).mean()
    
    # Distance from moving averages (mean reversion)
    df['Distance_SMA'] = (df['close'] - df['SMA_slow']) / df['SMA_slow'] * 100
    
    # RSI
    df['RSI'] = calculate_rsi(df['close'], params['rsi_period'])
    df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
    
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
    
    # Mean reversion indicator
    df['BB_zscore'] = (df['close'] - df['BB_middle']) / bb_std
    
    # Volume analysis
    df['Volume_SMA'] = df['volume'].rolling(window=params['volume_sma']).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_SMA']
    
    # ATR for volatility
    df['ATR'] = calculate_atr(df, params['atr_period'])
    df['ATR_pct'] = df['ATR'] / df['close']
    
    # Volatility regime
    df['Volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['Vol_percentile'] = df['Volatility'].rolling(window=100).rank(pct=True)
    
    # MACD
    ema_fast = df['close'].ewm(span=params['macd_fast']).mean()
    ema_slow = df['close'].ewm(span=params['macd_slow']).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=params['macd_signal']).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Support/Resistance levels
    df['Resistance'] = df['high'].rolling(window=20).max()
    df['Support'] = df['low'].rolling(window=20).min()
    df['SR_position'] = (df['close'] - df['Support']) / (df['Resistance'] - df['Support'])
    
    # Quality momentum (slow and steady)
    df['ROC_20'] = df['close'].pct_change(periods=20) * 100
    df['Quality_momentum'] = df['ROC_20'].rolling(window=5).mean()
    
    return df


def calculate_aapl_signal_score(df: pd.DataFrame, idx: int, 
                                params: Dict, weights: Dict) -> float:
    """Calculate AAPL-specific signal score"""
    
    if idx < 1:
        return 0
    
    score = 0
    current_idx = idx
    
    # 1. MA Cross (reduced weight for AAPL)
    ma_fast = df['SMA_fast'].iloc[idx]
    ma_slow = df['SMA_slow'].iloc[idx]
    distance_sma = df['Distance_SMA'].iloc[idx]
    
    if not pd.isna(ma_fast) and not pd.isna(ma_slow):
        if ma_fast > ma_slow:
            score += 1.0 * weights['ma_cross']
        else:
            score -= 1.0 * weights['ma_cross']
        
        # Mean reversion from MA
        if not pd.isna(distance_sma):
            if distance_sma < -2.0:  # 2% below MA
                score += 1.5 * weights['ma_cross']
            elif distance_sma > 2.0:  # 2% above MA
                score -= 1.5 * weights['ma_cross']
    
    # 2. RSI (strong mean reversion)
    rsi = df['RSI'].iloc[idx]
    rsi_ma = df['RSI_MA'].iloc[idx]
    
    if not pd.isna(rsi):
        # Oversold/overbought
        if rsi < params['rsi_oversold']:
            if rsi < 30:  # Very oversold
                score += 2.5 * weights['rsi']
            else:
                score += 1.5 * weights['rsi']
        elif rsi > params['rsi_overbought']:
            if rsi > 70:  # Very overbought
                score -= 2.5 * weights['rsi']
            else:
                score -= 1.5 * weights['rsi']
        
        # RSI divergence
        if not pd.isna(rsi_ma) and rsi < rsi_ma and rsi < 40:
            score += 0.5 * weights['rsi']
    
    # 3. Stochastic (mean reversion)
    stoch_k = df['STOCH_K'].iloc[idx]
    stoch_d = df['STOCH_D'].iloc[idx]
    
    if not pd.isna(stoch_k) and not pd.isna(stoch_d):
        if stoch_k < params['stoch_oversold']:
            score += 1.5 * weights['stoch']
        elif stoch_k > params['stoch_overbought']:
            score -= 1.5 * weights['stoch']
        
        # Stochastic divergence
        if stoch_k > stoch_d and stoch_k < 30:
            score += 1.0 * weights['stoch']
        elif stoch_k < stoch_d and stoch_k > 70:
            score -= 1.0 * weights['stoch']
    
    # 4. Bollinger Bands (mean reversion primary)
    bb_position = df['BB_position'].iloc[idx]
    bb_zscore = df['BB_zscore'].iloc[idx]
    bb_width = df['BB_width'].iloc[idx]
    
    if not pd.isna(bb_position) and not pd.isna(bb_zscore):
        # Strong mean reversion signals
        if bb_position < 0.1:  # Near/below lower band
            score += 2.0 * weights['bb']
        elif bb_position > 0.9:  # Near/above upper band
            score -= 2.0 * weights['bb']
        
        # Z-score extreme
        if abs(bb_zscore) > 2.0:
            score += 1.0 * weights['bb'] * (-np.sign(bb_zscore))
        
        # Prefer normal width bands
        if not pd.isna(bb_width) and 0.015 < bb_width < 0.03:
            score += 0.5 * weights['bb']
    
    # 5. MACD (conservative signals)
    macd_hist = df['MACD_histogram'].iloc[idx]
    macd_hist_prev = df['MACD_histogram'].iloc[idx-1]
    
    if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev):
        # Only fresh crosses
        if macd_hist > 0 and macd_hist_prev <= 0:
            score += 1.5 * weights['macd']
        elif macd_hist < 0 and macd_hist_prev >= 0:
            score -= 1.5 * weights['macd']
    
    # 6. Volume (less important for AAPL)
    volume_ratio = df['Volume_ratio'].iloc[idx]
    
    if not pd.isna(volume_ratio):
        if 0.8 < volume_ratio < 1.5:  # Normal volume preferred
            score += 0.5 * weights['volume']
        elif volume_ratio > 2.0:  # High volume caution
            score *= 0.8
    
    # 7. Quality momentum (slow and steady)
    quality_mom = df['Quality_momentum'].iloc[idx]
    sr_position = df['SR_position'].iloc[idx]
    
    if not pd.isna(quality_mom):
        # Prefer modest momentum
        if 0.5 < quality_mom < 2.0:
            score += 1.0 * weights['momentum']
        elif -2.0 < quality_mom < -0.5:
            score -= 1.0 * weights['momentum']
        
        # Support/resistance
        if not pd.isna(sr_position):
            if sr_position < 0.2:  # Near support
                score += 0.5 * weights['momentum']
            elif sr_position > 0.8:  # Near resistance
                score -= 0.5 * weights['momentum']
    
    # 8. Volatility (prefer low volatility)
    vol_percentile = df['Vol_percentile'].iloc[idx]
    
    if not pd.isna(vol_percentile):
        if vol_percentile < 0.3:  # Low volatility
            score -= 1.0 * weights['volatility']  # Negative weight
        elif vol_percentile > 0.7:  # High volatility
            score -= 1.5 * weights['volatility']  # Even more negative
    
    return score


def is_quality_setup(df: pd.DataFrame, idx: int, params: Dict) -> bool:
    """Additional quality filters for AAPL"""
    
    # Check volatility regime
    vol_percentile = df['Vol_percentile'].iloc[idx]
    if not pd.isna(vol_percentile) and vol_percentile > 0.8:
        return False  # Skip high volatility periods
    
    # Check if price is in reasonable range
    distance_sma = df['Distance_SMA'].iloc[idx]
    if not pd.isna(distance_sma) and abs(distance_sma) > 5.0:
        return False  # Skip when too far from MA
    
    # Ensure sufficient liquidity
    volume_ratio = df['Volume_ratio'].iloc[idx]
    if not pd.isna(volume_ratio) and volume_ratio < 0.3:
        return False  # Skip very low volume
    
    return True


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
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
    atr = true_range.rolling(window=period).mean()
    return atr