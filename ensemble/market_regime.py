#!/usr/bin/env python3
"""
Market Regime Detection Module
Classifies market conditions as trending, ranging, or uncertain
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
    # True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    pos_dm = pd.Series(0.0, index=high.index)
    neg_dm = pd.Series(0.0, index=high.index)
    
    pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
    neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Smooth the DMs
    pos_di = 100 * (pos_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    neg_di = 100 * (neg_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx


def calculate_hurst_exponent(prices: pd.Series, window: int = 120) -> pd.Series:
    """
    Calculate rolling Hurst exponent using R/S analysis
    < 0.5: Mean reverting
    = 0.5: Random walk
    > 0.5: Trending
    """
    def hurst(ts):
        """Calculate Hurst exponent for a time series"""
        if len(ts) < 20:
            return 0.5
            
        # Create the range of lags
        lags = range(2, min(len(ts)//2, 100))
        
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0
    
    # Rolling calculation
    hurst_values = pd.Series(index=prices.index, dtype=float)
    
    for i in range(window, len(prices)):
        window_data = prices.iloc[i-window:i].values
        hurst_values.iloc[i] = hurst(window_data)
    
    # Fill initial values
    hurst_values.fillna(0.5, inplace=True)
    
    return hurst_values


def calculate_keltner_position(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """
    Calculate position relative to Keltner Channel
    Returns: 0 = inside channel, 1 = above upper, -1 = below lower
    Also returns a continuous score of how far outside
    """
    # Middle line (EMA)
    middle = df['close'].ewm(span=period, adjust=False).mean()
    
    # ATR for channel width
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    # Channel bands
    upper = middle + (atr * mult)
    lower = middle - (atr * mult)
    
    # Calculate position
    position = pd.Series(0.0, index=df.index)
    
    # Above upper band
    above_mask = df['close'] > upper
    position[above_mask] = (df['close'][above_mask] - upper[above_mask]) / atr[above_mask]
    
    # Below lower band
    below_mask = df['close'] < lower
    position[below_mask] = (df['close'][below_mask] - lower[below_mask]) / atr[below_mask]
    
    # Percentage of recent bars outside channel
    outside = (above_mask | below_mask).astype(int)
    pct_outside = outside.rolling(window=period).mean()
    
    return position, pct_outside


def calculate_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Calculate rolling z-score"""
    mean = series.rolling(window=window, min_periods=20).mean()
    std = series.rolling(window=window, min_periods=20).std()
    zscore = (series - mean) / (std + 1e-10)
    return zscore.fillna(0)


def classify_regime(df: pd.DataFrame, 
                   adx_threshold: float = 25,
                   hurst_threshold: float = 0.45,
                   keltner_threshold: float = 0.2) -> Dict:
    """
    Classify market regime based on multiple indicators
    
    Returns dict with:
    - regime_score: -1 (strong range) to +1 (strong trend)
    - regime_label: 'trending', 'ranging', 'uncertain'
    - components: individual indicator values
    """
    # Calculate indicators
    adx = calculate_adx(df['high'], df['low'], df['close'])
    hurst = calculate_hurst_exponent(df['close'])
    keltner_pos, keltner_pct = calculate_keltner_position(df)
    
    # ADX component: high ADX = trending
    adx_signal = (adx > adx_threshold).astype(float)
    adx_zscore = calculate_zscore(adx_signal)
    
    # Hurst component: low Hurst = mean reverting (ranging)
    hurst_signal = (hurst < hurst_threshold).astype(float)
    hurst_zscore = calculate_zscore(1 - hurst_signal)  # Invert for trend detection
    
    # Keltner component: high % outside = trending/breakout
    keltner_signal = (keltner_pct > keltner_threshold).astype(float)
    keltner_zscore = calculate_zscore(keltner_signal)
    
    # Composite score with weights
    composite_zscore = (
        0.4 * adx_zscore + 
        0.4 * hurst_zscore + 
        0.2 * keltner_zscore
    )
    
    # Map to [-1, 1] using tanh
    regime_score = np.tanh(composite_zscore / 3)
    
    # Classify into labels
    regime_label = pd.Series('uncertain', index=df.index)
    regime_label[regime_score > 0.3] = 'trending'
    regime_label[regime_score < -0.3] = 'ranging'
    
    # Check for ADX rising (trending strengthening)
    adx_rising = adx.diff(5) > 0
    
    return {
        'regime_score': regime_score,
        'regime_label': regime_label,
        'adx': adx,
        'adx_rising': adx_rising,
        'hurst': hurst,
        'keltner_pct': keltner_pct,
        'adx_zscore': adx_zscore,
        'hurst_zscore': hurst_zscore,
        'keltner_zscore': keltner_zscore
    }


def get_regime_weights(regime_score: float) -> Tuple[float, float]:
    """
    Calculate momentum and technical weights based on regime score
    
    Args:
        regime_score: -1 (ranging) to +1 (trending)
    
    Returns:
        (momentum_weight, technical_weight)
    """
    # Linear ramp as suggested
    w_momentum = 0.5 + 0.5 * regime_score
    w_technical = 1.0 - w_momentum
    
    # Ensure weights are in [0, 1]
    w_momentum = np.clip(w_momentum, 0.0, 1.0)
    w_technical = np.clip(w_technical, 0.0, 1.0)
    
    return w_momentum, w_technical


def smooth_regime(regime_scores: pd.Series, smoothing_window: int = 3) -> pd.Series:
    """
    Smooth regime scores to prevent excessive flipping
    Uses median filter for robustness
    """
    return regime_scores.rolling(window=smoothing_window, center=True, min_periods=1).median()