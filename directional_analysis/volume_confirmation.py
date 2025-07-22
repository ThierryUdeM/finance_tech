#!/usr/bin/env python3
"""
Volume Confirmation Module for Enhanced Pattern Matching
Adds volume analysis to improve signal quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class VolumeConfirmation:
    """Volume-based signal confirmation for pattern matching"""
    
    def __init__(self, lookback_periods: int = 20):
        """
        Initialize volume confirmation analyzer
        
        Args:
            lookback_periods: Number of periods for average volume calculation
        """
        self.lookback_periods = lookback_periods
        
    def calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various volume metrics
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional volume metrics
        """
        df = df.copy()
        
        # Average volume
        df['avg_volume'] = df['Volume'].rolling(window=self.lookback_periods).mean()
        df['volume_ratio'] = df['Volume'] / df['avg_volume']
        
        # Volume moving averages
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        
        # On-Balance Volume (OBV)
        df['price_change'] = df['Close'].diff()
        df['obv'] = (df['price_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['Volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(window=10).mean()
        
        # Volume Rate of Change
        df['volume_roc'] = df['Volume'].pct_change(periods=10) * 100
        
        # Price-Volume Trend (PVT)
        df['pvt'] = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).cumsum()
        
        # Volume-Price Confirmation Index (VPCI)
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['vwap'] = (df['typical_price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        return df
    
    def get_volume_signal_strength(self, 
                                   pattern_data: pd.DataFrame,
                                   future_direction: str,
                                   pattern_type: str = "general") -> float:
        """
        Calculate volume-based signal strength
        
        Args:
            pattern_data: DataFrame with pattern period data
            future_direction: Expected direction (BULLISH/BEARISH)
            pattern_type: Type of pattern being confirmed
            
        Returns:
            Volume confirmation score (0-1)
        """
        if len(pattern_data) < 2:
            return 0.5  # Neutral if insufficient data
        
        scores = []
        
        # 1. Volume surge on pattern completion (most recent bars)
        recent_volume_ratio = pattern_data['volume_ratio'].iloc[-3:].mean()
        if recent_volume_ratio > 1.5:
            scores.append(0.8)
        elif recent_volume_ratio > 1.2:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        # 2. Volume trend alignment
        volume_trend = pattern_data['volume_sma_5'].iloc[-1] > pattern_data['volume_sma_20'].iloc[-1]
        if future_direction == "BULLISH" and volume_trend:
            scores.append(0.7)
        elif future_direction == "BEARISH" and not volume_trend:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        # 3. OBV confirmation
        obv_rising = pattern_data['obv'].iloc[-1] > pattern_data['obv_sma'].iloc[-1]
        if (future_direction == "BULLISH" and obv_rising) or \
           (future_direction == "BEARISH" and not obv_rising):
            scores.append(0.8)
        else:
            scores.append(0.2)
        
        # 4. Volume climax detection
        max_volume_ratio = pattern_data['volume_ratio'].max()
        if max_volume_ratio > 2.0:
            # High volume climax - strong signal
            scores.append(0.9)
        elif max_volume_ratio < 0.5:
            # Very low volume - weak signal
            scores.append(0.1)
        else:
            scores.append(0.5)
        
        # 5. Price-Volume divergence check
        price_trend = pattern_data['Close'].iloc[-1] > pattern_data['Close'].iloc[0]
        volume_trend = pattern_data['Volume'].iloc[-5:].mean() > pattern_data['Volume'].iloc[:5].mean()
        
        if price_trend == volume_trend:
            # Confirmation
            scores.append(0.7)
        else:
            # Divergence - warning sign
            scores.append(0.3)
        
        # Return weighted average
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Prioritize recent volume surge
        return sum(s * w for s, w in zip(scores, weights))
    
    def adjust_confidence_with_volume(self,
                                      base_confidence: float,
                                      volume_score: float,
                                      pattern_strength: float = 1.0) -> float:
        """
        Adjust pattern confidence based on volume confirmation
        
        Args:
            base_confidence: Original confidence score
            volume_score: Volume confirmation score (0-1)
            pattern_strength: Additional pattern-specific strength
            
        Returns:
            Adjusted confidence score
        """
        # Volume weight increases with pattern strength
        volume_weight = 0.2 + (0.2 * pattern_strength)
        
        # Blend base confidence with volume score
        adjusted_confidence = (base_confidence * (1 - volume_weight)) + (volume_score * volume_weight)
        
        # Apply penalty for very weak volume
        if volume_score < 0.3:
            adjusted_confidence *= 0.8
        
        # Bonus for very strong volume confirmation
        if volume_score > 0.8:
            adjusted_confidence = min(adjusted_confidence * 1.1, 1.0)
        
        return adjusted_confidence
    
    def get_volume_filters(self) -> Dict[str, float]:
        """
        Get volume-based filtering thresholds
        
        Returns:
            Dictionary of filter thresholds
        """
        return {
            'min_volume_ratio': 0.8,      # Minimum 80% of average volume
            'max_volume_ratio': 3.0,      # Maximum 300% (avoid anomalies)
            'min_recent_volume': 0.5,     # Recent bars should have decent volume
            'obv_alignment_threshold': 0.6 # OBV should align with price
        }
    
    def create_volume_report(self, df: pd.DataFrame, signal_time: pd.Timestamp) -> Dict:
        """
        Create detailed volume analysis report for a signal
        
        Args:
            df: Full dataset with volume metrics
            signal_time: Timestamp of the signal
            
        Returns:
            Dictionary with volume analysis details
        """
        # Get data around signal time
        signal_idx = df.index.get_loc(signal_time)
        lookback = 20
        
        start_idx = max(0, signal_idx - lookback)
        end_idx = min(len(df), signal_idx + 1)
        
        signal_data = df.iloc[start_idx:end_idx]
        
        report = {
            'timestamp': signal_time,
            'current_volume': df.loc[signal_time, 'Volume'],
            'avg_volume': df.loc[signal_time, 'avg_volume'],
            'volume_ratio': df.loc[signal_time, 'volume_ratio'],
            'volume_trend': 'increasing' if signal_data['Volume'].iloc[-5:].mean() > signal_data['Volume'].iloc[:5].mean() else 'decreasing',
            'obv_signal': 'bullish' if df.loc[signal_time, 'obv'] > df.loc[signal_time, 'obv_sma'] else 'bearish',
            'volume_roc': df.loc[signal_time, 'volume_roc'],
            'unusual_volume': df.loc[signal_time, 'volume_ratio'] > 1.5,
            'volume_quality': 'high' if df.loc[signal_time, 'volume_ratio'] > 1.2 else ('low' if df.loc[signal_time, 'volume_ratio'] < 0.8 else 'normal')
        }
        
        return report


# Integration function for the pattern matcher
def enhance_pattern_matching_with_volume(df: pd.DataFrame,
                                          pattern_results: List[Dict],
                                          lookback_periods: int = 20) -> List[Dict]:
    """
    Enhance pattern matching results with volume confirmation
    
    Args:
        df: DataFrame with OHLCV data
        pattern_results: List of pattern matching results
        lookback_periods: Periods for volume average
        
    Returns:
        Enhanced pattern results with volume scores
    """
    # Initialize volume analyzer
    vol_analyzer = VolumeConfirmation(lookback_periods)
    
    # Calculate volume metrics
    df_with_volume = vol_analyzer.calculate_volume_metrics(df)
    
    # Enhance each pattern result
    enhanced_results = []
    
    for pattern in pattern_results:
        pattern_copy = pattern.copy()
        
        # Get pattern data
        pattern_start = pattern.get('pattern_start_time')
        pattern_end = pattern.get('pattern_end_time', df.index[-1])
        
        if pattern_start and pattern_end:
            pattern_data = df_with_volume.loc[pattern_start:pattern_end]
            
            # Calculate volume score
            direction = pattern.get('predicted_direction', 'NEUTRAL')
            volume_score = vol_analyzer.get_volume_signal_strength(
                pattern_data, 
                direction,
                pattern.get('pattern_type', 'general')
            )
            
            # Adjust confidence
            original_confidence = pattern.get('confidence', 0.5)
            adjusted_confidence = vol_analyzer.adjust_confidence_with_volume(
                original_confidence,
                volume_score,
                pattern.get('pattern_strength', 1.0)
            )
            
            # Add volume metrics to pattern
            pattern_copy['volume_score'] = volume_score
            pattern_copy['volume_adjusted_confidence'] = adjusted_confidence
            pattern_copy['volume_ratio'] = pattern_data['volume_ratio'].iloc[-1]
            pattern_copy['volume_quality'] = 'high' if volume_score > 0.7 else ('low' if volume_score < 0.3 else 'medium')
            
            # Create volume report
            pattern_copy['volume_report'] = vol_analyzer.create_volume_report(
                df_with_volume,
                pattern_end
            )
        
        enhanced_results.append(pattern_copy)
    
    return enhanced_results