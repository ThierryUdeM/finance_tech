#!/usr/bin/env python3
"""
Enhanced Pattern Scanner with Volume Validation and Dynamic Thresholds
Addresses: Volume patterns, ATR-based thresholds, VWAP, and look-ahead bias prevention
"""

import numpy as np
import pandas as pd
import talib
from datetime import datetime
import logging
import sys
import os

# Add the signal directory to path for volume_confirmation import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'signal', 'directional_analysis'))

try:
    from volume_confirmation import VolumeConfirmation
    VOLUME_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced volume confirmation module not available")
    VOLUME_MODULE_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedPatternAnalyzer:
    def __init__(self):
        self.min_volume_samples = 20  # Minimum bars for volume analysis
        if VOLUME_MODULE_AVAILABLE:
            self.volume_analyzer = VolumeConfirmation()
        
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range for dynamic thresholds"""
        return talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=period)
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def calculate_volatility_threshold(self, data, base_multiplier=0.25):
        """Calculate dynamic threshold based on ATR"""
        atr = self.calculate_atr(data)
        current_atr = atr[-1] if len(atr) > 0 and not np.isnan(atr[-1]) else None
        
        if current_atr:
            # Scale threshold by ATR as percentage of price
            current_price = data['Close'].iloc[-1]
            threshold_pct = (current_atr / current_price) * 100 * base_multiplier
            return max(threshold_pct, 0.1)  # Minimum 0.1%
        else:
            # Fallback to standard deviation
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                daily_vol = returns.std() * np.sqrt(78)  # 78 five-min bars in a day
                return max(daily_vol * 100 * base_multiplier, 0.1)
            return 0.5  # Default fallback
    
    def validate_head_shoulders_volume(self, data, pattern_indices):
        """
        Validate Head & Shoulders volume pattern:
        - Volume should be highest at the head
        - Lower volume on right shoulder vs left shoulder
        - Increase on breakout
        """
        if len(pattern_indices) < 5:  # Need at least 5 points for H&S
            return False, "Insufficient pattern points"
        
        try:
            # Identify pattern components (simplified)
            left_shoulder_idx = pattern_indices[0]
            head_idx = pattern_indices[len(pattern_indices)//2]
            right_shoulder_idx = pattern_indices[-1]
            
            # Get volumes
            left_vol = data['Volume'].iloc[left_shoulder_idx]
            head_vol = data['Volume'].iloc[head_idx]
            right_vol = data['Volume'].iloc[right_shoulder_idx]
            avg_vol = data['Volume'].mean()
            
            # Validation rules
            checks = {
                'head_volume_high': head_vol > avg_vol * 1.2,
                'right_shoulder_lower': right_vol < left_vol * 0.9,
                'volume_declining': right_vol < avg_vol
            }
            
            passed = sum(checks.values())
            reliability = passed / len(checks)
            
            return reliability > 0.6, {
                'reliability_score': reliability,
                'volume_pattern': 'valid' if reliability > 0.6 else 'weak',
                'checks': checks
            }
            
        except Exception as e:
            logger.error(f"Volume validation error: {e}")
            return False, "Validation error"
    
    def validate_double_top_volume(self, data, pattern_indices):
        """
        Validate Double Top volume pattern:
        - Second peak should have lower volume than first
        - Volume should increase on breakdown
        """
        if len(pattern_indices) < 2:
            return False, "Insufficient pattern points"
        
        try:
            first_peak_idx = pattern_indices[0]
            second_peak_idx = pattern_indices[-1]
            
            first_vol = data['Volume'].iloc[first_peak_idx]
            second_vol = data['Volume'].iloc[second_peak_idx]
            avg_vol = data['Volume'].mean()
            
            checks = {
                'second_peak_lower_volume': second_vol < first_vol * 0.85,
                'first_peak_above_average': first_vol > avg_vol,
                'volume_divergence': second_vol < avg_vol * 0.9
            }
            
            passed = sum(checks.values())
            reliability = passed / len(checks)
            
            return reliability > 0.5, {
                'reliability_score': reliability,
                'volume_pattern': 'valid' if reliability > 0.5 else 'weak',
                'checks': checks
            }
            
        except Exception as e:
            logger.error(f"Volume validation error: {e}")
            return False, "Validation error"
    
    def get_comprehensive_volume_score(self, data, pattern_info):
        """Get comprehensive volume score using both built-in and enhanced methods"""
        if not VOLUME_MODULE_AVAILABLE:
            # Fallback to basic volume validation
            return self.get_basic_volume_score(data, pattern_info)
        
        # Use enhanced volume confirmation
        pattern_start = pattern_info.get('start_idx', max(0, len(data) - 50))
        pattern_end = pattern_info.get('end_idx', len(data) - 1)
        
        # Get pattern data
        pattern_data = data.iloc[pattern_start:pattern_end + 1].copy()
        
        # Calculate volume metrics
        pattern_data = self.volume_analyzer.calculate_volume_metrics(pattern_data)
        
        # Get pattern direction
        pattern_type = pattern_info.get('pattern_type', '').lower()
        if 'bottom' in pattern_type or 'inverse' in pattern_type:
            direction = 'BULLISH'
        elif 'top' in pattern_type or 'head' in pattern_type:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Get volume score
        volume_score = self.volume_analyzer.get_volume_signal_strength(
            pattern_data,
            direction,
            pattern_type
        )
        
        return volume_score
    
    def get_basic_volume_score(self, data, pattern_info):
        """Basic volume scoring when enhanced module not available"""
        pattern_idx = pattern_info.get('end_idx', len(data) - 1)
        
        # Recent volume vs average
        recent_volume = data['Volume'].iloc[-5:].mean()
        avg_volume = data['Volume'].iloc[-20:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Score based on ratio
        if volume_ratio > 1.5:
            return 0.8
        elif volume_ratio > 1.2:
            return 0.6
        elif volume_ratio < 0.5:
            return 0.2
        else:
            return 0.5
    
    def enhance_pattern_data(self, data, patterns):
        """Add enhanced metrics to patterns"""
        # Calculate metrics
        atr = self.calculate_atr(data)
        vwap = self.calculate_vwap(data)
        threshold = self.calculate_volatility_threshold(data)
        
        # Calculate volume metrics if available
        if VOLUME_MODULE_AVAILABLE and 'Volume' in data.columns:
            data = self.volume_analyzer.calculate_volume_metrics(data)
        
        enhanced_patterns = []
        
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern['timestamp'])
            
            # Add market context
            pattern['atr'] = atr[pattern_idx] if pattern_idx < len(atr) else None
            pattern['vwap'] = vwap.iloc[pattern_idx] if pattern_idx < len(vwap) else None
            pattern['vwap_deviation'] = ((pattern['price'] - pattern['vwap']) / pattern['vwap'] * 100) if pattern['vwap'] else None
            pattern['volatility_threshold'] = threshold
            
            # Volume validation for chart patterns
            if pattern['pattern_type'] == 'chart':
                volume_valid = False
                volume_info = {}
                
                # Get pattern indices (simplified - in real implementation, get from pattern detector)
                pattern_window = 20  # Look back 20 bars
                start_idx = max(0, pattern_idx - pattern_window)
                pattern_indices = list(range(start_idx, pattern_idx + 1))
                
                if 'Head and Shoulder' in pattern['pattern_name']:
                    volume_valid, volume_info = self.validate_head_shoulders_volume(
                        data.iloc[start_idx:pattern_idx+1], 
                        list(range(len(pattern_indices)))
                    )
                elif 'Double Top' in pattern['pattern_name'] or 'Double Bottom' in pattern['pattern_name']:
                    volume_valid, volume_info = self.validate_double_top_volume(
                        data.iloc[start_idx:pattern_idx+1],
                        list(range(len(pattern_indices)))
                    )
                
                pattern['volume_validated'] = volume_valid
                pattern['volume_info'] = volume_info
                
                # Adjust confidence based on volume
                if volume_valid and isinstance(volume_info, dict):
                    reliability_boost = volume_info.get('reliability_score', 0) * 10
                    pattern['adjusted_confidence'] = min(pattern['confidence'] + reliability_boost, 100)
                else:
                    pattern['adjusted_confidence'] = pattern['confidence'] * 0.8  # Reduce confidence
            else:
                pattern['adjusted_confidence'] = pattern['confidence']
            
            enhanced_patterns.append(pattern)
        
        return enhanced_patterns

class EnhancedPatternEvaluator:
    """Enhanced evaluator with look-ahead bias prevention"""
    
    def evaluate_pattern_safe(self, pattern, price_data, horizon_minutes=60):
        """Evaluate pattern performance with strict look-ahead prevention"""
        pattern_time = pattern['timestamp']
        pattern_idx = None
        
        # Find exact pattern index
        try:
            pattern_idx = price_data.index.get_loc(pattern_time)
        except KeyError:
            # Find nearest index
            time_diffs = abs(price_data.index - pattern_time)
            pattern_idx = time_diffs.argmin()
        
        # Calculate horizon in bars (5-minute bars)
        horizon_bars = horizon_minutes // 5
        
        # Strict boundary check
        end_idx = pattern_idx + horizon_bars
        if end_idx >= len(price_data):
            return {
                'success': None,
                'price_change': None,
                'reason': 'Insufficient future data',
                'bars_available': len(price_data) - pattern_idx - 1,
                'bars_needed': horizon_bars
            }
        
        # Get entry and exit prices
        entry_price = price_data['Close'].iloc[pattern_idx]
        future_prices = price_data.iloc[pattern_idx + 1:end_idx + 1]  # Exclude entry bar
        
        if len(future_prices) == 0:
            return {'success': None, 'price_change': None, 'reason': 'No future data'}
        
        # Calculate price change
        exit_price = future_prices['Close'].iloc[-1]
        price_change_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Use dynamic threshold
        threshold = pattern.get('volatility_threshold', 0.5)
        
        # Determine success based on pattern direction and threshold
        if pattern['direction'] == 'bullish':
            success = price_change_pct > threshold
            max_favorable = ((future_prices['High'].max() - entry_price) / entry_price) * 100
            max_adverse = ((entry_price - future_prices['Low'].min()) / entry_price) * 100
        elif pattern['direction'] == 'bearish':
            success = price_change_pct < -threshold
            max_favorable = ((entry_price - future_prices['Low'].min()) / entry_price) * 100
            max_adverse = ((future_prices['High'].max() - entry_price) / entry_price) * 100
        else:  # neutral
            success = abs(price_change_pct) <= threshold
            max_favorable = abs(price_change_pct)
            max_adverse = 0
        
        return {
            'success': success,
            'price_change': round(price_change_pct, 3),
            'threshold_used': round(threshold, 3),
            'max_favorable': round(max_favorable, 3),
            'max_adverse': round(max_adverse, 3),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'horizon_minutes': horizon_minutes,
            'pattern_idx': pattern_idx,
            'exit_idx': end_idx
        }

def calculate_signal_quality_score(pattern, market_data):
    """Calculate comprehensive signal quality score"""
    score = 0
    max_score = 100
    
    # Base confidence (30 points)
    confidence_score = (pattern.get('adjusted_confidence', pattern['confidence']) / 100) * 30
    score += confidence_score
    
    # Volume validation (25 points)
    if pattern.get('volume_validated', False):
        volume_score = pattern.get('volume_info', {}).get('reliability_score', 0) * 25
        score += volume_score
    elif pattern['pattern_type'] == 'candlestick':
        # Candlestick patterns get partial credit
        score += 15
    
    # VWAP alignment (20 points)
    vwap_dev = pattern.get('vwap_deviation', 0)
    if vwap_dev is not None:
        if pattern['direction'] == 'bullish' and vwap_dev < 0:  # Below VWAP, good for longs
            score += min(abs(vwap_dev) * 4, 20)
        elif pattern['direction'] == 'bearish' and vwap_dev > 0:  # Above VWAP, good for shorts
            score += min(vwap_dev * 4, 20)
    
    # ATR context (15 points) - Higher volatility = stronger signals needed
    atr_ratio = pattern.get('atr', 0) / market_data.get('current_price', 1) * 100
    if atr_ratio > 2:  # High volatility environment
        score += 15
    elif atr_ratio > 1:
        score += 10
    else:
        score += 5
    
    # Pattern rarity (10 points) - Less common patterns score higher
    rare_patterns = ['Three Black Crows', 'Three White Soldiers', 'Morning Star', 'Evening Star']
    if pattern['pattern_name'] in rare_patterns:
        score += 10
    elif pattern['pattern_type'] == 'chart':
        score += 7
    else:
        score += 3
    
    return min(score, max_score)