#!/usr/bin/env python3
"""
Volume-Enhanced Pattern Scanner
Adds comprehensive volume confirmation to pattern detection
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add path for volume confirmation module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'signal', 'directional_analysis'))

try:
    from volume_confirmation import VolumeConfirmation
    VOLUME_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: Volume confirmation module not available")
    VOLUME_MODULE_AVAILABLE = False

def enhance_patterns_with_volume(data, patterns, lookback=20):
    """
    Enhance detected patterns with volume confirmation
    
    Args:
        data: DataFrame with OHLCV data
        patterns: List of detected patterns
        lookback: Number of bars to analyze for volume
        
    Returns:
        List of patterns with volume scores and adjusted confidence
    """
    if not VOLUME_MODULE_AVAILABLE:
        print("Volume module not available, returning original patterns")
        return patterns
    
    # Initialize volume analyzer
    vol_analyzer = VolumeConfirmation(lookback)
    
    # Calculate volume metrics for the entire dataset
    data_with_volume = vol_analyzer.calculate_volume_metrics(data)
    
    enhanced_patterns = []
    
    for pattern in patterns:
        pattern_copy = pattern.copy()
        
        # Get pattern timestamp and find index
        pattern_time = pattern.get('timestamp')
        if pattern_time is None:
            enhanced_patterns.append(pattern_copy)
            continue
            
        try:
            pattern_idx = data.index.get_loc(pattern_time)
        except KeyError:
            # Find nearest index
            time_diffs = abs(data.index - pattern_time)
            pattern_idx = time_diffs.argmin()
        
        # Define pattern window
        start_idx = max(0, pattern_idx - lookback)
        end_idx = min(len(data) - 1, pattern_idx)
        
        # Get pattern data
        pattern_data = data_with_volume.iloc[start_idx:end_idx + 1]
        
        # Determine pattern direction
        pattern_type = pattern.get('pattern', '').lower()
        if 'bottom' in pattern_type or 'inverse' in pattern_type or 'bullish' in pattern_type:
            direction = 'BULLISH'
        elif 'top' in pattern_type or 'head' in pattern_type or 'bearish' in pattern_type:
            direction = 'BEARISH'
        else:
            # Use action if available
            action = pattern.get('action', '').upper()
            if action in ['BUY', 'LONG']:
                direction = 'BULLISH'
            elif action in ['SELL', 'SHORT']:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
        
        # Calculate volume score
        volume_score = vol_analyzer.get_volume_signal_strength(
            pattern_data,
            direction,
            pattern_type
        )
        
        # Adjust confidence with volume
        original_confidence = pattern.get('confidence', 50) / 100.0
        adjusted_confidence = vol_analyzer.adjust_confidence_with_volume(
            original_confidence,
            volume_score,
            pattern_strength=0.85  # Technical patterns are generally reliable
        )
        
        # Add volume metrics to pattern
        pattern_copy['volume_score'] = round(volume_score, 3)
        pattern_copy['volume_adjusted_confidence'] = round(adjusted_confidence * 100, 1)
        pattern_copy['original_confidence'] = pattern.get('confidence', 50)
        pattern_copy['volume_quality'] = 'high' if volume_score > 0.7 else ('low' if volume_score < 0.3 else 'medium')
        
        # Add current volume metrics
        if pattern_idx < len(data_with_volume):
            current_data = data_with_volume.iloc[pattern_idx]
            pattern_copy['current_volume'] = int(current_data['Volume'])
            pattern_copy['avg_volume'] = int(current_data.get('avg_volume', 0))
            pattern_copy['volume_ratio'] = round(current_data.get('volume_ratio', 1.0), 2)
            pattern_copy['volume_trend'] = 'increasing' if pattern_data['Volume'].iloc[-5:].mean() > pattern_data['Volume'].iloc[:5].mean() else 'decreasing'
            
            # Add OBV signal if available
            if 'obv' in current_data and 'obv_sma' in current_data:
                pattern_copy['obv_signal'] = 'bullish' if current_data['obv'] > current_data['obv_sma'] else 'bearish'
        
        # Create volume report
        volume_report = vol_analyzer.create_volume_report(data_with_volume, pattern_time)
        pattern_copy['volume_report'] = volume_report
        
        # Determine if pattern should be filtered
        if volume_score < 0.2:
            pattern_copy['volume_warning'] = 'Very low volume - pattern may be unreliable'
        elif volume_score > 0.8:
            pattern_copy['volume_confirmation'] = 'Strong volume confirmation'
        
        enhanced_patterns.append(pattern_copy)
    
    return enhanced_patterns


def filter_patterns_by_volume(patterns, min_volume_score=0.3, min_confidence=40):
    """
    Filter patterns based on volume criteria
    
    Args:
        patterns: List of volume-enhanced patterns
        min_volume_score: Minimum volume score (0-1)
        min_confidence: Minimum adjusted confidence (0-100)
        
    Returns:
        Filtered list of high-quality patterns
    """
    filtered = []
    
    for pattern in patterns:
        volume_score = pattern.get('volume_score', 0.5)
        adjusted_confidence = pattern.get('volume_adjusted_confidence', pattern.get('confidence', 50))
        
        if volume_score >= min_volume_score and adjusted_confidence >= min_confidence:
            filtered.append(pattern)
    
    return filtered


def generate_volume_summary(patterns):
    """
    Generate summary statistics for volume-enhanced patterns
    
    Args:
        patterns: List of volume-enhanced patterns
        
    Returns:
        Dictionary with summary statistics
    """
    if not patterns:
        return {}
    
    volume_scores = [p.get('volume_score', 0) for p in patterns]
    confidences = [p.get('volume_adjusted_confidence', 0) for p in patterns]
    
    high_volume = sum(1 for p in patterns if p.get('volume_quality') == 'high')
    medium_volume = sum(1 for p in patterns if p.get('volume_quality') == 'medium')
    low_volume = sum(1 for p in patterns if p.get('volume_quality') == 'low')
    
    summary = {
        'total_patterns': len(patterns),
        'avg_volume_score': round(np.mean(volume_scores), 3),
        'avg_adjusted_confidence': round(np.mean(confidences), 1),
        'high_volume_patterns': high_volume,
        'medium_volume_patterns': medium_volume,
        'low_volume_patterns': low_volume,
        'high_quality_ratio': round(high_volume / len(patterns), 2) if patterns else 0,
        'patterns_by_volume_trend': {
            'increasing': sum(1 for p in patterns if p.get('volume_trend') == 'increasing'),
            'decreasing': sum(1 for p in patterns if p.get('volume_trend') == 'decreasing')
        }
    }
    
    return summary


# Integration helper for existing pattern scanners
def integrate_volume_confirmation(pattern_scanner_func, data, *args, **kwargs):
    """
    Wrapper to add volume confirmation to any pattern scanner
    
    Args:
        pattern_scanner_func: Original pattern scanning function
        data: OHLCV DataFrame
        *args, **kwargs: Arguments for the pattern scanner
        
    Returns:
        Volume-enhanced patterns
    """
    # Run original pattern scanner
    patterns = pattern_scanner_func(data, *args, **kwargs)
    
    # Enhance with volume
    enhanced_patterns = enhance_patterns_with_volume(data, patterns)
    
    # Add summary
    summary = generate_volume_summary(enhanced_patterns)
    
    return {
        'patterns': enhanced_patterns,
        'summary': summary,
        'filtered_patterns': filter_patterns_by_volume(enhanced_patterns)
    }


if __name__ == "__main__":
    # Example usage
    print("Volume-Enhanced Pattern Scanner Module")
    print("Use enhance_patterns_with_volume() to add volume confirmation to detected patterns")