#!/usr/bin/env python3
"""
Generate NVDA predictions and save to CSV
"""
import sys
import os
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'directional_analysis'))

# Import the pattern matcher
from intraday_pattern_matcher_enhanced import forecast_shape

def generate_predictions():
    """Generate NVDA predictions using pattern matching"""
    try:
        # Get predictions for different timeframes
        short = forecast_shape("NVDA", query_length=8, K=20)   # 2 hours
        medium = forecast_shape("NVDA", query_length=12, K=30) # 3 hours
        long = forecast_shape("NVDA", query_length=20, K=50)   # 5 hours
        
        # Calculate ensemble predictions
        predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': short['current_price'],
            '1h_percent': (short['1h'] + medium['1h'] + long['1h']) / 3 * 100,
            '1h_price': short['current_price'] * (1 + (short['1h'] + medium['1h'] + long['1h']) / 3),
            '3h_percent': (short['3h'] + medium['3h'] + long['3h']) / 3 * 100,
            '3h_price': short['current_price'] * (1 + (short['3h'] + medium['3h'] + long['3h']) / 3),
            'eod_percent': (short['eod'] + medium['eod'] + long['eod']) / 3 * 100,
            'eod_price': short['current_price'] * (1 + (short['eod'] + medium['eod'] + long['eod']) / 3),
            'patterns_analyzed': long['library_size'],
            'forecast_method': short['method']
        }
        
        # Determine directions
        predictions['1h_direction'] = 'BULLISH' if predictions['1h_percent'] > 0.1 else ('BEARISH' if predictions['1h_percent'] < -0.1 else 'NEUTRAL')
        predictions['3h_direction'] = 'BULLISH' if predictions['3h_percent'] > 0.1 else ('BEARISH' if predictions['3h_percent'] < -0.1 else 'NEUTRAL')
        predictions['eod_direction'] = 'BULLISH' if predictions['eod_percent'] > 0.1 else ('BEARISH' if predictions['eod_percent'] < -0.1 else 'NEUTRAL')
        
        # Calculate confidence
        max_move = max(abs(predictions['1h_percent']), abs(predictions['3h_percent']), abs(predictions['eod_percent']))
        predictions['confidence'] = 'HIGH' if max_move > 0.5 else ('MEDIUM' if max_move > 0.2 else 'LOW')
        
        # Save to CSV
        df = pd.DataFrame([predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Predictions saved to {csv_path}")
        print(f"Current price: ${predictions['current_price']:.2f}")
        print(f"1H: {predictions['1h_percent']:+.3f}% (${predictions['1h_price']:.2f}) - {predictions['1h_direction']}")
        print(f"3H: {predictions['3h_percent']:+.3f}% (${predictions['3h_price']:.2f}) - {predictions['3h_direction']}")
        print(f"EOD: {predictions['eod_percent']:+.3f}% (${predictions['eod_price']:.2f}) - {predictions['eod_direction']}")
        print(f"Patterns analyzed: {predictions['patterns_analyzed']:,}")
        print(f"Confidence: {predictions['confidence']}")
        
        return predictions
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        # Save error state to CSV
        error_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': 0,
            '1h_percent': 0,
            '1h_price': 0,
            '3h_percent': 0,
            '3h_price': 0,
            'eod_percent': 0,
            'eod_price': 0,
            'patterns_analyzed': 0,
            'forecast_method': 'ERROR',
            '1h_direction': 'ERROR',
            '3h_direction': 'ERROR',
            'eod_direction': 'ERROR',
            'confidence': 'ERROR'
        }
        df = pd.DataFrame([error_data])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df.to_csv(csv_path, index=False)
        return None

if __name__ == "__main__":
    generate_predictions()