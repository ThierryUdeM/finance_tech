#!/usr/bin/env python3
"""
Robust NVDA predictions with fallback systems
Designed to work reliably in GitHub Actions environment
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import traceback

def get_nvda_data():
    """Load NVDA data and get current price - robust path handling"""
    # Try multiple possible paths for the data
    possible_paths = [
        'directional_analysis/NVDA_15min_pattern_ready.csv',
        'data/NVDA_15min_pattern_ready.csv',
        '../directional_analysis/NVDA_15min_pattern_ready.csv',
        os.path.join(os.path.dirname(__file__), 'directional_analysis', 'NVDA_15min_pattern_ready.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'NVDA_15min_pattern_ready.csv'),
        'NVDA_15min_pattern_ready.csv'
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                print(f"Found NVDA data at: {data_path}")
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                # Ensure column names are standardized
                df.columns = [col.lower() for col in df.columns]
                current_price = df['close'].iloc[-1]
                last_time = df.index[-1]
                return current_price, last_time, df
        except Exception as e:
            print(f"Failed to load {data_path}: {e}")
            continue
    
    raise FileNotFoundError("NVDA data file not found in any expected location")

def calculate_atr_threshold(df, atr_period=14, atr_multiplier=0.25, min_threshold=0.05, max_threshold=0.50):
    """Calculate adaptive threshold based on ATR - simplified version"""
    try:
        # Calculate True Range
        df_calc = df.copy()
        df_calc['prev_close'] = df_calc['close'].shift(1)
        df_calc['tr1'] = df_calc['high'] - df_calc['low']
        df_calc['tr2'] = abs(df_calc['high'] - df_calc['prev_close'])
        df_calc['tr3'] = abs(df_calc['low'] - df_calc['prev_close'])
        df_calc['true_range'] = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        atr = df_calc['true_range'].rolling(window=atr_period).mean().iloc[-1]
        current_price = df_calc['close'].iloc[-1]
        
        # Convert to percentage
        atr_pct = (atr / current_price) * 100
        
        # Apply multiplier and constraints
        threshold = atr_pct * atr_multiplier
        threshold = max(min_threshold, min(threshold, max_threshold))
        
        print(f"Adaptive threshold: {threshold:.3f}% (ATR: {atr_pct:.3f}%)")
        return threshold
        
    except Exception as e:
        print(f"ATR calculation failed: {e}, using fallback")
        return 0.1  # Fallback threshold

def simple_pattern_prediction(df, current_price):
    """Simple pattern-based prediction as fallback"""
    try:
        # Use recent price action for simple prediction
        recent_bars = min(20, len(df))
        recent_data = df.iloc[-recent_bars:]
        
        # Calculate momentum
        returns = recent_data['close'].pct_change().dropna()
        momentum = returns.mean()
        volatility = returns.std()
        
        # Simple prediction based on momentum and volatility
        pred_1h = momentum + np.random.normal(0, volatility * 0.5)
        pred_3h = momentum + np.random.normal(0, volatility * 1.0)
        pred_eod = momentum + np.random.normal(0, volatility * 1.5)
        
        return {
            '1h': pred_1h,
            '3h': pred_3h, 
            'eod': pred_eod,
            'confidence_score': 0.5,
            'method': 'simple_pattern_fallback'
        }
    except Exception as e:
        print(f"Simple pattern prediction failed: {e}")
        return None

def volatility_prediction(df, current_price):
    """Volatility-based prediction with proper data separation"""
    try:
        if len(df) < 60:
            raise ValueError(f"Insufficient data: need 60 bars, have {len(df)}")
        
        # Training data: bars [t-60, t-21] (40 bars for volatility estimation)
        training_data = df.iloc[-60:-20]
        
        # Current data: bars [t-20, t] (20 bars for current pattern)
        current_data = df.iloc[-20:]
        
        # Calculate volatility from training data only
        training_returns = training_data['close'].pct_change().dropna()
        volatility = training_returns.std()
        
        # Calculate recent momentum from current data
        current_returns = current_data['close'].pct_change().dropna()
        momentum = current_returns.mean()
        
        # Generate predictions
        np.random.seed(int(datetime.now().timestamp()) % 1000)  # Semi-deterministic
        
        pred_1h = np.random.normal(momentum, volatility * 2)
        pred_3h = np.random.normal(momentum, volatility * 3)  
        pred_eod = np.random.normal(momentum, volatility * 4)
        
        # Calculate confidence based on volatility consistency
        current_vol = current_returns.std()
        vol_ratio = min(volatility, current_vol) / max(volatility, current_vol)
        vol_confidence = vol_ratio * 0.6
        
        return {
            '1h': pred_1h,
            '3h': pred_3h,
            'eod': pred_eod,
            'confidence_score': vol_confidence,
            'method': 'volatility_fixed'
        }
    except Exception as e:
        print(f"Volatility prediction failed: {e}")
        return None

def try_enhanced_predictions():
    """Try to use enhanced predictions if available"""
    try:
        # Try to import enhanced system
        sys.path.append('directional_analysis')
        from intraday_pattern_matcher_enhanced_v2 import forecast_shape_enhanced
        
        print("Attempting enhanced pattern matching...")
        results = forecast_shape_enhanced("NVDA", query_length=20, K=10)
        
        if results and not any(np.isnan(results.get(h, np.nan)) for h in ['1h', '3h', 'eod']):
            print("✓ Enhanced pattern matching succeeded")
            return results
        else:
            print("Enhanced pattern matching returned invalid results")
            return None
            
    except Exception as e:
        print(f"Enhanced predictions failed: {e}")
        return None

def robust_predictions():
    """Generate predictions with multiple fallback methods"""
    error_response = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_price': 0,
        'pred_1h_pct': 0, 'pred_1h_price': 0, 'pred_1h_dir': 'ERROR',
        'pred_3h_pct': 0, 'pred_3h_price': 0, 'pred_3h_dir': 'ERROR', 
        'pred_eod_pct': 0, 'pred_eod_price': 0, 'pred_eod_dir': 'ERROR',
        'patterns_analyzed': 0, 'confidence': 'ERROR',
        'ensemble_method': 'error'
    }
    
    try:
        # Load data
        current_price, last_time, df = get_nvda_data()
        print(f"Loaded {len(df)} bars, current price: ${current_price:.2f}")
        
        # Calculate adaptive threshold
        adaptive_threshold = calculate_atr_threshold(df)
        
        # Try prediction methods in order of preference
        predictions = None
        
        # Method 1: Enhanced predictions
        predictions = try_enhanced_predictions()
        
        # Method 2: Simple pattern prediction
        if predictions is None:
            print("Trying simple pattern prediction...")
            predictions = simple_pattern_prediction(df, current_price)
        
        # Method 3: Volatility prediction
        if predictions is None:
            print("Trying volatility prediction...")
            predictions = volatility_prediction(df, current_price)
        
        # Method 4: Ultra-simple fallback
        if predictions is None:
            print("Using ultra-simple fallback...")
            recent_change = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) >= 5 else 0
            predictions = {
                '1h': recent_change * 0.2,
                '3h': recent_change * 0.5,
                'eod': recent_change * 0.8,
                'confidence_score': 0.3,
                'method': 'ultra_simple_fallback'
            }
        
        # Convert predictions to final format
        final_predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
        }
        
        for horizon in ['1h', '3h', 'eod']:
            pred_decimal = predictions.get(horizon, 0)
            pred_pct = pred_decimal * 100
            pred_price = current_price * (1 + pred_decimal)
            
            # Use adaptive threshold for direction classification
            if pred_pct > adaptive_threshold:
                direction = 'BULLISH'
            elif pred_pct < -adaptive_threshold:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            final_predictions[f'pred_{horizon}_pct'] = pred_pct
            final_predictions[f'pred_{horizon}_price'] = pred_price
            final_predictions[f'pred_{horizon}_dir'] = direction
        
        # Add metadata
        max_pred = max(abs(final_predictions['pred_1h_pct']), 
                      abs(final_predictions['pred_3h_pct']), 
                      abs(final_predictions['pred_eod_pct']))
        
        if max_pred > 0.5:
            confidence = 'HIGH'
        elif max_pred > 0.2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        final_predictions.update({
            'patterns_analyzed': 0,  # Will be updated if enhanced system works
            'confidence': confidence,
            'confidence_score': predictions.get('confidence_score', 0.5),
            'adaptive_threshold': adaptive_threshold,
            'ensemble_method': predictions.get('method', 'robust_fallback')
        })
        
        # Save to CSV
        df_pred = pd.DataFrame([final_predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_pred.to_csv(csv_path, index=False)
        
        print(f"\nRobust predictions saved to {csv_path}")
        print(f"Current price: ${final_predictions['current_price']:.2f}")
        print(f"Method: {final_predictions['ensemble_method']}")
        print(f"1H: {final_predictions['pred_1h_pct']:+.3f}% - {final_predictions['pred_1h_dir']}")
        print(f"3H: {final_predictions['pred_3h_pct']:+.3f}% - {final_predictions['pred_3h_dir']}")
        print(f"EOD: {final_predictions['pred_eod_pct']:+.3f}% - {final_predictions['pred_eod_dir']}")
        print(f"Confidence: {final_predictions['confidence']}")
        
        return final_predictions
        
    except Exception as e:
        print(f"All prediction methods failed: {e}")
        print("Stack trace:")
        traceback.print_exc()
        
        # Save error state
        df_error = pd.DataFrame([error_response])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_error.to_csv(csv_path, index=False)
        
        return error_response

# Backward compatibility
def generate_predictions():
    """Backward compatible wrapper"""
    return robust_predictions()

def generate_enhanced_predictions():
    """Enhanced wrapper"""
    return robust_predictions()

if __name__ == "__main__":
    robust_predictions()