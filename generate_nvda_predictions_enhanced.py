#!/usr/bin/env python3
"""
Enhanced NVDA predictions with probability-based ensemble and fixed data leakage
Combines advanced pattern matching with improved volatility modeling
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import yaml

# Add directional_analysis to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'directional_analysis'))

# Import the enhanced pattern matcher
from intraday_pattern_matcher_enhanced_v2 import forecast_shape_enhanced, load_config

def load_ensemble_config():
    """Load ensemble configuration"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'ensemble_weights.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Config file not found, using defaults")
        return get_default_config()

def get_default_config():
    """Default configuration"""
    return {
        'timeframe_weights': {'short_term': 0.4, 'medium_term': 0.35, 'long_term': 0.25},
        'volatility_model': {'training_lookback': 40, 'prediction_window': 20, 'confidence_blend': True},
        'adaptive_thresholds': {'enabled': True, 'fallback_threshold': 0.1}
    }

def get_nvda_data():
    """Load NVDA data and get current price"""
    # Try multiple possible paths for the data
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'directional_analysis', 'NVDA_15min_pattern_ready.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'NVDA_15min_pattern_ready.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'directional_analysis', 'NVDA_15min_pattern_ready.csv')
    ]
    
    for data_path in possible_paths:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            # Ensure column names are standardized
            df.columns = [col.lower() for col in df.columns]
            current_price = df['close'].iloc[-1]
            last_time = df.index[-1]
            return current_price, last_time, df
    
    raise FileNotFoundError("NVDA data file not found in any expected location")

def calculate_volatility_predictions_fixed(df, config):
    """
    Generate volatility-based predictions with fixed data leakage
    Uses separate training/prediction windows
    """
    vol_config = config['volatility_model']
    training_lookback = vol_config['training_lookback']  # 40 bars
    prediction_window = vol_config['prediction_window']   # 20 bars
    
    if len(df) < training_lookback + prediction_window:
        raise ValueError(f"Insufficient data: need {training_lookback + prediction_window}, have {len(df)}")
    
    # Training data: bars [t-60, t-21] (40 bars for volatility estimation)
    training_end = len(df) - prediction_window
    training_start = training_end - training_lookback
    training_data = df.iloc[training_start:training_end]
    
    # Current data: bars [t-20, t] (20 bars for current pattern)
    current_data = df.iloc[-prediction_window:]
    
    # Calculate volatility from training data only
    training_returns = training_data['close'].pct_change().dropna()
    volatility = training_returns.std()
    
    # Calculate recent momentum from current data
    current_returns = current_data['close'].pct_change().dropna()
    momentum = current_returns.mean()
    
    # Generate predictions with probabilistic approach
    np.random.seed(int(datetime.now().timestamp()) % 1000)  # Semi-deterministic
    
    # Base predictions on training volatility + current momentum
    pred_1h = np.random.normal(momentum, volatility * 2) * 100
    pred_3h = np.random.normal(momentum, volatility * 3) * 100  
    pred_eod = np.random.normal(momentum, volatility * 4) * 100
    
    # Calculate confidence based on volatility consistency
    current_vol = current_returns.std()
    vol_ratio = min(volatility, current_vol) / max(volatility, current_vol)
    vol_confidence = vol_ratio * 0.8  # Scale to [0, 0.8]
    
    return {
        '1h': pred_1h / 100,  # Convert to decimal
        '3h': pred_3h / 100,
        'eod': pred_eod / 100,
        'confidence': vol_confidence,
        'method': 'volatility_fixed',
        'training_volatility': volatility,
        'current_volatility': current_vol
    }

def ensemble_predictions(pattern_preds, volatility_preds, config):
    """
    Combine pattern matching and volatility predictions using confidence weighting
    """
    if not config['volatility_model']['confidence_blend']:
        # Use pattern matching only
        return pattern_preds
    
    # Get confidence scores
    pattern_conf = pattern_preds.get('confidence_score', 0.5)
    vol_conf = volatility_preds.get('confidence', 0.3)
    
    # Normalize confidences
    total_conf = pattern_conf + vol_conf
    if total_conf == 0:
        pattern_weight, vol_weight = 0.7, 0.3  # Default fallback
    else:
        pattern_weight = pattern_conf / total_conf
        vol_weight = vol_conf / total_conf
    
    print(f"Ensemble weights: Pattern {pattern_weight:.2f}, Volatility {vol_weight:.2f}")
    
    # Combine predictions
    ensemble_preds = {}
    for horizon in ['1h', '3h', 'eod']:
        if horizon in pattern_preds and horizon in volatility_preds:
            if not np.isnan(pattern_preds[horizon]):
                ensemble_preds[horizon] = (
                    pattern_weight * pattern_preds[horizon] + 
                    vol_weight * volatility_preds[horizon]
                )
            else:
                ensemble_preds[horizon] = volatility_preds[horizon]
        elif horizon in pattern_preds:
            ensemble_preds[horizon] = pattern_preds[horizon]
        elif horizon in volatility_preds:
            ensemble_preds[horizon] = volatility_preds[horizon]
        else:
            ensemble_preds[horizon] = 0.0
    
    # Combine other metadata
    ensemble_preds.update({
        'confidence_score': pattern_conf,
        'vol_confidence': vol_conf,
        'ensemble_method': 'probability_weighted',
        'library_size': pattern_preds.get('library_size', 0),
        'adaptive_threshold': pattern_preds.get('adaptive_threshold', 0.1),
        'current_price': pattern_preds.get('current_price', 0),
        'method': 'enhanced_ensemble_v2'
    })
    
    return ensemble_preds

def generate_enhanced_predictions():
    """Generate enhanced predictions with all improvements"""
    try:
        # Load configuration
        config = load_ensemble_config()
        
        # Get data
        current_price, last_time, df = get_nvda_data()
        print(f"Using {len(df)} bars of NVDA data, current price: ${current_price:.2f}")
        
        # Configure ensemble weights from YAML
        timeframe_weights = config['timeframe_weights']
        
        # Generate pattern-based predictions using ensemble of timeframes
        print("Generating pattern-based predictions...")
        
        # Short term (2h lookback)
        short_preds = forecast_shape_enhanced(
            "NVDA", query_length=8, K=20, config=config
        )
        
        # Medium term (3h lookback)  
        medium_preds = forecast_shape_enhanced(
            "NVDA", query_length=12, K=30, config=config
        )
        
        # Long term (5h lookbook)
        long_preds = forecast_shape_enhanced(
            "NVDA", query_length=20, K=50, config=config
        )
        
        # Combine pattern predictions using configurable weights
        pattern_ensemble = {}
        for horizon in ['1h', '3h', 'eod']:
            predictions = []
            weights = []
            
            for preds, weight_key in [
                (short_preds, 'short_term'),
                (medium_preds, 'medium_term'), 
                (long_preds, 'long_term')
            ]:
                if horizon in preds and not np.isnan(preds[horizon]):
                    predictions.append(preds[horizon])
                    weights.append(timeframe_weights[weight_key])
            
            if predictions:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                pattern_ensemble[horizon] = np.average(predictions, weights=weights)
            else:
                pattern_ensemble[horizon] = 0.0
        
        # Add metadata from long-term prediction (most comprehensive)
        pattern_ensemble.update({
            'confidence_score': long_preds.get('confidence_score', 0.5),
            'library_size': long_preds.get('library_size', 0),
            'adaptive_threshold': long_preds.get('adaptive_threshold', 0.1),
            'current_price': current_price,
            'data_source': long_preds.get('data_source', 'local')
        })
        
        # Generate volatility-based predictions (fixed data leakage)
        print("Generating volatility-based predictions...")
        volatility_preds = calculate_volatility_predictions_fixed(df, config)
        
        # Combine using probability-based ensemble
        print("Combining predictions using probability ensemble...")
        final_preds = ensemble_predictions(pattern_ensemble, volatility_preds, config)
        
        # Convert to percentage and calculate price targets
        predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
        }
        
        adaptive_threshold = final_preds.get('adaptive_threshold', 0.1)
        
        for horizon in ['1h', '3h', 'eod']:
            pred_pct = final_preds[horizon] * 100
            pred_price = current_price * (1 + final_preds[horizon])
            
            # Use adaptive threshold for direction classification
            if pred_pct > adaptive_threshold:
                direction = 'BULLISH'
            elif pred_pct < -adaptive_threshold:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            predictions[f'pred_{horizon}_pct'] = pred_pct
            predictions[f'pred_{horizon}_price'] = pred_price
            predictions[f'pred_{horizon}_dir'] = direction
        
        # Add metadata
        max_pred = max(abs(predictions['pred_1h_pct']), 
                      abs(predictions['pred_3h_pct']), 
                      abs(predictions['pred_eod_pct']))
        
        if max_pred > 0.5:
            confidence = 'HIGH'
        elif max_pred > 0.2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        predictions.update({
            'patterns_analyzed': final_preds.get('library_size', 0),
            'confidence': confidence,
            'confidence_score': final_preds.get('confidence_score', 0.5),
            'adaptive_threshold': adaptive_threshold,
            'ensemble_method': 'enhanced_v2',
            'improvements': [
                'fixed_data_leakage',
                'probability_ensemble', 
                'configurable_weights',
                'adaptive_thresholds',
                'time_decay_weighting'
            ]
        })
        
        # Save to CSV (backward compatibility)
        df_pred = pd.DataFrame([predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_pred.to_csv(csv_path, index=False)
        
        # Print results
        print(f"\nEnhanced Predictions saved to {csv_path}")
        print(f"Current price: ${predictions['current_price']:.2f}")
        print(f"Adaptive threshold: {adaptive_threshold:.3f}%")
        print(f"1H: {predictions['pred_1h_pct']:+.3f}% (${predictions['pred_1h_price']:.2f}) - {predictions['pred_1h_dir']}")
        print(f"3H: {predictions['pred_3h_pct']:+.3f}% (${predictions['pred_3h_price']:.2f}) - {predictions['pred_3h_dir']}")
        print(f"EOD: {predictions['pred_eod_pct']:+.3f}% (${predictions['pred_eod_price']:.2f}) - {predictions['pred_eod_dir']}")
        print(f"Patterns analyzed: {predictions['patterns_analyzed']:,}")
        print(f"Confidence: {predictions['confidence']} (Score: {predictions['confidence_score']:.3f})")
        print(f"Method: {predictions['ensemble_method']}")
        
        return predictions
        
    except Exception as e:
        print(f"Error generating enhanced predictions: {e}")
        # Return error state
        error_predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': 0,
            'pred_1h_pct': 0, 'pred_1h_price': 0, 'pred_1h_dir': 'ERROR',
            'pred_3h_pct': 0, 'pred_3h_price': 0, 'pred_3h_dir': 'ERROR', 
            'pred_eod_pct': 0, 'pred_eod_price': 0, 'pred_eod_dir': 'ERROR',
            'patterns_analyzed': 0, 'confidence': 'ERROR',
            'ensemble_method': 'error', 'error': str(e)
        }
        
        # Still save error state
        df_error = pd.DataFrame([error_predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_error.to_csv(csv_path, index=False)
        return error_predictions

# Backward compatibility function
def generate_predictions():
    """Backward compatible wrapper"""
    return generate_enhanced_predictions()

if __name__ == "__main__":
    generate_enhanced_predictions()