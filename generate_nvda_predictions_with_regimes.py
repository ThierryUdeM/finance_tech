#!/usr/bin/env python3
"""
NVDA predictions with regime-aware pattern matching
Combines enhanced system with volatility regime clustering
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import traceback

# Add directional_analysis to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'directional_analysis'))

def get_nvda_data():
    """Load NVDA data - robust path handling"""
    possible_paths = [
        'directional_analysis/NVDA_15min_pattern_ready.csv',
        'data/NVDA_15min_pattern_ready.csv',
        '../directional_analysis/NVDA_15min_pattern_ready.csv'
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                df.columns = [col.lower() for col in df.columns]
                return df.iloc[-1]['close'], df.index[-1], df
        except Exception as e:
            continue
    
    raise FileNotFoundError("NVDA data not found")

def calculate_atr_threshold(df, atr_period=14, atr_multiplier=0.25, min_threshold=0.05, max_threshold=0.50):
    """Calculate adaptive threshold based on ATR"""
    try:
        df_calc = df.copy()
        df_calc['prev_close'] = df_calc['close'].shift(1)
        df_calc['tr1'] = df_calc['high'] - df_calc['low']
        df_calc['tr2'] = abs(df_calc['high'] - df_calc['prev_close'])
        df_calc['tr3'] = abs(df_calc['low'] - df_calc['prev_close'])
        df_calc['true_range'] = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        atr = df_calc['true_range'].rolling(window=atr_period).mean().iloc[-1]
        current_price = df_calc['close'].iloc[-1]
        atr_pct = (atr / current_price) * 100
        threshold = max(min_threshold, min(atr_pct * atr_multiplier, max_threshold))
        
        return threshold
    except:
        return 0.1

def generate_regime_aware_predictions():
    """Generate predictions with regime awareness"""
    try:
        # Load data
        current_price, last_time, df = get_nvda_data()
        print(f"Loaded {len(df)} bars, current price: ${current_price:.2f}")
        
        # Calculate adaptive threshold
        adaptive_threshold = calculate_atr_threshold(df)
        
        # Try regime-aware predictions first
        try:
            from regime_clustering_simple import regime_pattern_matching
            regime_results = regime_pattern_matching(df, query_length=20, K=20, regime_weight=0.7)
            
            if regime_results:
                print("✓ Using regime-aware pattern matching")
                method = 'regime_aware_enhanced'
                confidence_boost = regime_results['confidence_score']
                regime_info = {
                    'current_regime': regime_results['current_regime'],
                    'regime_match_ratio': regime_results['regime_match_ratio'],
                    'same_regime_patterns': regime_results['same_regime_patterns']
                }
                predictions = regime_results
            else:
                raise Exception("Regime matching failed")
                
        except Exception as e:
            print(f"Regime-aware matching failed: {e}")
            print("Falling back to enhanced pattern matching...")
            
            # Fallback to enhanced system
            try:
                from intraday_pattern_matcher_enhanced_v2 import forecast_shape_enhanced
                enhanced_results = forecast_shape_enhanced("NVDA", query_length=20, K=20)
                predictions = enhanced_results
                method = 'enhanced_pattern_matching_fallback'
                confidence_boost = enhanced_results.get('confidence_score', 0.5)
                regime_info = {}
            except:
                # Ultra-simple fallback
                print("All advanced methods failed, using simple prediction")
                recent_returns = df['close'].pct_change().iloc[-20:].dropna()
                momentum = recent_returns.mean()
                vol = recent_returns.std()
                
                predictions = {
                    '1h': momentum + np.random.normal(0, vol * 0.5),
                    '3h': momentum + np.random.normal(0, vol * 1.0),
                    'eod': momentum + np.random.normal(0, vol * 1.5)
                }
                method = 'simple_fallback'
                confidence_boost = 0.3
                regime_info = {}
        
        # Convert to final format
        final_predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
        }
        
        for horizon in ['1h', '3h', 'eod']:
            pred_decimal = predictions.get(horizon, 0)
            pred_pct = pred_decimal * 100
            pred_price = current_price * (1 + pred_decimal)
            
            # Adaptive direction classification
            if pred_pct > adaptive_threshold:
                direction = 'BULLISH'
            elif pred_pct < -adaptive_threshold:
                direction = 'BEARISH'
            else:
                direction = 'NEUTRAL'
            
            final_predictions[f'pred_{horizon}_pct'] = pred_pct
            final_predictions[f'pred_{horizon}_price'] = pred_price
            final_predictions[f'pred_{horizon}_dir'] = direction
        
        # Enhanced confidence calculation
        max_pred = max(abs(final_predictions['pred_1h_pct']), 
                      abs(final_predictions['pred_3h_pct']), 
                      abs(final_predictions['pred_eod_pct']))
        
        base_confidence = 'HIGH' if max_pred > 0.5 else ('MEDIUM' if max_pred > 0.2 else 'LOW')
        
        # Boost confidence for regime consistency
        if confidence_boost > 0.8:
            enhanced_confidence = base_confidence + '+'
        elif confidence_boost < 0.4:
            enhanced_confidence = base_confidence + '-'
        else:
            enhanced_confidence = base_confidence
        
        final_predictions.update({
            'patterns_analyzed': predictions.get('patterns_analyzed', 0),
            'confidence': enhanced_confidence,
            'confidence_score': confidence_boost,
            'adaptive_threshold': adaptive_threshold,
            'ensemble_method': method,
            **regime_info  # Add regime info if available
        })
        
        # Save to CSV
        df_pred = pd.DataFrame([final_predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_pred.to_csv(csv_path, index=False)
        
        # Print results
        print(f"\nRegime-Aware Predictions saved to {csv_path}")
        print(f"Current price: ${final_predictions['current_price']:.2f}")
        print(f"Method: {final_predictions['ensemble_method']}")
        print(f"Adaptive threshold: {adaptive_threshold:.3f}%")
        
        if 'current_regime' in final_predictions:
            print(f"Current volatility regime: {final_predictions['current_regime']}")
            print(f"Same-regime pattern match: {final_predictions['regime_match_ratio']:.1%}")
        
        print(f"1H: {final_predictions['pred_1h_pct']:+.3f}% - {final_predictions['pred_1h_dir']}")
        print(f"3H: {final_predictions['pred_3h_pct']:+.3f}% - {final_predictions['pred_3h_dir']}")
        print(f"EOD: {final_predictions['pred_eod_pct']:+.3f}% - {final_predictions['pred_eod_dir']}")
        print(f"Enhanced Confidence: {final_predictions['confidence']} (Score: {confidence_boost:.3f})")
        
        return final_predictions
        
    except Exception as e:
        print(f"All prediction methods failed: {e}")
        traceback.print_exc()
        
        # Error state
        error_predictions = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': 0,
            'pred_1h_pct': 0, 'pred_1h_price': 0, 'pred_1h_dir': 'ERROR',
            'pred_3h_pct': 0, 'pred_3h_price': 0, 'pred_3h_dir': 'ERROR', 
            'pred_eod_pct': 0, 'pred_eod_price': 0, 'pred_eod_dir': 'ERROR',
            'patterns_analyzed': 0, 'confidence': 'ERROR', 'ensemble_method': 'error'
        }
        
        df_error = pd.DataFrame([error_predictions])
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        df_error.to_csv(csv_path, index=False)
        return error_predictions

# Backward compatibility
def generate_predictions():
    return generate_regime_aware_predictions()

def generate_enhanced_predictions():
    return generate_regime_aware_predictions()

def robust_predictions():
    return generate_regime_aware_predictions()

if __name__ == "__main__":
    generate_regime_aware_predictions()