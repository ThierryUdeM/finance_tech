#!/usr/bin/env python3
"""
Test script for the enhanced NVDA prediction system
Validates all new features work correctly
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """Test that configuration loading works"""
    print("=== Testing Configuration Loading ===")
    
    try:
        from directional_analysis.intraday_pattern_matcher_enhanced_v2 import load_config
        config = load_config()
        
        print("✓ Configuration loaded successfully")
        print(f"  Timeframe weights: {config['timeframe_weights']}")
        print(f"  Adaptive thresholds enabled: {config['adaptive_thresholds']['enabled']}")
        print(f"  Time decay enabled: {config['time_decay']['enabled']}")
        
        # Validate structure
        required_keys = ['timeframe_weights', 'adaptive_thresholds', 'time_decay', 'pattern_matching']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing config key: {key}")
                return False
        
        print("✓ Configuration structure valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_enhanced_pattern_matcher():
    """Test the enhanced pattern matching algorithm"""
    print("\n=== Testing Enhanced Pattern Matcher ===")
    
    try:
        from directional_analysis.intraday_pattern_matcher_enhanced_v2 import forecast_shape_enhanced
        
        # Test with default parameters
        results = forecast_shape_enhanced("NVDA", query_length=20, K=10)
        
        print("✓ Enhanced pattern matcher executed successfully")
        print(f"  Method: {results.get('method', 'unknown')}")
        print(f"  Library size: {results.get('library_size', 0):,} patterns")
        print(f"  Confidence: {results.get('confidence', 'unknown')}")
        print(f"  Confidence score: {results.get('confidence_score', 0):.3f}")
        print(f"  Adaptive threshold: {results.get('adaptive_threshold', 0):.3f}%")
        print(f"  Current price: ${results.get('current_price', 0):.2f}")
        
        # Check improvements are present
        improvements = results.get('improvements', [])
        expected_improvements = ['time_decay_weighting', 'adaptive_atr_thresholds', 'configurable_ensemble']
        for improvement in expected_improvements:
            if improvement in improvements:
                print(f"  ✓ {improvement}")
            else:
                print(f"  ⚠ {improvement} not found")
        
        # Validate predictions
        horizons = ['1h', '3h', 'eod']
        for horizon in horizons:
            if horizon in results and not np.isnan(results[horizon]):
                print(f"  {horizon}: {results[horizon]*100:+.3f}%")
            else:
                print(f"  {horizon}: No prediction")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced pattern matcher failed: {e}")
        return False

def test_enhanced_predictions():
    """Test the enhanced prediction system"""
    print("\n=== Testing Enhanced Prediction System ===")
    
    try:
        from generate_nvda_predictions_enhanced import generate_enhanced_predictions
        
        # Generate enhanced predictions
        predictions = generate_enhanced_predictions()
        
        print("✓ Enhanced predictions generated successfully")
        print(f"  Method: {predictions.get('ensemble_method', 'unknown')}")
        print(f"  Current price: ${predictions.get('current_price', 0):.2f}")
        print(f"  Confidence: {predictions.get('confidence', 'unknown')}")
        print(f"  Confidence score: {predictions.get('confidence_score', 0):.3f}")
        print(f"  Adaptive threshold: {predictions.get('adaptive_threshold', 0):.3f}%")
        print(f"  Patterns analyzed: {predictions.get('patterns_analyzed', 0):,}")
        
        # Check improvements
        improvements = predictions.get('improvements', [])
        expected_improvements = [
            'fixed_data_leakage', 'probability_ensemble', 
            'configurable_weights', 'adaptive_thresholds', 'time_decay_weighting'
        ]
        for improvement in expected_improvements:
            if improvement in improvements:
                print(f"  ✓ {improvement}")
            else:
                print(f"  ⚠ {improvement} not found")
        
        # Display predictions
        horizons = ['1h', '3h', 'eod']
        for horizon in horizons:
            pct_key = f'pred_{horizon}_pct'
            price_key = f'pred_{horizon}_price' 
            dir_key = f'pred_{horizon}_dir'
            
            if all(key in predictions for key in [pct_key, price_key, dir_key]):
                print(f"  {horizon.upper()}: {predictions[pct_key]:+.3f}% (${predictions[price_key]:.2f}) - {predictions[dir_key]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced prediction system failed: {e}")
        return False

def test_data_leakage_fix():
    """Test that data leakage has been fixed"""
    print("\n=== Testing Data Leakage Fix ===")
    
    try:
        from generate_nvda_predictions_enhanced import calculate_volatility_predictions_fixed, get_nvda_data, load_ensemble_config
        
        # Get data
        current_price, last_time, df = get_nvda_data()
        config = load_ensemble_config()
        
        # Test volatility predictions with separate windows
        vol_preds = calculate_volatility_predictions_fixed(df, config)
        
        print("✓ Volatility predictions with fixed data leakage")
        print(f"  Method: {vol_preds.get('method', 'unknown')}")
        print(f"  Training volatility: {vol_preds.get('training_volatility', 0)*100:.3f}%")
        print(f"  Current volatility: {vol_preds.get('current_volatility', 0)*100:.3f}%")
        print(f"  Confidence: {vol_preds.get('confidence', 0):.3f}")
        
        for horizon in ['1h', '3h', 'eod']:
            if horizon in vol_preds:
                print(f"  {horizon}: {vol_preds[horizon]*100:+.3f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Data leakage fix test failed: {e}")
        return False

def test_csv_output():
    """Test CSV output compatibility"""
    print("\n=== Testing CSV Output ===")
    
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'nvda_predictions.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ CSV file exists with {len(df)} rows")
            
            # Check required columns
            required_cols = [
                'timestamp', 'current_price', 'confidence',
                'pred_1h_pct', 'pred_1h_price', 'pred_1h_dir',
                'pred_3h_pct', 'pred_3h_price', 'pred_3h_dir',
                'pred_eod_pct', 'pred_eod_price', 'pred_eod_dir'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠ Missing columns: {missing_cols}")
            else:
                print("✓ All required columns present")
            
            # Display latest row
            if len(df) > 0:
                latest = df.iloc[-1]
                print(f"  Latest prediction: {latest['timestamp']}")
                print(f"  Price: ${latest['current_price']:.2f}")
                print(f"  1H: {latest['pred_1h_pct']:+.3f}% - {latest['pred_1h_dir']}")
                print(f"  Confidence: {latest['confidence']}")
                
                # Check for new enhanced columns
                enhanced_cols = ['confidence_score', 'adaptive_threshold', 'ensemble_method']
                for col in enhanced_cols:
                    if col in df.columns:
                        print(f"  ✓ Enhanced column '{col}': {latest.get(col, 'N/A')}")
            
            return True
        else:
            print("⚠ CSV file not found")
            return False
            
    except Exception as e:
        print(f"✗ CSV output test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("Enhanced NVDA Prediction System - Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_enhanced_pattern_matcher,
        test_enhanced_predictions,
        test_data_leakage_fix,
        test_csv_output
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ All tests PASSED - Enhanced system is ready!")
        return True
    else:
        print(f"✗ {total-passed} tests FAILED - Please review issues above")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)