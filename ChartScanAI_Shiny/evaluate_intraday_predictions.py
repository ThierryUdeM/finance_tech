#!/usr/bin/env python3
"""
Evaluate NVDA intraday pattern predictions
This script:
1. Makes predictions using current intraday data
2. Stores predictions with timestamps
3. Evaluates past predictions against actual outcomes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Import the enhanced prediction generator
sys.path.append('..')  # Add parent directory to path
try:
    from generate_nvda_predictions_enhanced import generate_enhanced_predictions as generate_predictions
    print("Using enhanced prediction generator")
except ImportError:
    from generate_nvda_predictions_simple import generate_predictions
    print("Fallback to simple prediction generator")

def load_prediction_history(history_file='prediction_history.json'):
    """Load historical predictions"""
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_prediction_history(history, history_file='prediction_history.json'):
    """Save prediction history"""
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def make_current_prediction():
    """Make predictions based on current intraday data"""
    # Generate predictions using the simple predictor
    predictions = generate_predictions()
    
    # Add metadata
    predictions['prediction_time'] = datetime.now().isoformat()
    predictions['evaluation_status'] = {
        '1h': 'pending',
        '3h': 'pending',
        'eod': 'pending'
    }
    
    return predictions

def evaluate_past_predictions(history, current_data_path='directional_analysis/NVDA_intraday_current.csv'):
    """Evaluate past predictions that are now mature"""
    
    # Load current data
    if not os.path.exists(current_data_path):
        print("No current data available for evaluation")
        return history
    
    current_data = pd.read_csv(current_data_path, index_col='timestamp', parse_dates=True)
    latest_price = current_data['Close'].iloc[-1]
    latest_time = current_data.index[-1]
    
    # Evaluate each pending prediction
    for pred in history:
        pred_time = datetime.fromisoformat(pred['prediction_time'])
        
        # Evaluate 1-hour prediction
        if pred['evaluation_status']['1h'] == 'pending':
            target_time = pred_time + timedelta(hours=1)
            if latest_time >= target_time:
                # Find actual price at target time
                actual_prices = current_data[current_data.index >= target_time]
                if len(actual_prices) > 0:
                    actual_price = actual_prices['Close'].iloc[0]
                    actual_change = ((actual_price - pred['current_price']) / pred['current_price']) * 100
                    
                    pred['evaluation_status']['1h'] = 'evaluated'
                    pred['actual_1h'] = {
                        'price': actual_price,
                        'percent': actual_change,
                        'direction': 'BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL'),
                        'correct': pred['pred_1h_dir'] == ('BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL')),
                        'error': abs(pred['pred_1h_pct'] - actual_change)
                    }
        
        # Evaluate 3-hour prediction
        if pred['evaluation_status']['3h'] == 'pending':
            target_time = pred_time + timedelta(hours=3)
            if latest_time >= target_time:
                actual_prices = current_data[current_data.index >= target_time]
                if len(actual_prices) > 0:
                    actual_price = actual_prices['Close'].iloc[0]
                    actual_change = ((actual_price - pred['current_price']) / pred['current_price']) * 100
                    
                    pred['evaluation_status']['3h'] = 'evaluated'
                    pred['actual_3h'] = {
                        'price': actual_price,
                        'percent': actual_change,
                        'direction': 'BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL'),
                        'correct': pred['pred_3h_dir'] == ('BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL')),
                        'error': abs(pred['pred_3h_pct'] - actual_change)
                    }
        
        # Evaluate EOD prediction
        if pred['evaluation_status']['eod'] == 'pending':
            # Check if market has closed (4 PM)
            pred_date = pred_time.date()
            market_close = datetime.combine(pred_date, datetime.strptime("16:00", "%H:%M").time())
            
            if latest_time >= market_close:
                # Find price at market close
                eod_prices = current_data[current_data.index.date == pred_date]
                if len(eod_prices) > 0:
                    eod_price = eod_prices['Close'].iloc[-1]
                    actual_change = ((eod_price - pred['current_price']) / pred['current_price']) * 100
                    
                    pred['evaluation_status']['eod'] = 'evaluated'
                    pred['actual_eod'] = {
                        'price': eod_price,
                        'percent': actual_change,
                        'direction': 'BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL'),
                        'correct': pred['pred_eod_dir'] == ('BULLISH' if actual_change > 0.1 else ('BEARISH' if actual_change < -0.1 else 'NEUTRAL')),
                        'error': abs(pred['pred_eod_pct'] - actual_change)
                    }
    
    return history

def generate_performance_report(history):
    """Generate performance metrics from prediction history"""
    
    metrics = {
        '1h': {'total': 0, 'correct': 0, 'errors': []},
        '3h': {'total': 0, 'correct': 0, 'errors': []},
        'eod': {'total': 0, 'correct': 0, 'errors': []}
    }
    
    for pred in history:
        # 1-hour metrics
        if 'actual_1h' in pred:
            metrics['1h']['total'] += 1
            if pred['actual_1h']['correct']:
                metrics['1h']['correct'] += 1
            metrics['1h']['errors'].append(pred['actual_1h']['error'])
        
        # 3-hour metrics
        if 'actual_3h' in pred:
            metrics['3h']['total'] += 1
            if pred['actual_3h']['correct']:
                metrics['3h']['correct'] += 1
            metrics['3h']['errors'].append(pred['actual_3h']['error'])
        
        # EOD metrics
        if 'actual_eod' in pred:
            metrics['eod']['total'] += 1
            if pred['actual_eod']['correct']:
                metrics['eod']['correct'] += 1
            metrics['eod']['errors'].append(pred['actual_eod']['error'])
    
    # Calculate summary statistics
    summary = {}
    for timeframe in ['1h', '3h', 'eod']:
        if metrics[timeframe]['total'] > 0:
            summary[timeframe] = {
                'total_predictions': metrics[timeframe]['total'],
                'correct_predictions': metrics[timeframe]['correct'],
                'direction_accuracy': (metrics[timeframe]['correct'] / metrics[timeframe]['total']) * 100,
                'avg_error': np.mean(metrics[timeframe]['errors']),
                'max_error': np.max(metrics[timeframe]['errors']),
                'min_error': np.min(metrics[timeframe]['errors'])
            }
        else:
            summary[timeframe] = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'direction_accuracy': 0,
                'avg_error': 0,
                'max_error': 0,
                'min_error': 0
            }
    
    return summary

def main():
    """Main evaluation process"""
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Load prediction history
    history_file = 'evaluation_results/nvda_prediction_history.json'
    history = load_prediction_history(history_file)
    
    print("=== NVDA Intraday Pattern Prediction Evaluation ===")
    print(f"Evaluation time: {datetime.now()}")
    
    # Step 1: Make new prediction for current time
    print("\n1. Making current prediction...")
    try:
        current_pred = make_current_prediction()
        history.append(current_pred)
        print(f"  Current price: ${current_pred['current_price']:.2f}")
        print(f"  1H prediction: {current_pred['pred_1h_pct']:+.3f}% ({current_pred['pred_1h_dir']})")
        print(f"  3H prediction: {current_pred['pred_3h_pct']:+.3f}% ({current_pred['pred_3h_dir']})")
        print(f"  EOD prediction: {current_pred['pred_eod_pct']:+.3f}% ({current_pred['pred_eod_dir']})")
    except Exception as e:
        print(f"  Error making prediction: {e}")
    
    # Step 2: Evaluate past predictions
    print("\n2. Evaluating past predictions...")
    history = evaluate_past_predictions(history)
    
    # Step 3: Generate performance report
    print("\n3. Performance Summary:")
    performance = generate_performance_report(history)
    
    for timeframe in ['1h', '3h', 'eod']:
        perf = performance[timeframe]
        print(f"\n{timeframe.upper()} Predictions:")
        print(f"  Total evaluated: {perf['total_predictions']}")
        print(f"  Correct direction: {perf['correct_predictions']}")
        print(f"  Direction accuracy: {perf['direction_accuracy']:.1f}%")
        print(f"  Average error: {perf['avg_error']:.3f}%")
    
    # Save updated history
    save_prediction_history(history, history_file)
    
    # Save performance metrics
    metrics_file = f"evaluation_results/nvda_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(performance, f, indent=2)
    
    # Generate detailed report
    report_file = f"evaluation_results/nvda_intraday_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write("# NVDA Intraday Pattern Prediction Report\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("## Current Prediction\n")
        if 'current_pred' in locals():
            f.write(f"- Current Price: ${current_pred['current_price']:.2f}\n")
            f.write(f"- 1H: {current_pred['pred_1h_pct']:+.3f}% to ${current_pred['pred_1h_price']:.2f}\n")
            f.write(f"- 3H: {current_pred['pred_3h_pct']:+.3f}% to ${current_pred['pred_3h_price']:.2f}\n")
            f.write(f"- EOD: {current_pred['pred_eod_pct']:+.3f}% to ${current_pred['pred_eod_price']:.2f}\n")
        
        f.write("\n## Performance Metrics\n")
        for tf in ['1h', '3h', 'eod']:
            perf = performance[tf]
            f.write(f"\n### {tf.upper()} Predictions\n")
            f.write(f"- Total Evaluated: {perf['total_predictions']}\n")
            f.write(f"- Direction Accuracy: {perf['direction_accuracy']:.1f}%\n")
            f.write(f"- Average Error: {perf['avg_error']:.3f}%\n")
            f.write(f"- Error Range: {perf['min_error']:.3f}% - {perf['max_error']:.3f}%\n")
    
    print(f"\nReports saved to evaluation_results/")
    
    # Return overall accuracy for CI/CD
    overall_accuracy = np.mean([performance[tf]['direction_accuracy'] for tf in ['1h', '3h', 'eod']])
    print(f"\nOverall Direction Accuracy: {overall_accuracy:.1f}%")
    
    # Check if this is the first run (no predictions evaluated)
    total_evaluated = sum(performance[tf]['total_predictions'] for tf in ['1h', '3h', 'eod'])
    
    if total_evaluated == 0:
        print("✓ First run - no past predictions to evaluate yet")
        print("  New predictions have been saved for future evaluation")
        return 0
    
    # Exit with appropriate code
    if overall_accuracy >= 45:
        print("✓ Performance meets threshold")
        return 0
    else:
        print("✗ Performance below threshold")
        return 1

if __name__ == "__main__":
    sys.exit(main())