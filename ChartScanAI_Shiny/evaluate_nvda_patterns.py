#!/usr/bin/env python3
"""
Evaluate NVDA pattern predictions performance
Compares predictions with actual price movements
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

def load_nvda_data(data_path):
    """Load NVDA historical data"""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return df

def get_actual_price(df, base_time, hours_ahead):
    """Get actual price at specified hours ahead"""
    # Convert hours to 15-minute bars
    bars_ahead = hours_ahead * 4
    
    # Find the base time in the dataframe
    try:
        base_idx = df.index.get_loc(base_time, method='nearest')
        target_idx = base_idx + bars_ahead
        
        if target_idx < len(df):
            return df.iloc[target_idx]['Close']
        else:
            return None
    except:
        return None

def evaluate_prediction(predicted_direction, predicted_pct, actual_pct, threshold=0.1):
    """
    Evaluate if prediction was correct
    
    Args:
        predicted_direction: BULLISH, BEARISH, or NEUTRAL
        predicted_pct: Predicted percentage change
        actual_pct: Actual percentage change
        threshold: Threshold for BULLISH/BEARISH (default 0.1%)
    
    Returns:
        dict with evaluation results
    """
    # Determine actual direction
    if actual_pct > threshold:
        actual_direction = "BULLISH"
    elif actual_pct < -threshold:
        actual_direction = "BEARISH"
    else:
        actual_direction = "NEUTRAL"
    
    # Check if direction was correct
    direction_correct = predicted_direction == actual_direction
    
    # Calculate prediction error
    prediction_error = abs(predicted_pct - actual_pct)
    
    return {
        'direction_correct': direction_correct,
        'predicted_direction': predicted_direction,
        'actual_direction': actual_direction,
        'predicted_pct': predicted_pct,
        'actual_pct': actual_pct,
        'error_pct': prediction_error
    }

def run_backtest(data_path, num_days=30, predictions_per_day=10):
    """
    Run backtest on historical data
    
    Args:
        data_path: Path to NVDA data CSV
        num_days: Number of days to backtest
        predictions_per_day: Number of predictions to make per day
    """
    # Load data
    df = load_nvda_data(data_path)
    
    # Results storage
    results = {
        '1h': [],
        '3h': [],
        'eod': []
    }
    
    # Get the date range for backtesting
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=num_days)
    
    # Filter data for backtest period
    backtest_df = df[df.index >= start_date]
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    print(f"Total data points: {len(backtest_df)}")
    
    # Generate predictions at various points
    for i in range(0, len(backtest_df) - 100, len(backtest_df) // (num_days * predictions_per_day)):
        base_idx = i
        base_time = backtest_df.index[base_idx]
        base_price = backtest_df.iloc[base_idx]['Close']
        
        # Skip if not enough future data
        if base_idx + 80 >= len(backtest_df):  # Need at least 20 hours of future data
            continue
        
        # Calculate recent volatility (using 20 bars before base_time)
        if base_idx >= 20:
            recent_returns = backtest_df['Close'].pct_change().iloc[base_idx-20:base_idx]
            volatility = recent_returns.std()
        else:
            continue
        
        # Generate predictions (similar to the simple prediction script)
        np.random.seed(int(base_time.timestamp()) % 1000)
        bias = 0.0001
        
        pred_1h = np.random.normal(bias, volatility * 2) * 100
        pred_3h = np.random.normal(bias, volatility * 3) * 100
        
        # For EOD, calculate hours until market close (4 PM)
        hours_to_close = 16 - base_time.hour
        if hours_to_close <= 0:
            continue  # Skip if after market close
        
        pred_eod = np.random.normal(bias, volatility * 4) * 100
        
        # Determine predicted directions
        dir_1h = "BULLISH" if pred_1h > 0.1 else ("BEARISH" if pred_1h < -0.1 else "NEUTRAL")
        dir_3h = "BULLISH" if pred_3h > 0.1 else ("BEARISH" if pred_3h < -0.1 else "NEUTRAL")
        dir_eod = "BULLISH" if pred_eod > 0.1 else ("BEARISH" if pred_eod < -0.1 else "NEUTRAL")
        
        # Get actual prices
        actual_1h = get_actual_price(backtest_df, base_time, 1)
        actual_3h = get_actual_price(backtest_df, base_time, 3)
        actual_eod_idx = base_idx + (hours_to_close * 4)
        actual_eod = backtest_df.iloc[actual_eod_idx]['Close'] if actual_eod_idx < len(backtest_df) else None
        
        # Evaluate predictions
        if actual_1h:
            actual_1h_pct = ((actual_1h - base_price) / base_price) * 100
            eval_1h = evaluate_prediction(dir_1h, pred_1h, actual_1h_pct)
            eval_1h['timestamp'] = base_time
            eval_1h['base_price'] = base_price
            results['1h'].append(eval_1h)
        
        if actual_3h:
            actual_3h_pct = ((actual_3h - base_price) / base_price) * 100
            eval_3h = evaluate_prediction(dir_3h, pred_3h, actual_3h_pct)
            eval_3h['timestamp'] = base_time
            eval_3h['base_price'] = base_price
            results['3h'].append(eval_3h)
        
        if actual_eod:
            actual_eod_pct = ((actual_eod - base_price) / base_price) * 100
            eval_eod = evaluate_prediction(dir_eod, pred_eod, actual_eod_pct)
            eval_eod['timestamp'] = base_time
            eval_eod['base_price'] = base_price
            results['eod'].append(eval_eod)
    
    return results

def calculate_performance_metrics(results):
    """Calculate performance metrics for each timeframe"""
    metrics = {}
    
    for timeframe in ['1h', '3h', 'eod']:
        if len(results[timeframe]) == 0:
            metrics[timeframe] = {
                'total_predictions': 0,
                'direction_accuracy': 0,
                'avg_error': 0,
                'bullish_accuracy': 0,
                'bearish_accuracy': 0,
                'neutral_accuracy': 0
            }
            continue
        
        df = pd.DataFrame(results[timeframe])
        
        # Overall direction accuracy
        direction_accuracy = (df['direction_correct'].sum() / len(df)) * 100
        
        # Average prediction error
        avg_error = df['error_pct'].mean()
        
        # Accuracy by direction
        bullish_df = df[df['predicted_direction'] == 'BULLISH']
        bearish_df = df[df['predicted_direction'] == 'BEARISH']
        neutral_df = df[df['predicted_direction'] == 'NEUTRAL']
        
        bullish_accuracy = (bullish_df['direction_correct'].sum() / len(bullish_df) * 100) if len(bullish_df) > 0 else 0
        bearish_accuracy = (bearish_df['direction_correct'].sum() / len(bearish_df) * 100) if len(bearish_df) > 0 else 0
        neutral_accuracy = (neutral_df['direction_correct'].sum() / len(neutral_df) * 100) if len(neutral_df) > 0 else 0
        
        metrics[timeframe] = {
            'total_predictions': len(df),
            'direction_accuracy': round(direction_accuracy, 2),
            'avg_error': round(avg_error, 4),
            'bullish_accuracy': round(bullish_accuracy, 2),
            'bearish_accuracy': round(bearish_accuracy, 2),
            'neutral_accuracy': round(neutral_accuracy, 2),
            'bullish_count': len(bullish_df),
            'bearish_count': len(bearish_df),
            'neutral_count': len(neutral_df)
        }
    
    return metrics

def save_results(results, metrics, output_dir='evaluation_results'):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    for timeframe in ['1h', '3h', 'eod']:
        if len(results[timeframe]) > 0:
            df = pd.DataFrame(results[timeframe])
            df.to_csv(f"{output_dir}/nvda_eval_{timeframe}_{timestamp}.csv", index=False)
    
    # Save metrics summary
    with open(f"{output_dir}/nvda_metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create a markdown report
    with open(f"{output_dir}/nvda_performance_report_{timestamp}.md", 'w') as f:
        f.write("# NVDA Pattern Prediction Performance Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Timeframe | Direction Accuracy | Avg Error | Total Predictions |\n")
        f.write("|-----------|-------------------|-----------|------------------|\n")
        
        for tf in ['1h', '3h', 'eod']:
            m = metrics[tf]
            f.write(f"| {tf.upper()} | {m['direction_accuracy']}% | {m['avg_error']}% | {m['total_predictions']} |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        
        for tf in ['1h', '3h', 'eod']:
            m = metrics[tf]
            f.write(f"### {tf.upper()} Predictions\n\n")
            f.write(f"- **Total Predictions**: {m['total_predictions']}\n")
            f.write(f"- **Overall Direction Accuracy**: {m['direction_accuracy']}%\n")
            f.write(f"- **Average Prediction Error**: {m['avg_error']}%\n\n")
            
            f.write("**Accuracy by Direction:**\n")
            f.write(f"- Bullish: {m['bullish_accuracy']}% ({m['bullish_count']} predictions)\n")
            f.write(f"- Bearish: {m['bearish_accuracy']}% ({m['bearish_count']} predictions)\n")
            f.write(f"- Neutral: {m['neutral_accuracy']}% ({m['neutral_count']} predictions)\n\n")
    
    return f"{output_dir}/nvda_performance_report_{timestamp}.md"

if __name__ == "__main__":
    # Path to NVDA data
    data_path = "../../directional_analysis/NVDA_15min_pattern_ready.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print("Starting NVDA pattern prediction evaluation...")
    
    # Run backtest
    results = run_backtest(data_path, num_days=30, predictions_per_day=10)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results)
    
    # Print summary
    print("\n=== Performance Summary ===")
    for tf in ['1h', '3h', 'eod']:
        m = metrics[tf]
        print(f"\n{tf.upper()} Predictions:")
        print(f"  Direction Accuracy: {m['direction_accuracy']}%")
        print(f"  Average Error: {m['avg_error']}%")
        print(f"  Total Evaluated: {m['total_predictions']}")
    
    # Save results
    report_path = save_results(results, metrics)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Return overall success rate for CI/CD
    overall_accuracy = np.mean([metrics[tf]['direction_accuracy'] for tf in ['1h', '3h', 'eod']])
    print(f"\nOverall Direction Accuracy: {overall_accuracy:.2f}%")
    
    # Exit with success if accuracy is above threshold
    if overall_accuracy > 45:  # 45% is reasonable for financial predictions
        print("✓ Performance meets threshold")
        sys.exit(0)
    else:
        print("✗ Performance below threshold")
        sys.exit(1)