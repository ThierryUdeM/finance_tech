#!/usr/bin/env python3
"""
Multi-Ticker Signal Analysis - Prediction Evaluation
Evaluates historical predictions for any ticker after sufficient time has passed.

Based on the original NVDA-specific evaluation but made ticker-agnostic.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ChartScanAI_Shiny.azure_utils import upload_to_azure, download_from_azure

def setup_logging(ticker):
    """Set up logging for the ticker"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{ticker}_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_ticker_config(ticker):
    """Load ticker-specific configuration (same as in prediction script)"""
    config = {
        'evaluation_threshold': 0.3,
        'atr_period': 14,
        'min_pattern_library': 1000,
        'volatility_multiplier': 1.0
    }
    
    ticker_configs = {
        'BTC-USD': {
            'evaluation_threshold': 0.5,
            'volatility_multiplier': 2.0
        },
        'TSLA': {
            'evaluation_threshold': 0.4,
            'volatility_multiplier': 1.5
        },
        'AAPL': {
            'evaluation_threshold': 0.25,
            'volatility_multiplier': 0.8
        },
        'MSFT': {
            'evaluation_threshold': 0.25,
            'volatility_multiplier': 0.9
        },
        'AC.TO': {
            'evaluation_threshold': 0.35,
            'volatility_multiplier': 1.2
        }
    }
    
    if ticker in ticker_configs:
        config.update(ticker_configs[ticker])
    
    return config

def get_recent_predictions(ticker, days_back=7):
    """Get recent predictions from Azure storage"""
    predictions = []
    
    et_tz = pytz.timezone('US/Eastern')
    current_date = datetime.now(et_tz)
    
    for i in range(days_back):
        check_date = current_date - timedelta(days=i)
        date_str = check_date.strftime('%Y-%m-%d')
        
        # Check each hour of the trading day
        for hour in range(9, 17):  # 9 AM to 4 PM
            azure_path = f"predictions/{ticker}/{date_str}/{hour:02d}.json"
            
            try:
                prediction_data = download_from_azure(azure_path)
                if prediction_data:
                    prediction = json.loads(prediction_data)
                    prediction['azure_path'] = azure_path
                    predictions.append(prediction)
            except Exception:
                continue  # No prediction for this hour
    
    return predictions

def get_actual_prices(ticker, start_date, end_date):
    """Get actual price data for evaluation"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval="15m")
        
        if data.empty:
            return None
        
        data.reset_index(inplace=True)
        data.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to ET timezone
        et_tz = pytz.timezone('US/Eastern')
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_convert(et_tz)
        
        return data
        
    except Exception as e:
        print(f"Error fetching actual prices: {str(e)}")
        return None

def evaluate_prediction(prediction, actual_data, config, logger):
    """Evaluate a single prediction against actual outcomes"""
    pred_time = pd.to_datetime(prediction['timestamp'])
    current_price = prediction['current_price']
    
    evaluation = {
        'prediction_time': prediction['timestamp'],
        'ticker': prediction['ticker'],
        'current_price': current_price,
        'evaluation_threshold': config['evaluation_threshold']
    }
    
    # Evaluate each horizon
    horizons = {
        '1h': timedelta(hours=1),
        '3h': timedelta(hours=3),
        'eod': None  # End of day
    }
    
    for horizon, time_delta in horizons.items():
        pred_key = f'pred_{horizon}_pct'
        dir_key = f'pred_{horizon}_dir'
        
        if pred_key not in prediction or prediction[pred_key] is None:
            continue
        
        predicted_pct = prediction[pred_key]
        predicted_dir = prediction[dir_key]
        
        # Calculate target time
        if horizon == 'eod':
            # End of day is 4 PM ET
            target_time = pred_time.replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            target_time = pred_time + time_delta
        
        # Find actual price at target time (or closest)
        time_mask = actual_data['timestamp'] >= target_time
        if not time_mask.any():
            logger.warning(f"No actual data found for {horizon} evaluation")
            continue
            
        actual_price = actual_data[time_mask]['Close'].iloc[0]
        actual_pct = ((actual_price - current_price) / current_price) * 100
        
        # Determine actual direction
        if actual_pct > config['evaluation_threshold']:
            actual_dir = 'BULLISH'
        elif actual_pct < -config['evaluation_threshold']:
            actual_dir = 'BEARISH'
        else:
            actual_dir = 'NEUTRAL'
        
        # Calculate accuracy
        direction_correct = (predicted_dir == actual_dir)
        prediction_error = abs(predicted_pct - actual_pct)
        
        evaluation[f'{horizon}_predicted_pct'] = predicted_pct
        evaluation[f'{horizon}_actual_pct'] = round(actual_pct, 3)
        evaluation[f'{horizon}_predicted_dir'] = predicted_dir
        evaluation[f'{horizon}_actual_dir'] = actual_dir
        evaluation[f'{horizon}_direction_correct'] = direction_correct
        evaluation[f'{horizon}_prediction_error'] = round(prediction_error, 3)
        evaluation[f'{horizon}_actual_price'] = round(actual_price, 2)
        
        logger.info(f"{horizon}: Predicted {predicted_dir} ({predicted_pct:+.2f}%), "
                   f"Actual {actual_dir} ({actual_pct:+.2f}%) - {'✅' if direction_correct else '❌'}")
    
    return evaluation

def calculate_performance_metrics(evaluations, logger):
    """Calculate overall performance metrics"""
    if not evaluations:
        return {}
    
    metrics = {
        'total_evaluations': len(evaluations),
        'evaluation_period': {
            'start': min(e['prediction_time'] for e in evaluations),
            'end': max(e['prediction_time'] for e in evaluations)
        }
    }
    
    horizons = ['1h', '3h', 'eod']
    
    for horizon in horizons:
        direction_key = f'{horizon}_direction_correct'
        error_key = f'{horizon}_prediction_error'
        
        # Filter evaluations that have this horizon
        horizon_evals = [e for e in evaluations if direction_key in e]
        
        if not horizon_evals:
            continue
        
        # Direction accuracy
        correct_directions = sum(1 for e in horizon_evals if e[direction_key])
        direction_accuracy = correct_directions / len(horizon_evals) * 100
        
        # Average prediction error
        errors = [e[error_key] for e in horizon_evals if error_key in e]
        avg_error = np.mean(errors) if errors else 0
        
        # Count by predicted direction
        bullish_correct = sum(1 for e in horizon_evals 
                            if e.get(f'{horizon}_predicted_dir') == 'BULLISH' and e[direction_key])
        bearish_correct = sum(1 for e in horizon_evals 
                            if e.get(f'{horizon}_predicted_dir') == 'BEARISH' and e[direction_key])
        neutral_correct = sum(1 for e in horizon_evals 
                            if e.get(f'{horizon}_predicted_dir') == 'NEUTRAL' and e[direction_key])
        
        bullish_total = sum(1 for e in horizon_evals 
                          if e.get(f'{horizon}_predicted_dir') == 'BULLISH')
        bearish_total = sum(1 for e in horizon_evals 
                          if e.get(f'{horizon}_predicted_dir') == 'BEARISH')
        neutral_total = sum(1 for e in horizon_evals 
                          if e.get(f'{horizon}_predicted_dir') == 'NEUTRAL')
        
        metrics[f'{horizon}_accuracy'] = round(direction_accuracy, 2)
        metrics[f'{horizon}_avg_error'] = round(avg_error, 3)
        metrics[f'{horizon}_total_predictions'] = len(horizon_evals)
        metrics[f'{horizon}_bullish_accuracy'] = round(bullish_correct / bullish_total * 100, 2) if bullish_total > 0 else 0
        metrics[f'{horizon}_bearish_accuracy'] = round(bearish_correct / bearish_total * 100, 2) if bearish_total > 0 else 0
        metrics[f'{horizon}_neutral_accuracy'] = round(neutral_correct / neutral_total * 100, 2) if neutral_total > 0 else 0
        
        logger.info(f"{horizon} Performance: {direction_accuracy:.1f}% accuracy, "
                   f"{avg_error:.2f}% avg error ({len(horizon_evals)} predictions)")
    
    return metrics

def save_evaluation_results(evaluations, metrics, ticker, logger):
    """Save evaluation results locally and to Azure"""
    
    # Create results dictionary
    results = {
        'ticker': ticker,
        'evaluation_timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        'performance_metrics': metrics,
        'individual_evaluations': evaluations
    }
    
    # Save local copy
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now(pytz.timezone('US/Eastern'))
    filename = f"output/{ticker}_evaluation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved locally: {filename}")
    
    # Upload to Azure
    try:
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H%M%S')
        azure_path = f"evaluations/{ticker}/{date_str}/{time_str}.json"
        
        success = upload_to_azure(json.dumps(results, default=str), azure_path)
        if success:
            logger.info(f"Results uploaded to Azure: {azure_path}")
        else:
            logger.warning("Failed to upload results to Azure")
            
    except Exception as e:
        logger.error(f"Azure upload error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-ticker signal predictions')
    parser.add_argument('--ticker', required=True, help='Ticker symbol to evaluate')
    parser.add_argument('--days', type=int, default=7, help='Days back to evaluate (default: 7)')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging(ticker)
    
    try:
        logger.info(f"Starting evaluation for {ticker} (last {args.days} days)")
        
        # Load configuration
        config = load_ticker_config(ticker)
        
        # Get recent predictions
        logger.info("Fetching recent predictions from Azure...")
        predictions = get_recent_predictions(ticker, args.days)
        logger.info(f"Found {len(predictions)} predictions to evaluate")
        
        if not predictions:
            logger.info("No predictions found to evaluate")
            return
        
        # Get date range for actual data
        pred_times = [pd.to_datetime(p['timestamp']) for p in predictions]
        start_date = min(pred_times).date()
        end_date = (max(pred_times) + timedelta(days=1)).date()
        
        # Get actual price data
        logger.info(f"Fetching actual prices from {start_date} to {end_date}...")
        actual_data = get_actual_prices(ticker, start_date, end_date)
        
        if actual_data is None or actual_data.empty:
            logger.error("Could not fetch actual price data")
            return
        
        logger.info(f"Loaded {len(actual_data)} actual price records")
        
        # Evaluate each prediction
        evaluations = []
        for i, prediction in enumerate(predictions):
            logger.info(f"Evaluating prediction {i+1}/{len(predictions)}")
            
            try:
                evaluation = evaluate_prediction(prediction, actual_data, config, logger)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Error evaluating prediction: {str(e)}")
                continue
        
        if not evaluations:
            logger.warning("No successful evaluations completed")
            return
        
        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        metrics = calculate_performance_metrics(evaluations, logger)
        
        # Save results
        save_evaluation_results(evaluations, metrics, ticker, logger)
        
        logger.info(f"✅ Evaluation completed successfully for {ticker}")
        logger.info(f"   Total evaluations: {len(evaluations)}")
        
        # Log key metrics
        for horizon in ['1h', '3h', 'eod']:
            accuracy_key = f'{horizon}_accuracy'
            if accuracy_key in metrics:
                logger.info(f"   {horizon} accuracy: {metrics[accuracy_key]}%")
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation for {ticker}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()