#!/usr/bin/env python3
"""
Multi-Ticker Signal Analysis - Pattern-based Predictions
Generates intraday predictions for any ticker with sufficient historical data.

Fixed version that uses the actual intraday_shape_matcher functions.
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

from directional_analysis import intraday_shape_matcher
from ChartScanAI_Shiny.azure_utils import upload_to_azure, download_from_azure

def setup_logging(ticker):
    """Set up logging for the ticker"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{ticker}_predictions.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_ticker_config(ticker):
    """Load ticker-specific configuration"""
    # Default configuration
    config = {
        'evaluation_threshold': 0.3,
        'query_length': 30,  # Number of bars to look back
        'K': 10,  # Number of similar patterns to consider
        'interval': '15m',  # 15-minute bars
        'period': '60d',  # 60 days of history
        'confidence_boost': 1.0
    }
    
    # Ticker-specific adjustments
    ticker_configs = {
        'BTC-USD': {
            'evaluation_threshold': 0.5,
            'K': 15,
            'confidence_boost': 1.2
        },
        'TSLA': {
            'evaluation_threshold': 0.4,
            'K': 12,
            'confidence_boost': 1.1
        },
        'AAPL': {
            'evaluation_threshold': 0.25,
            'K': 10,
            'confidence_boost': 1.0
        },
        'MSFT': {
            'evaluation_threshold': 0.25,
            'K': 10,
            'confidence_boost': 1.0
        },
        'AC.TO': {
            'evaluation_threshold': 0.35,
            'K': 8,
            'confidence_boost': 0.9
        }
    }
    
    if ticker in ticker_configs:
        config.update(ticker_configs[ticker])
    
    return config

def generate_predictions(ticker, logger):
    """Generate predictions using the shape matcher"""
    
    config = load_ticker_config(ticker)
    logger.info(f"Using configuration: {config}")
    
    try:
        # Use the forecast_shape function from intraday_shape_matcher
        predictions = intraday_shape_matcher.forecast_shape(
            ticker=ticker,
            interval=config['interval'],
            period=config['period'],
            query_length=config['query_length'],
            K=config['K']
        )
        
        # Get current price for reference
        current_data = yf.Ticker(ticker).history(period='1d', interval='15m')
        if not current_data.empty:
            current_price = current_data['Close'].iloc[-1]
        else:
            current_price = None
        
        # Format predictions
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        
        formatted_predictions = {
            'ticker': ticker,
            'timestamp': current_time.isoformat(),
            'current_price': float(current_price) if current_price else None,
            'config_used': config,
            'predictions': predictions,
            'horizons': {
                '1h': {
                    'return': predictions.get('expected_return_1h', 0),
                    'confidence': predictions.get('confidence_1h', 0) * config['confidence_boost'],
                    'signal': get_signal(predictions.get('expected_return_1h', 0), config['evaluation_threshold'])
                },
                '3h': {
                    'return': predictions.get('expected_return_3h', 0),
                    'confidence': predictions.get('confidence_3h', 0) * config['confidence_boost'],
                    'signal': get_signal(predictions.get('expected_return_3h', 0), config['evaluation_threshold'])
                },
                'eod': {
                    'return': predictions.get('expected_return_eod', 0),
                    'confidence': predictions.get('confidence_eod', 0) * config['confidence_boost'],
                    'signal': get_signal(predictions.get('expected_return_eod', 0), config['evaluation_threshold'])
                }
            }
        }
        
        # Add overall recommendation
        signals = [h['signal'] for h in formatted_predictions['horizons'].values()]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            formatted_predictions['recommendation'] = 'BUY'
        elif sell_count > buy_count:
            formatted_predictions['recommendation'] = 'SELL'
        else:
            formatted_predictions['recommendation'] = 'HOLD'
        
        formatted_predictions['signal_counts'] = {
            'buy': buy_count,
            'sell': sell_count,
            'hold': signals.count('HOLD')
        }
        
        return formatted_predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise

def get_signal(expected_return, threshold):
    """Convert expected return to trading signal"""
    if expected_return > threshold:
        return 'BUY'
    elif expected_return < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

def save_predictions(predictions, ticker, logger):
    """Save predictions locally and to Azure"""
    
    # Create local directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save local copy
    timestamp = datetime.now(pytz.timezone('US/Eastern'))
    filename = f"output/{ticker}_predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    logger.info(f"Predictions saved locally: {filename}")
    
    # Upload to Azure
    try:
        date_str = timestamp.strftime('%Y-%m-%d')
        hour_str = timestamp.strftime('%H')
        azure_path = f"predictions/{ticker}/{date_str}/{hour_str}.json"
        
        success = upload_to_azure(json.dumps(predictions, default=str), azure_path)
        if success:
            logger.info(f"Predictions uploaded to Azure: {azure_path}")
        else:
            logger.warning("Failed to upload predictions to Azure")
            
    except Exception as e:
        logger.error(f"Azure upload error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate multi-ticker signal predictions')
    parser.add_argument('--ticker', required=True, help='Ticker symbol to analyze')
    parser.add_argument('--force', action='store_true', help='Force run regardless of market hours')
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    # Set up logging
    logger = setup_logging(ticker)
    logger.info(f"Starting signal analysis for {ticker}")
    
    # Check market hours unless forced
    if not args.force:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        if now.weekday() >= 5:  # Weekend
            logger.info("Market closed (weekend). Use --force to run anyway.")
            return
            
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if not (market_open <= now <= market_close):
            logger.info(f"Market closed. Current time: {now.strftime('%H:%M:%S %Z')}")
            logger.info("Use --force to run anyway.")
            return
    
    try:
        # Generate predictions
        predictions = generate_predictions(ticker, logger)
        
        # Save predictions
        save_predictions(predictions, ticker, logger)
        
        # Log summary
        logger.info(f"Analysis complete for {ticker}")
        logger.info(f"Recommendation: {predictions['recommendation']}")
        logger.info(f"Signals: {predictions['signal_counts']}")
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {str(e)}")
        raise

if __name__ == "__main__":
    main()