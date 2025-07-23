#!/usr/bin/env python3
"""
Multi-Ticker Signal Analysis - Pattern-based Predictions
Generates intraday predictions for any ticker with sufficient historical data.

Based on the original NVDA-specific signal analysis but made ticker-agnostic.
Requires 2000+ historical records (approximately 2-3 months of 15-minute data).
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

from directional_analysis.intraday_shape_matcher import ShapeMatcher
from ChartScanAI_Shiny.azure_utils import upload_to_azure

def setup_logging(ticker):
    """Set up logging for the ticker"""
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
        'atr_period': 14,
        'min_pattern_library': 1000,
        'volatility_multiplier': 1.0,
        'confidence_boost': 1.0
    }
    
    # Ticker-specific adjustments
    ticker_configs = {
        'BTC-USD': {
            'evaluation_threshold': 0.5,
            'volatility_multiplier': 2.0,
            'confidence_boost': 1.2
        },
        'TSLA': {
            'evaluation_threshold': 0.4,
            'volatility_multiplier': 1.5,
            'confidence_boost': 1.1
        },
        'AAPL': {
            'evaluation_threshold': 0.25,
            'volatility_multiplier': 0.8,
            'confidence_boost': 1.0
        },
        'MSFT': {
            'evaluation_threshold': 0.25,
            'volatility_multiplier': 0.9,
            'confidence_boost': 1.0
        },
        'AC.TO': {
            'evaluation_threshold': 0.35,
            'volatility_multiplier': 1.2,
            'confidence_boost': 0.9
        }
    }
    
    if ticker in ticker_configs:
        config.update(ticker_configs[ticker])
    
    return config

def load_historical_data(ticker):
    """Load historical data for the ticker"""
    data_file = f'data/{ticker}_15min_pattern_ready.csv'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Historical data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Validate data quality
    required_columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data quantity
    min_records = load_ticker_config(ticker)['min_pattern_library']
    if len(df) < min_records:
        raise ValueError(f"Insufficient historical data: {len(df)} records < {min_records} required")
    
    return df

def get_current_market_data(ticker):
    """Fetch current market data for prediction"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get current price and recent data
        current_data = stock.history(period="5d", interval="15m")
        
        if current_data.empty:
            raise ValueError(f"No current market data available for {ticker}")
        
        # Convert to our standard format
        current_data.reset_index(inplace=True)
        current_data.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert timezone to Eastern
        et_tz = pytz.timezone('US/Eastern')
        current_data['timestamp'] = pd.to_datetime(current_data['timestamp']).dt.tz_convert(et_tz)
        
        # Filter to regular market hours
        market_hours = (
            (current_data['timestamp'].dt.hour > 9) |
            ((current_data['timestamp'].dt.hour == 9) & (current_data['timestamp'].dt.minute >= 30))
        ) & (current_data['timestamp'].dt.hour < 16)
        
        weekdays = current_data['timestamp'].dt.weekday < 5
        current_data = current_data[market_hours & weekdays].copy()
        
        return current_data.sort_values('timestamp').reset_index(drop=True)
        
    except Exception as e:
        raise RuntimeError(f"Error fetching current market data for {ticker}: {str(e)}")

def check_market_status():
    """Check if market is currently open"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False, "Weekend"
    
    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if market_open <= now <= market_close:
        return True, "Market Open"
    elif now < market_open:
        return False, "Pre-Market"
    else:
        return False, "After-Hours"

def generate_predictions(ticker, logger):
    """Generate predictions for the specified ticker"""
    logger.info(f"Starting prediction generation for {ticker}")
    
    # Load configuration
    config = load_ticker_config(ticker)
    logger.info(f"Configuration loaded: {config}")
    
    # Load historical data
    logger.info("Loading historical data...")
    historical_data = load_historical_data(ticker)
    logger.info(f"Loaded {len(historical_data)} historical records")
    
    # Get current market data
    logger.info("Fetching current market data...")
    current_data = get_current_market_data(ticker)
    
    if len(current_data) < 20:
        logger.warning(f"Limited current data: {len(current_data)} records")
    
    # Initialize pattern matcher
    logger.info("Initializing pattern matcher...")
    matcher = ShapeMatcher(
        data=historical_data,
        atr_period=config['atr_period'],
        volatility_multiplier=config['volatility_multiplier']
    )
    
    # Generate predictions for different horizons
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    current_price = current_data['Close'].iloc[-1]
    
    predictions = {
        'ticker': ticker,
        'timestamp': current_time.isoformat(),
        'current_price': float(current_price),
        'config_used': config,
        'historical_records': len(historical_data),
        'current_records': len(current_data)
    }
    
    # Generate predictions for 1h, 3h, and EOD
    horizons = {
        '1h': {'lookback': 8, 'forward': 4},   # 2 hours back, 1 hour forward
        '3h': {'lookback': 12, 'forward': 12}, # 3 hours back, 3 hours forward  
        'eod': {'lookback': 20, 'forward': -1} # 5 hours back, end of day
    }
    
    for horizon, params in horizons.items():
        try:
            logger.info(f"Generating {horizon} prediction...")
            
            # Get recent price pattern
            if len(current_data) >= params['lookback']:
                recent_pattern = current_data['Close'].iloc[-params['lookback']:].values
                
                # Find similar patterns and predict
                prediction_result = matcher.predict_direction(
                    recent_pattern,
                    forward_periods=params['forward'] if params['forward'] > 0 else None
                )
                
                # Calculate percentage prediction
                if prediction_result['predicted_return'] is not None:
                    pred_pct = prediction_result['predicted_return'] * 100
                    pred_price = current_price * (1 + prediction_result['predicted_return'])
                    
                    # Determine direction
                    if pred_pct > config['evaluation_threshold']:
                        direction = 'BULLISH'
                    elif pred_pct < -config['evaluation_threshold']:
                        direction = 'BEARISH'
                    else:
                        direction = 'NEUTRAL'
                    
                    predictions[f'pred_{horizon}_pct'] = round(pred_pct, 3)
                    predictions[f'pred_{horizon}_price'] = round(pred_price, 2)
                    predictions[f'pred_{horizon}_dir'] = direction
                    predictions[f'confidence_{horizon}'] = prediction_result.get('confidence', 0.5)
                    predictions[f'patterns_used_{horizon}'] = prediction_result.get('patterns_found', 0)
                    
                    logger.info(f"{horizon} prediction: {direction} ({pred_pct:+.2f}%)")
                else:
                    logger.warning(f"No prediction generated for {horizon}")
                    predictions[f'pred_{horizon}_pct'] = None
                    predictions[f'pred_{horizon}_dir'] = 'UNKNOWN'
                    
            else:
                logger.warning(f"Insufficient current data for {horizon} prediction")
                predictions[f'pred_{horizon}_pct'] = None
                predictions[f'pred_{horizon}_dir'] = 'INSUFFICIENT_DATA'
                
        except Exception as e:
            logger.error(f"Error generating {horizon} prediction: {str(e)}")
            predictions[f'pred_{horizon}_pct'] = None
            predictions[f'pred_{horizon}_dir'] = 'ERROR'
    
    # Overall confidence and recommendation
    confident_predictions = [
        p for p in [predictions.get(f'confidence_{h}', 0) for h in horizons.keys()]
        if p is not None and p > 0.6
    ]
    
    overall_confidence = np.mean(confident_predictions) if confident_predictions else 0.3
    predictions['overall_confidence'] = round(overall_confidence * config['confidence_boost'], 3)
    
    # Generate overall recommendation
    bullish_count = sum(1 for h in horizons.keys() if predictions.get(f'pred_{h}_dir') == 'BULLISH')
    bearish_count = sum(1 for h in horizons.keys() if predictions.get(f'pred_{h}_dir') == 'BEARISH')
    
    if bullish_count > bearish_count:
        predictions['recommendation'] = 'BUY'
    elif bearish_count > bullish_count:
        predictions['recommendation'] = 'SELL' 
    else:
        predictions['recommendation'] = 'HOLD'
    
    logger.info(f"Overall recommendation: {predictions['recommendation']} (confidence: {predictions['overall_confidence']})")
    
    return predictions

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
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging(ticker)
    
    try:
        # Check market status
        market_open, market_status = check_market_status()
        logger.info(f"Market status: {market_status}")
        
        if not market_open and not args.force:
            logger.info("Market is closed and --force not specified. Exiting.")
            return
        
        # Generate predictions
        predictions = generate_predictions(ticker, logger)
        
        # Save results
        save_predictions(predictions, ticker, logger)
        
        logger.info(f"✅ Predictions generated successfully for {ticker}")
        
    except Exception as e:
        logger.error(f"❌ Error generating predictions for {ticker}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()