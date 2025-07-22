#!/usr/bin/env python3
"""
Daily Pattern Scanner for Next-Day Trading Signals
Scans daily candlestick and chart patterns at market close
Generates predictions for next trading day
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required libraries
try:
    import talib
    logger.info(f"TA-Lib loaded successfully (version {talib.__version__})")
except ImportError as e:
    logger.error(f"CRITICAL: TA-Lib is required but not available: {e}")
    raise ImportError("TA-Lib is required. Please install it using: pip install TA-Lib")

try:
    from tradingpatterns.tradingpatterns import (
        detect_head_shoulder,
        detect_double_top_bottom,
        detect_triangle_pattern
    )
    TRADINGPATTERNS_AVAILABLE = True
    logger.info("tradingpattern library loaded successfully")
except ImportError as e:
    TRADINGPATTERNS_AVAILABLE = False
    logger.warning(f"tradingpatterns library not available: {e}")
    logger.info("Will use TA-Lib only for pattern detection")

# Load Azure credentials
load_dotenv('config/.env')

class DailyPatternScanner:
    def __init__(self, ticker=None):
        # Azure configuration
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_key = os.getenv('AZURE_STORAGE_KEY')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        # Ticker configuration
        self.ticker = ticker or os.getenv('TICKER', 'NVDA')
        
        # Check for mode override
        self.pattern_mode = os.getenv('PATTERN_MODE', 'daily')
        self.azure_folder = f"next_day_technical/{self.ticker.lower()}"
        
        if not all([self.storage_account, self.storage_key, self.container_name]):
            raise ValueError("Azure storage credentials not found")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.storage_key
        )
        
        # Ticker-specific volatility configurations
        self.ticker_configs = {
            'NVDA': {
                'name': 'NVIDIA Corporation',
                'volatility_threshold': 2.0,
                'volume_multiplier': 1.5,
                'pattern_confidence_boost': 1.0
            },
            'AAPL': {
                'name': 'Apple Inc.',
                'volatility_threshold': 1.2,
                'volume_multiplier': 1.3,
                'pattern_confidence_boost': 0.9
            },
            'MSFT': {
                'name': 'Microsoft Corporation',
                'volatility_threshold': 1.3,
                'volume_multiplier': 1.4,
                'pattern_confidence_boost': 0.9
            },
            'TSLA': {
                'name': 'Tesla Inc.',
                'volatility_threshold': 2.5,
                'volume_multiplier': 1.6,
                'pattern_confidence_boost': 1.1
            },
            'BTC-USD': {
                'name': 'Bitcoin',
                'volatility_threshold': 3.0,
                'volume_multiplier': 1.2,
                'pattern_confidence_boost': 1.2
            },
            'AC.TO': {
                'name': 'Air Canada',
                'volatility_threshold': 2.0,
                'volume_multiplier': 1.4,
                'pattern_confidence_boost': 1.0
            }
        }
        
        # Get current ticker config
        self.current_config = self.ticker_configs.get(self.ticker, self.ticker_configs['NVDA'])
        
        # Pattern holding periods (in days) - can be ticker-specific
        self.pattern_holding_periods = {
            'Hammer': 2,
            'Inverted Hammer': 2,
            'Shooting Star': 2,
            'Doji': 1,
            'Engulfing Pattern': 3,
            'Harami': 2,
            'Morning Star': 3,
            'Evening Star': 3,
            'Three White Soldiers': 5,
            'Three Black Crows': 5,
            'Head and Shoulder': 5,
            'Inverse Head and Shoulder': 5,
            'Double Top': 5,
            'Double Bottom': 5,
            'Ascending Triangle': 3,
            'Descending Triangle': 3
        }
    
    def clean_data(self, data):
        """Clean and prepare data for analysis"""
        cleaned = data.copy()
        
        # Handle multi-index columns
        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = cleaned.columns.get_level_values(0)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        
        cleaned.dropna(inplace=True)
        
        # Ensure datetime index
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index)
        
        return cleaned
    
    def detect_candlestick_patterns(self, data):
        """Detect candlestick patterns using TA-Lib"""
        patterns = []
        
        # Extract OHLC arrays
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Key candlestick patterns for daily trading
        pattern_functions = {
            'CDLDOJI': ('Doji', 'neutral'),
            'CDLHAMMER': ('Hammer', 'bullish'),
            'CDLINVERTEDHAMMER': ('Inverted Hammer', 'bullish'),
            'CDLSHOOTINGSTAR': ('Shooting Star', 'bearish'),
            'CDLENGULFING': ('Engulfing Pattern', 'reversal'),
            'CDLHARAMI': ('Harami', 'reversal'),
            'CDLMORNINGSTAR': ('Morning Star', 'bullish'),
            'CDLEVENINGSTAR': ('Evening Star', 'bearish'),
            'CDL3WHITESOLDIERS': ('Three White Soldiers', 'bullish'),
            'CDL3BLACKCROWS': ('Three Black Crows', 'bearish')
        }
        
        for func_name, (pattern_name, signal_type) in pattern_functions.items():
            try:
                pattern_func = getattr(talib, func_name)
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                
                # Find patterns (non-zero values)
                pattern_indices = np.where(result != 0)[0]
                
                for idx in pattern_indices:
                    if idx >= 0 and idx < len(data):
                        pattern_value = result[idx]
                        
                        # Only take strong patterns on daily charts
                        if abs(pattern_value) >= 100:  # Strong pattern threshold
                            signal = 'bullish' if pattern_value > 0 else 'bearish'
                            if signal_type == 'reversal':
                                # For reversal patterns, consider prior trend
                                prior_trend = 'up' if close_prices[idx] > close_prices[max(0, idx-5)] else 'down'
                                signal = 'bearish' if prior_trend == 'up' else 'bullish'
                            elif signal_type == 'neutral':
                                signal = 'neutral'
                            else:
                                signal = signal_type
                            
                            # Apply ticker-specific confidence boost
                            base_strength = abs(pattern_value) / 100
                            adjusted_strength = min(1.0, base_strength * self.current_config['pattern_confidence_boost'])
                            
                            patterns.append({
                                'timestamp': data.index[idx],
                                'pattern': pattern_name,
                                'type': 'candlestick',
                                'signal': signal,
                                'strength': adjusted_strength,
                                'price': close_prices[idx],
                                'volume': data['Volume'].iloc[idx],
                                'holding_days': self.pattern_holding_periods.get(pattern_name, 3),
                                'ticker': self.ticker
                            })
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")
                continue
        
        return patterns
    
    def detect_chart_patterns(self, data):
        """Detect chart patterns using tradingpatterns library or TA-Lib fallback"""
        patterns = []
        
        if not TRADINGPATTERNS_AVAILABLE:
            logger.info("Using TA-Lib-based pattern detection fallback")
            return self.detect_chart_patterns_talib_fallback(data)
        
        # Prepare data for pattern detection
        data_dict = {
            'Date': data.index,
            'Open': data['Open'].values,
            'High': data['High'].values,
            'Low': data['Low'].values,
            'Close': data['Close'].values,
            'Volume': data['Volume'].values
        }
        pattern_df = pd.DataFrame(data_dict)
        
        # Head and Shoulders
        try:
            hs_patterns = detect_head_shoulder(pattern_df)
            if hs_patterns is not None and len(hs_patterns) > 0:
                for _, pattern in hs_patterns.iterrows():
                    pattern_name = pattern['Pattern']
                    signal = 'bearish' if pattern_name == 'Head and Shoulder' else 'bullish'
                    
                    # Apply ticker-specific confidence boost
                    base_strength = 0.8  # Chart patterns are generally strong
                    adjusted_strength = min(1.0, base_strength * self.current_config['pattern_confidence_boost'])
                    
                    patterns.append({
                        'timestamp': pattern['Date'],
                        'pattern': pattern_name,
                        'type': 'chart',
                        'signal': signal,
                        'strength': adjusted_strength,
                        'price': data.loc[pattern['Date'], 'Close'],
                        'volume': data.loc[pattern['Date'], 'Volume'],
                        'holding_days': self.pattern_holding_periods.get(pattern_name, 5),
                        'ticker': self.ticker
                    })
        except Exception as e:
            logger.error(f"Error detecting Head & Shoulders: {e}")
        
        # Double Top/Bottom
        try:
            dt_patterns = detect_double_top_bottom(pattern_df)
            if dt_patterns is not None and len(dt_patterns) > 0:
                for _, pattern in dt_patterns.iterrows():
                    pattern_name = pattern['Pattern']
                    signal = 'bearish' if pattern_name == 'Double Top' else 'bullish'
                    
                    # Apply ticker-specific confidence boost
                    base_strength = 0.8
                    adjusted_strength = min(1.0, base_strength * self.current_config['pattern_confidence_boost'])
                    
                    patterns.append({
                        'timestamp': pattern['Date'],
                        'pattern': pattern_name,
                        'type': 'chart',
                        'signal': signal,
                        'strength': adjusted_strength,
                        'price': data.loc[pattern['Date'], 'Close'],
                        'volume': data.loc[pattern['Date'], 'Volume'],
                        'holding_days': self.pattern_holding_periods.get(pattern_name, 5),
                        'ticker': self.ticker
                    })
        except Exception as e:
            logger.error(f"Error detecting Double Top/Bottom: {e}")
        
        # Triangle patterns
        try:
            tri_patterns = detect_triangle_pattern(pattern_df)
            if tri_patterns is not None and len(tri_patterns) > 0:
                for _, pattern in tri_patterns.iterrows():
                    pattern_name = pattern['Pattern']
                    signal = 'bullish' if 'Ascending' in pattern_name else 'bearish'
                    
                    # Apply ticker-specific confidence boost
                    base_strength = 0.7
                    adjusted_strength = min(1.0, base_strength * self.current_config['pattern_confidence_boost'])
                    
                    patterns.append({
                        'timestamp': pattern['Date'],
                        'pattern': pattern_name,
                        'type': 'chart',
                        'signal': signal,
                        'strength': adjusted_strength,
                        'price': data.loc[pattern['Date'], 'Close'],
                        'volume': data.loc[pattern['Date'], 'Volume'],
                        'holding_days': self.pattern_holding_periods.get(pattern_name, 3),
                        'ticker': self.ticker
                    })
        except Exception as e:
            logger.error(f"Error detecting Triangle patterns: {e}")
        
        return patterns
    
    def detect_chart_patterns_talib_fallback(self, data):
        """Alternative chart pattern detection using TA-Lib indicators"""
        patterns = []
        
        if len(data) < 50:  # Need sufficient data for pattern analysis
            return patterns
        
        # Extract price data
        close_prices = data['Close'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # Simple trend-based pattern detection using moving averages and RSI
        try:
            # Calculate technical indicators for pattern identification
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            rsi = talib.RSI(close_prices, timeperiod=14)
            
            # Look for potential reversal patterns in the last 5 days
            for i in range(len(data) - 5, len(data)):
                if i < 50:  # Need enough data for indicators
                    continue
                
                current_price = close_prices[i]
                current_rsi = rsi[i] if not np.isnan(rsi[i]) else 50
                
                # Detect potential double top (bearish reversal)
                if (current_price > sma_20[i] and current_rsi > 70 and
                    i > 10 and current_price > close_prices[i-5:i].max() * 0.98):
                    patterns.append({
                        'timestamp': data.index[i],
                        'pattern': 'Potential Double Top',
                        'type': 'chart',
                        'signal': 'bearish',
                        'strength': min(1.0, 0.6 * self.current_config['pattern_confidence_boost']),
                        'price': current_price,
                        'volume': data['Volume'].iloc[i],
                        'holding_days': 5,
                        'ticker': self.ticker
                    })
                
                # Detect potential double bottom (bullish reversal)
                elif (current_price < sma_20[i] and current_rsi < 30 and
                      i > 10 and current_price < close_prices[i-5:i].min() * 1.02):
                    patterns.append({
                        'timestamp': data.index[i],
                        'pattern': 'Potential Double Bottom',
                        'type': 'chart',
                        'signal': 'bullish',
                        'strength': min(1.0, 0.6 * self.current_config['pattern_confidence_boost']),
                        'price': current_price,
                        'volume': data['Volume'].iloc[i],
                        'holding_days': 5,
                        'ticker': self.ticker
                    })
                
                # Detect ascending triangle (bullish continuation)
                elif (sma_20[i] > sma_50[i] and current_rsi > 50 and current_rsi < 70 and
                      current_price > sma_20[i]):
                    patterns.append({
                        'timestamp': data.index[i],
                        'pattern': 'Ascending Triangle Pattern',
                        'type': 'chart',
                        'signal': 'bullish',
                        'strength': min(1.0, 0.5 * self.current_config['pattern_confidence_boost']),
                        'price': current_price,
                        'volume': data['Volume'].iloc[i],
                        'holding_days': 3,
                        'ticker': self.ticker
                    })
        
        except Exception as e:
            logger.error(f"Error in TA-Lib fallback pattern detection: {e}")
        
        return patterns
    
    def calculate_volume_confirmation(self, data, pattern_idx):
        """Calculate volume confirmation for pattern using ticker-specific thresholds"""
        if pattern_idx < 20:  # Need history for volume analysis
            return 0.5
        
        current_volume = data['Volume'].iloc[pattern_idx]
        avg_volume = data['Volume'].iloc[pattern_idx-20:pattern_idx].mean()
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Apply ticker-specific volume multiplier
        volume_multiplier = self.current_config['volume_multiplier']
        
        # Strong volume confirmation (adjusted for ticker volatility)
        if volume_ratio > (1.5 * volume_multiplier):
            return 0.9
        elif volume_ratio > (1.2 * volume_multiplier):
            return 0.7
        elif volume_ratio > (0.8 * volume_multiplier):
            return 0.5
        else:
            return 0.3
    
    def scan_daily_patterns(self, ticker=None):
        """Scan for daily patterns"""
        # Use instance ticker if none provided
        scan_ticker = ticker or self.ticker
        logger.info(f"Scanning daily patterns for {scan_ticker}")
        
        # Get daily data (100 days for pattern detection)
        data = yf.download(scan_ticker, period='100d', interval='1d', progress=False)
        
        if data.empty:
            logger.error(f"No data retrieved for {scan_ticker}")
            return []
        
        # Clean data
        data = self.clean_data(data)
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} bars)")
        
        # Detect all patterns
        all_patterns = []
        
        # Candlestick patterns
        candlestick_patterns = self.detect_candlestick_patterns(data)
        all_patterns.extend(candlestick_patterns)
        logger.info(f"Found {len(candlestick_patterns)} candlestick patterns")
        
        # Chart patterns
        chart_patterns = self.detect_chart_patterns(data)
        all_patterns.extend(chart_patterns)
        logger.info(f"Found {len(chart_patterns)} chart patterns")
        
        # Add ticker and volume confirmation (ticker already added in pattern detection)
        for pattern in all_patterns:
            pattern['ticker'] = scan_ticker
            # Find pattern index
            pattern_idx = data.index.get_loc(pattern['timestamp'])
            pattern['volume_confirmation'] = self.calculate_volume_confirmation(data, pattern_idx)
            # Calculate entry price (next day's open)
            if pattern_idx < len(data) - 1:
                pattern['entry_price'] = data['Open'].iloc[pattern_idx + 1]
            else:
                # For today's pattern, estimate tomorrow's open
                pattern['entry_price'] = data['Close'].iloc[-1] * 1.001  # Small gap
        
        # Filter for recent patterns (last 5 days)
        recent_patterns = []
        cutoff_date = data.index[-1] - timedelta(days=5)
        
        for pattern in all_patterns:
            if pattern['timestamp'] >= cutoff_date:
                recent_patterns.append(pattern)
        
        logger.info(f"Found {len(recent_patterns)} recent patterns (last 5 days)")
        return recent_patterns
    
    def load_or_create_evaluation_file(self):
        """Load existing evaluation file or create new one"""
        blob_name = f"{self.azure_folder}/pattern_evaluations.json"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Try to download existing file
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            evaluations = json.loads(content)
            logger.info(f"Loaded existing evaluations with {len(evaluations)} entries")
            return evaluations
        except:
            logger.info("Creating new evaluation file")
            return []
    
    def save_evaluation(self, evaluation_data):
        """Append evaluation to the cumulative file"""
        # Load existing evaluations
        evaluations = self.load_or_create_evaluation_file()
        
        # Add new evaluation
        evaluations.append(evaluation_data)
        
        # Save back to Azure
        blob_name = f"{self.azure_folder}/pattern_evaluations.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(evaluations, indent=2, default=str)
        blob_client.upload_blob(json_data, overwrite=True)
        logger.info(f"Saved evaluation. Total evaluations: {len(evaluations)}")
    
    def save_next_day_predictions(self, patterns, ticker):
        """Save next day trading predictions"""
        if not patterns:
            logger.info(f"No patterns to save for {ticker}")
            predictions = {
                'ticker': ticker,
                'scan_date': datetime.now().isoformat(),
                'market_close_date': datetime.now().date().isoformat(),
                'predictions': [],
                'summary': {
                    'total_patterns': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'recommendation': 'HOLD',
                    'confidence': 0.0
                }
            }
        else:
            # Analyze patterns for overall signal
            bullish_count = sum(1 for p in patterns if p['signal'] == 'bullish')
            bearish_count = sum(1 for p in patterns if p['signal'] == 'bearish')
            neutral_count = sum(1 for p in patterns if p['signal'] == 'neutral')
            
            # Calculate weighted confidence
            bullish_strength = sum(p['strength'] * p['volume_confirmation'] 
                                  for p in patterns if p['signal'] == 'bullish')
            bearish_strength = sum(p['strength'] * p['volume_confirmation'] 
                                  for p in patterns if p['signal'] == 'bearish')
            
            # Determine recommendation
            if bullish_strength > bearish_strength * 1.5:
                recommendation = 'BUY'
                confidence = bullish_strength / (bullish_strength + bearish_strength) if bearish_strength > 0 else 0.9
            elif bearish_strength > bullish_strength * 1.5:
                recommendation = 'SELL'
                confidence = bearish_strength / (bullish_strength + bearish_strength) if bullish_strength > 0 else 0.9
            else:
                recommendation = 'HOLD'
                confidence = 0.5
            
            predictions = {
                'ticker': ticker,
                'scan_date': datetime.now().isoformat(),
                'market_close_date': datetime.now().date().isoformat(),
                'predictions': patterns,
                'summary': {
                    'total_patterns': len(patterns),
                    'bullish_count': bullish_count,
                    'bearish_count': bearish_count,
                    'neutral_count': neutral_count,
                    'bullish_strength': round(bullish_strength, 3),
                    'bearish_strength': round(bearish_strength, 3),
                    'recommendation': recommendation,
                    'confidence': round(confidence, 3),
                    'next_trading_day': (datetime.now() + timedelta(days=1)).date().isoformat()
                }
            }
        
        # Save to Azure
        blob_name = f"{self.azure_folder}/next_day_predictions.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(predictions, indent=2, default=str)
        blob_client.upload_blob(json_data, overwrite=True)
        logger.info(f"Saved next day predictions: {predictions['summary']['recommendation']} "
                   f"with {predictions['summary']['confidence']:.1%} confidence")
        
        return predictions
    
    def run_daily_scan(self, tickers=['NVDA']):
        """Run daily scan for specified tickers"""
        scan_time = datetime.now()
        
        all_patterns = []
        for ticker in tickers:
            try:
                patterns = self.scan_daily_patterns(ticker)
                all_patterns.extend(patterns)
                
                # Save predictions for this ticker
                predictions = self.save_next_day_predictions(patterns, ticker)
                
                # Save evaluation data
                evaluation = {
                    'scan_date': scan_time.isoformat(),
                    'ticker': ticker,
                    'patterns_found': len(patterns),
                    'recommendation': predictions['summary']['recommendation'],
                    'confidence': predictions['summary']['confidence'],
                    'patterns': patterns
                }
                self.save_evaluation(evaluation)
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        logger.info(f"Daily scan complete. Found {len(all_patterns)} total patterns")
        return all_patterns


def main():
    """Main function for GitHub Actions"""
    # Get ticker from environment or default
    ticker = os.getenv('TICKER', 'NVDA')
    
    # Create scanner for specific ticker
    scanner = DailyPatternScanner(ticker=ticker)
    
    # Run scan for single ticker
    patterns = scanner.run_daily_scan([ticker])
    
    logger.info(f"Daily pattern scan completed successfully for {ticker}")
    return 0


if __name__ == "__main__":
    exit(main())