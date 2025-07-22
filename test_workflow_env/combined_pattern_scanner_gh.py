#!/usr/bin/env python3
"""
Combined Pattern Scanner for GitHub Actions Deployment
Detects both candlestick patterns (TA-Lib) and chart patterns (tradingpattern)
Saves results to Azure blob storage
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
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
        detect_double_top_bottom
    )
    logger.info("tradingpattern library loaded successfully")
except ImportError as e:
    logger.error(f"CRITICAL: tradingpatterns is required but not available: {e}")
    raise ImportError("tradingpatterns is required. Please install it from: https://github.com/white07S/TradingPatternScanner")

# Load Azure credentials
load_dotenv('config/.env')

class CombinedPatternScanner:
    def __init__(self):
        # Azure configuration
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_key = os.getenv('AZURE_STORAGE_KEY')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        if not all([self.storage_account, self.storage_key, self.container_name]):
            raise ValueError("Azure storage credentials not found")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.storage_key
        )
    
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
        
        # Key candlestick patterns
        pattern_functions = {
            'CDLDOJI': ('Doji', 'neutral'),
            'CDLHAMMER': ('Hammer', 'bullish'),
            'CDLINVERTEDHAMMER': ('Inverted Hammer', 'bullish'),
            'CDLSHOOTINGSTAR': ('Shooting Star', 'bearish'),
            'CDLENGULFING': ('Engulfing Pattern', 'reversal'),
            'CDLHARAMI': ('Harami', 'reversal'),
            'CDLMORNINGSTAR': ('Morning Star', 'bullish'),
            'CDLEVENINGSTAR': ('Evening Star', 'bearish'),
            'CDLSPINNINGTOP': ('Spinning Top', 'neutral'),
            'CDLMARUBOZU': ('Marubozu', 'continuation'),
            'CDLDRAGONFLYDOJI': ('Dragonfly Doji', 'bullish'),
            'CDLGRAVESTONEDOJI': ('Gravestone Doji', 'bearish'),
            'CDL3WHITESOLDIERS': ('Three White Soldiers', 'bullish'),
            'CDL3BLACKCROWS': ('Three Black Crows', 'bearish')
        }
        
        # Detect patterns
        for func_name, (pattern_name, bias) in pattern_functions.items():
            try:
                pattern_func = getattr(talib, func_name)
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                
                # Find where pattern occurs
                pattern_indices = np.where(result != 0)[0]
                
                for idx in pattern_indices:
                    signal = int(result[idx])
                    
                    # Determine direction
                    if bias == 'reversal' or bias == 'continuation':
                        direction = 'bullish' if signal > 0 else 'bearish'
                    elif bias == 'neutral':
                        direction = 'neutral'
                    else:
                        direction = bias
                    
                    patterns.append({
                        'timestamp': data.index[idx],
                        'pattern_type': 'candlestick',
                        'pattern_name': pattern_name,
                        'direction': direction,
                        'confidence': abs(signal),  # TA-Lib returns 100 or -100
                        'price': float(data['Close'].iloc[idx]),
                        'high': float(data['High'].iloc[idx]),
                        'low': float(data['Low'].iloc[idx])
                    })
                    
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")
                continue
        
        return patterns
    
    def detect_chart_patterns(self, data):
        """Detect chart patterns using tradingpattern library"""
        
        patterns = []
        
        try:
            # Head and Shoulders
            df_hs = detect_head_shoulder(data.copy(), window=5)
            hs_patterns = df_hs[df_hs['head_shoulder_pattern'].notna()]
            
            for idx, row in hs_patterns.iterrows():
                pattern_type = row['head_shoulder_pattern']
                
                patterns.append({
                    'timestamp': idx,
                    'pattern_type': 'chart',
                    'pattern_name': pattern_type,
                    'direction': 'bullish' if 'Inverse' in pattern_type else 'bearish',
                    'confidence': 84.5,  # Based on library documentation
                    'price': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low'])
                })
            
            # Double Tops/Bottoms
            df_dt = detect_double_top_bottom(data.copy(), window=5, threshold=0.02)
            dt_patterns = df_dt[df_dt['double_pattern'].notna()]
            
            for idx, row in dt_patterns.iterrows():
                pattern_type = row['double_pattern']
                
                patterns.append({
                    'timestamp': idx,
                    'pattern_type': 'chart',
                    'pattern_name': pattern_type,
                    'direction': 'bearish' if 'Top' in pattern_type else 'bullish',
                    'confidence': 84.5,  # Based on library documentation
                    'price': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low'])
                })
                
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
        
        return patterns
    
    def scan_patterns(self, ticker, interval='5m', period='1d'):
        """Main scanning function"""
        logger.info(f"Scanning patterns for {ticker}")
        
        # Download data
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data retrieved for {ticker}")
            return []
        
        # Clean data
        data = self.clean_data(data)
        
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
        
        # Add ticker to all patterns
        for pattern in all_patterns:
            pattern['ticker'] = ticker
        
        return all_patterns
    
    def save_patterns_to_azure(self, patterns, ticker):
        """Save patterns to Azure blob storage"""
        if not patterns:
            logger.info(f"No patterns to save for {ticker}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(patterns)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"python_package_evaluation/combined_scanner/{ticker}_patterns_{timestamp}.csv"
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        
        # Upload to Azure
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            blob_client.upload_blob(csv_data, overwrite=True)
            logger.info(f"Saved {len(patterns)} patterns to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving to Azure: {e}")
            return None
    
    def scan_multiple_tickers(self, tickers):
        """Scan multiple tickers and save results"""
        all_results = []
        
        for ticker in tickers:
            try:
                patterns = self.scan_patterns(ticker)
                if patterns:
                    self.save_patterns_to_azure(patterns, ticker)
                    all_results.extend(patterns)
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Save combined results
        if all_results:
            self.save_patterns_to_azure(all_results, 'ALL')
        
        return all_results

def main():
    """Main function for GitHub Actions"""
    scanner = CombinedPatternScanner()
    
    # Define tickers to scan
    tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
    
    # Scan patterns
    logger.info("Starting combined pattern scan")
    results = scanner.scan_multiple_tickers(tickers)
    
    # Summary
    logger.info(f"\nScan completed:")
    logger.info(f"Total patterns found: {len(results)}")
    
    if results:
        # Count by type
        candlestick_count = sum(1 for p in results if p['pattern_type'] == 'candlestick')
        chart_count = sum(1 for p in results if p['pattern_type'] == 'chart')
        
        logger.info(f"  Candlestick patterns: {candlestick_count}")
        logger.info(f"  Chart patterns: {chart_count}")
        
        # Count by direction
        bullish_count = sum(1 for p in results if p['direction'] == 'bullish')
        bearish_count = sum(1 for p in results if p['direction'] == 'bearish')
        neutral_count = sum(1 for p in results if p['direction'] == 'neutral')
        
        logger.info(f"  Bullish: {bullish_count}")
        logger.info(f"  Bearish: {bearish_count}")
        logger.info(f"  Neutral: {neutral_count}")

if __name__ == "__main__":
    main()