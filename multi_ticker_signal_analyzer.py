#!/usr/bin/env python3
"""
Multi-Ticker Signal Analysis using KNN Pattern Matching
Extends the proven NVDA Signal model to support multiple tickers
Based on the +4.33% return, 4.14 Sharpe ratio Signal model architecture
"""

import pandas as pd
import numpy as np
import os
import json
import yaml
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Load Azure credentials
load_dotenv('config/.env')

class MultiTickerSignalAnalyzer:
    def __init__(self):
        """Initialize the multi-ticker signal analyzer"""
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
        
        # Supported tickers with their data requirements
        self.tickers = {
            'NVDA': {
                'name': 'NVIDIA Corporation',
                'data_source': 'local',  # Has 2.5 years of Databento data
                'local_file': '/home/thierrygc/test_1/github/signal/directional_analysis/NVDA_15min_pattern_ready.csv',
                'volatility_multiplier': 1.0,
                'min_pattern_threshold': 0.5
            },
            'AAPL': {
                'name': 'Apple Inc.',
                'data_source': 'yfinance',  # Will need Databento data for full performance
                'local_file': None,
                'volatility_multiplier': 0.8,  # Lower volatility stock
                'min_pattern_threshold': 0.4
            },
            'MSFT': {
                'name': 'Microsoft Corporation', 
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 0.8,
                'min_pattern_threshold': 0.4
            },
            'GOOGL': {
                'name': 'Alphabet Inc.',
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 0.9,
                'min_pattern_threshold': 0.45
            },
            'AMZN': {
                'name': 'Amazon.com Inc.',
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 1.1,
                'min_pattern_threshold': 0.55
            },
            'TSLA': {
                'name': 'Tesla Inc.',
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 1.5,  # High volatility stock
                'min_pattern_threshold': 0.7
            },
            'BTC-USD': {
                'name': 'Bitcoin',
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 2.0,  # Crypto volatility
                'min_pattern_threshold': 0.8
            },
            'AC.TO': {
                'name': 'Air Canada',
                'data_source': 'yfinance',
                'local_file': None,
                'volatility_multiplier': 1.2,
                'min_pattern_threshold': 0.6
            }
        }
        
        # Load configuration
        self.config = self.load_config()
        
        # Pattern matching parameters (from proven NVDA model)
        self.query_length = 20  # 20-bar patterns
        self.k_neighbors = 10   # Top 10 nearest neighbors
        self.time_decay_factor = 0.1  # Emphasize recent bars
        
    def load_config(self):
        """Load configuration with defaults optimized for multi-ticker"""
        return {
            'timeframe_weights': {
                'short_term': 0.4,   # 1-hour predictions
                'medium_term': 0.35, # 3-hour predictions
                'long_term': 0.25    # End-of-day predictions
            },
            'adaptive_thresholds': {
                'atr_multiplier': 2.0,
                'min_threshold': 0.05,  # 0.05% minimum
                'max_threshold': 0.50   # 0.50% maximum
            },
            'pattern_matching': {
                'confidence_levels': {
                    'high': {'min_library_size': 100, 'max_distance': 0.8},
                    'medium': {'min_library_size': 50, 'max_distance': 1.2},
                    'low': {'min_library_size': 20, 'max_distance': 2.0}
                }
            }
        }
    
    def load_ticker_data(self, ticker, days_back=60):
        """Load data for a specific ticker"""
        print(f"Loading data for {ticker}...")
        
        ticker_config = self.tickers.get(ticker)
        if not ticker_config:
            raise ValueError(f"Ticker {ticker} not supported")
        
        # Try to load local Databento data first (best quality)
        if ticker_config['data_source'] == 'local' and ticker_config['local_file']:
            if os.path.exists(ticker_config['local_file']):
                print(f"Loading high-quality Databento data for {ticker}")
                df = pd.read_csv(ticker_config['local_file'])
                df['ts_event'] = pd.to_datetime(df['ts_event'])
                df = df.set_index('ts_event')
                df = df.sort_index()
                print(f"Loaded {len(df)} bars of Databento data for {ticker}")
                return df
            else:
                print(f"Warning: Local data file not found for {ticker}, falling back to yfinance")
        
        # Fall back to yfinance (limited but available)
        print(f"Loading yfinance data for {ticker} (last {days_back} days)")
        try:
            data = yf.download(ticker, period=f"{days_back}d", interval="15m", progress=False)
            
            if data.empty:
                raise ValueError(f"No data retrieved for {ticker}")
            
            # Clean and standardize the data
            data = self.clean_data(data)
            print(f"Loaded {len(data)} bars of yfinance data for {ticker}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None
    
    def clean_data(self, data):
        """Clean and standardize data format"""
        df = data.copy()
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names
        column_mapping = {
            'Adj Close': 'Close',
            'adj close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low', 
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    def build_pattern_library(self, data, ticker):
        """Build pattern library from historical data"""
        print(f"Building pattern library for {ticker}...")
        
        if len(data) < self.query_length + 10:
            print(f"Warning: Insufficient data for {ticker} ({len(data)} bars)")
            return None
        
        patterns = []
        returns_1h = []
        returns_3h = []
        returns_eod = []
        
        # Extract patterns every bar (dense sampling for better library)
        for i in range(len(data) - self.query_length - 20):  # Leave room for future returns
            # Get 20-bar pattern
            pattern_data = data.iloc[i:i + self.query_length]
            pattern_prices = pattern_data['Close'].values
            
            # Normalize pattern using z-score
            pattern_normalized = (pattern_prices - np.mean(pattern_prices)) / (np.std(pattern_prices) + 1e-8)
            
            # Calculate future returns
            current_price = data.iloc[i + self.query_length]['Close']
            
            # 1-hour return (4 bars ahead for 15-min data)
            if i + self.query_length + 4 < len(data):
                future_price_1h = data.iloc[i + self.query_length + 4]['Close']
                ret_1h = (future_price_1h - current_price) / current_price
                returns_1h.append(ret_1h)
            else:
                returns_1h.append(np.nan)
            
            # 3-hour return (12 bars ahead)
            if i + self.query_length + 12 < len(data):
                future_price_3h = data.iloc[i + self.query_length + 12]['Close']
                ret_3h = (future_price_3h - current_price) / current_price
                returns_3h.append(ret_3h)
            else:
                returns_3h.append(np.nan)
            
            # End-of-day return (next trading day)
            if i + self.query_length + 26 < len(data):  # ~6.5 hours ahead
                future_price_eod = data.iloc[i + self.query_length + 26]['Close']
                ret_eod = (future_price_eod - current_price) / current_price
                returns_eod.append(ret_eod)
            else:
                returns_eod.append(np.nan)
            
            patterns.append(pattern_normalized)
        
        if len(patterns) == 0:
            print(f"No valid patterns found for {ticker}")
            return None
        
        library = {
            'patterns': np.array(patterns),
            'returns_1h': np.array(returns_1h),
            'returns_3h': np.array(returns_3h),
            'returns_eod': np.array(returns_eod),
            'ticker': ticker,
            'library_size': len(patterns)
        }
        
        print(f"Built pattern library for {ticker}: {len(patterns)} patterns")
        return library
    
    def calculate_pattern_distances(self, query_pattern, library_patterns):
        """Calculate weighted distances between query pattern and library patterns"""
        # Time-decay weights (emphasize recent bars)
        weights = np.exp(-self.time_decay_factor * np.arange(self.query_length))
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate weighted Euclidean distances
        distances = []
        for lib_pattern in library_patterns:
            # Weighted squared differences
            weighted_diff = weights * (query_pattern - lib_pattern) ** 2
            distance = np.sqrt(np.sum(weighted_diff))
            distances.append(distance)
        
        return np.array(distances)
    
    def make_prediction(self, query_pattern, pattern_library, ticker):
        """Make prediction using K-NN pattern matching"""
        if pattern_library is None:
            return None
        
        # Calculate distances
        distances = self.calculate_pattern_distances(query_pattern, pattern_library['patterns'])
        
        # Find K nearest neighbors
        neighbor_indices = np.argsort(distances)[:self.k_neighbors]
        neighbor_distances = distances[neighbor_indices]
        
        # Calculate confidence based on library size and distance spread
        ticker_config = self.tickers[ticker]
        min_threshold = ticker_config['min_pattern_threshold']
        
        confidence_score = min(1.0, pattern_library['library_size'] / 100.0)
        distance_confidence = 1.0 / (1.0 + np.mean(neighbor_distances))
        overall_confidence = confidence_score * distance_confidence
        
        if overall_confidence < min_threshold:
            return {
                'ticker': ticker,
                'pred_1h_dir': 'HOLD',
                'pred_3h_dir': 'HOLD', 
                'pred_eod_dir': 'HOLD',
                'confidence': overall_confidence,
                'reason': 'Low confidence prediction'
            }
        
        # Weight predictions by inverse distance
        weights = 1.0 / (neighbor_distances + 1e-8)
        weights = weights / np.sum(weights)
        
        # Calculate weighted predictions for each timeframe
        predictions = {}
        for timeframe in ['1h', '3h', 'eod']:
            returns_key = f'returns_{timeframe}'
            
            if returns_key in pattern_library:
                neighbor_returns = pattern_library[returns_key][neighbor_indices]
                # Filter out NaN values
                valid_mask = ~np.isnan(neighbor_returns)
                
                if np.sum(valid_mask) > 0:
                    valid_returns = neighbor_returns[valid_mask]
                    valid_weights = weights[valid_mask]
                    valid_weights = valid_weights / np.sum(valid_weights)
                    
                    pred_return = np.sum(valid_returns * valid_weights)
                    predictions[f'pred_{timeframe}_ret'] = pred_return
                    
                    # Convert to direction with adaptive thresholds
                    atr_threshold = self.calculate_adaptive_threshold(ticker)
                    
                    if pred_return > atr_threshold:
                        predictions[f'pred_{timeframe}_dir'] = 'BUY'
                    elif pred_return < -atr_threshold:
                        predictions[f'pred_{timeframe}_dir'] = 'SELL'
                    else:
                        predictions[f'pred_{timeframe}_dir'] = 'HOLD'
                else:
                    predictions[f'pred_{timeframe}_ret'] = 0.0
                    predictions[f'pred_{timeframe}_dir'] = 'HOLD'
        
        # Add metadata
        predictions.update({
            'ticker': ticker,
            'confidence': overall_confidence,
            'library_size': pattern_library['library_size'],
            'avg_neighbor_distance': np.mean(neighbor_distances),
            'timestamp': datetime.now().isoformat()
        })
        
        return predictions
    
    def calculate_adaptive_threshold(self, ticker):
        """Calculate adaptive threshold based on ticker volatility"""
        ticker_config = self.tickers[ticker]
        base_threshold = self.config['adaptive_thresholds']['min_threshold']
        volatility_mult = ticker_config['volatility_multiplier']
        
        threshold = base_threshold * volatility_mult
        
        # Clamp to min/max bounds
        min_thresh = self.config['adaptive_thresholds']['min_threshold']
        max_thresh = self.config['adaptive_thresholds']['max_threshold']
        
        return max(min_thresh, min(max_thresh, threshold))
    
    def analyze_ticker(self, ticker):
        """Analyze a single ticker and generate signal"""
        try:
            # Load data
            data = self.load_ticker_data(ticker)
            if data is None or len(data) < self.query_length + 20:
                print(f"Insufficient data for {ticker}")
                return None
            
            # Build pattern library
            pattern_library = self.build_pattern_library(data, ticker)
            if pattern_library is None:
                print(f"Failed to build pattern library for {ticker}")
                return None
            
            # Get current pattern (last 20 bars)
            current_pattern = data['Close'].tail(self.query_length).values
            current_pattern_normalized = (current_pattern - np.mean(current_pattern)) / (np.std(current_pattern) + 1e-8)
            
            # Make prediction
            prediction = self.make_prediction(current_pattern_normalized, pattern_library, ticker)
            
            if prediction:
                print(f"Generated signal for {ticker}: {prediction['pred_1h_dir']} (confidence: {prediction['confidence']:.3f})")
            
            return prediction
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            return None
    
    def run_multi_ticker_analysis(self, specific_tickers=None):
        """Run signal analysis for multiple tickers"""
        if specific_tickers:
            tickers_to_analyze = [t for t in specific_tickers if t in self.tickers]
        else:
            tickers_to_analyze = list(self.tickers.keys())
        
        print(f"Running multi-ticker signal analysis for: {', '.join(tickers_to_analyze)}")
        
        results = {}
        for ticker in tickers_to_analyze:
            print(f"\n--- Analyzing {ticker} ---")
            result = self.analyze_ticker(ticker)
            if result:
                results[ticker] = result
        
        # Save results to Azure
        if results:
            self.save_results(results)
        
        print(f"\nMulti-ticker analysis complete. Generated signals for {len(results)} tickers.")
        return results
    
    def save_results(self, results):
        """Save results to Azure Storage"""
        try:
            # Create results summary
            timestamp = datetime.now()
            results_summary = {
                'timestamp': timestamp.isoformat(),
                'scan_type': 'multi_ticker_signal_analysis',
                'total_tickers': len(results),
                'results': results
            }
            
            # Save to Azure
            blob_name = f"signal_analysis/multi_ticker_predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            json_data = json.dumps(results_summary, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)
            
            print(f"Results saved to Azure: {blob_name}")
            
        except Exception as e:
            print(f"Error saving results to Azure: {e}")


def main():
    """Main function for testing/standalone execution"""
    analyzer = MultiTickerSignalAnalyzer()
    
    # Run analysis for all supported tickers
    results = analyzer.run_multi_ticker_analysis()
    
    # Print summary
    if results:
        print("\n=== MULTI-TICKER SIGNAL SUMMARY ===")
        for ticker, result in results.items():
            print(f"{ticker}: {result['pred_1h_dir']} (confidence: {result['confidence']:.3f})")
    else:
        print("No signals generated")
    
    return 0


if __name__ == "__main__":
    exit(main())