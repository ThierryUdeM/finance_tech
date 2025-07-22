#!/usr/bin/env python3
"""
Pattern Evaluator - End of Day Analysis
Evaluates the success rate of patterns detected during the day
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Azure credentials
load_dotenv('config/.env')

class PatternEvaluator:
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
    
    def load_patterns_from_azure(self, date=None):
        """Load pattern files from Azure for a specific date"""
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        patterns = []
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # List all pattern files for the date
        prefix = f"python_package_evaluation/combined_scanner/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blobs:
            # Check if it's from today and not an evaluation file
            if date in blob.name and '_patterns_' in blob.name and '_evaluation_' not in blob.name:
                try:
                    # Download blob
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container_name,
                        blob=blob.name
                    )
                    blob_data = blob_client.download_blob().readall()
                    
                    # Parse CSV
                    import io
                    df = pd.read_csv(io.StringIO(blob_data.decode('utf-8')))
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    patterns.append(df)
                    logger.info(f"Loaded {len(df)} patterns from {blob.name}")
                except Exception as e:
                    logger.error(f"Error loading {blob.name}: {e}")
                    continue
        
        if patterns:
            return pd.concat(patterns, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def evaluate_pattern(self, pattern, price_data):
        """Evaluate a single pattern's performance"""
        pattern_time = pattern['timestamp']
        
        # Get price data after pattern
        future_prices = price_data[price_data.index > pattern_time]
        
        if future_prices.empty:
            return {
                'success': None,
                'price_change': 0,
                'max_favorable': 0,
                'max_adverse': 0,
                'duration': 0
            }
        
        # Calculate metrics for different time horizons
        horizons = {
            '5min': 1,     # 1 candle
            '15min': 3,    # 3 candles
            '30min': 6,    # 6 candles
            '1hour': 12,   # 12 candles
            '2hour': 24,   # 24 candles
            'eod': len(future_prices)  # End of day
        }
        
        results = {}
        entry_price = pattern['price']
        
        for horizon_name, num_candles in horizons.items():
            if num_candles > len(future_prices):
                num_candles = len(future_prices)
            
            horizon_data = future_prices.iloc[:num_candles]
            
            if horizon_data.empty:
                continue
            
            # Calculate price change
            final_price = horizon_data['Close'].iloc[-1]
            price_change_pct = ((final_price - entry_price) / entry_price) * 100
            
            # Maximum favorable and adverse excursions
            if pattern['direction'] == 'bullish':
                max_favorable = ((horizon_data['High'].max() - entry_price) / entry_price) * 100
                max_adverse = ((entry_price - horizon_data['Low'].min()) / entry_price) * 100
                success = price_change_pct > 0.1  # 0.1% threshold for success
            elif pattern['direction'] == 'bearish':
                max_favorable = ((entry_price - horizon_data['Low'].min()) / entry_price) * 100
                max_adverse = ((horizon_data['High'].max() - entry_price) / entry_price) * 100
                success = price_change_pct < -0.1  # -0.1% threshold for success
            else:  # neutral
                max_favorable = abs(price_change_pct)
                max_adverse = 0
                success = None
            
            results[horizon_name] = {
                'success': success,
                'price_change': round(price_change_pct, 3),
                'max_favorable': round(max_favorable, 3),
                'max_adverse': round(max_adverse, 3),
                'final_price': round(final_price, 2)
            }
        
        return results
    
    def evaluate_all_patterns(self, patterns_df):
        """Evaluate all patterns"""
        if patterns_df.empty:
            logger.warning("No patterns to evaluate")
            return pd.DataFrame()
        
        # Group by ticker
        evaluation_results = []
        
        for ticker in patterns_df['ticker'].unique():
            ticker_patterns = patterns_df[patterns_df['ticker'] == ticker].copy()
            
            # Get price data for evaluation
            try:
                # Get 2 days of data to ensure we have enough future data
                price_data = yf.download(ticker, period='2d', interval='5m', progress=False)
                
                if price_data.empty:
                    logger.error(f"No price data for {ticker}")
                    continue
                
                # Clean price data
                if isinstance(price_data.columns, pd.MultiIndex):
                    price_data.columns = price_data.columns.get_level_values(0)
                
                # Evaluate each pattern
                for idx, pattern in ticker_patterns.iterrows():
                    eval_result = self.evaluate_pattern(pattern, price_data)
                    
                    # Create evaluation record
                    for horizon, metrics in eval_result.items():
                        evaluation_results.append({
                            'timestamp': pattern['timestamp'],
                            'ticker': pattern['ticker'],
                            'pattern_type': pattern['pattern_type'],
                            'pattern_name': pattern['pattern_name'],
                            'direction': pattern['direction'],
                            'confidence': pattern['confidence'],
                            'entry_price': pattern['price'],
                            'horizon': horizon,
                            'success': metrics.get('success'),
                            'price_change': metrics.get('price_change', 0),
                            'max_favorable': metrics.get('max_favorable', 0),
                            'max_adverse': metrics.get('max_adverse', 0),
                            'final_price': metrics.get('final_price', pattern['price'])
                        })
                        
            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}")
                continue
        
        return pd.DataFrame(evaluation_results)
    
    def calculate_statistics(self, evaluation_df):
        """Calculate win rates and statistics"""
        if evaluation_df.empty:
            return {}
        
        stats = {}
        
        # Overall statistics
        for horizon in evaluation_df['horizon'].unique():
            horizon_data = evaluation_df[evaluation_df['horizon'] == horizon]
            
            # Filter out neutral patterns for success rate
            directional_data = horizon_data[horizon_data['success'].notna()]
            
            if len(directional_data) > 0:
                win_rate = (directional_data['success'].sum() / len(directional_data)) * 100
            else:
                win_rate = 0
            
            stats[f'{horizon}_win_rate'] = round(win_rate, 2)
            stats[f'{horizon}_total_patterns'] = len(horizon_data)
            stats[f'{horizon}_avg_price_change'] = round(horizon_data['price_change'].mean(), 3)
            stats[f'{horizon}_avg_favorable'] = round(horizon_data['max_favorable'].mean(), 3)
            stats[f'{horizon}_avg_adverse'] = round(horizon_data['max_adverse'].mean(), 3)
        
        # Statistics by pattern type
        pattern_stats = []
        
        for pattern_type in evaluation_df['pattern_type'].unique():
            for pattern_name in evaluation_df[evaluation_df['pattern_type'] == pattern_type]['pattern_name'].unique():
                pattern_data = evaluation_df[
                    (evaluation_df['pattern_type'] == pattern_type) & 
                    (evaluation_df['pattern_name'] == pattern_name) &
                    (evaluation_df['horizon'] == '1hour')  # Use 1-hour horizon for pattern stats
                ]
                
                directional_data = pattern_data[pattern_data['success'].notna()]
                
                if len(directional_data) > 0:
                    win_rate = (directional_data['success'].sum() / len(directional_data)) * 100
                else:
                    win_rate = 0
                
                pattern_stats.append({
                    'pattern_type': pattern_type,
                    'pattern_name': pattern_name,
                    'count': len(pattern_data),
                    'win_rate': round(win_rate, 2),
                    'avg_price_change': round(pattern_data['price_change'].mean(), 3),
                    'avg_favorable': round(pattern_data['max_favorable'].mean(), 3),
                    'avg_adverse': round(pattern_data['max_adverse'].mean(), 3)
                })
        
        stats['pattern_performance'] = pd.DataFrame(pattern_stats)
        
        return stats
    
    def save_evaluation_to_azure(self, evaluation_df, stats):
        """Save evaluation results to Azure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed evaluation
        if not evaluation_df.empty:
            filename = f"python_package_evaluation/combined_scanner/evaluation_{timestamp}_detailed.csv"
            csv_data = evaluation_df.to_csv(index=False)
            
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=filename
                )
                blob_client.upload_blob(csv_data, overwrite=True)
                logger.info(f"Saved detailed evaluation to {filename}")
            except Exception as e:
                logger.error(f"Error saving detailed evaluation: {e}")
        
        # Save summary statistics
        if 'pattern_performance' in stats:
            filename = f"python_package_evaluation/combined_scanner/evaluation_{timestamp}_summary.csv"
            csv_data = stats['pattern_performance'].to_csv(index=False)
            
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=filename
                )
                blob_client.upload_blob(csv_data, overwrite=True)
                logger.info(f"Saved summary evaluation to {filename}")
            except Exception as e:
                logger.error(f"Error saving summary: {e}")
        
        return stats
    
    def run_evaluation(self):
        """Main evaluation function"""
        logger.info("Starting pattern evaluation")
        
        # Load today's patterns
        patterns_df = self.load_patterns_from_azure()
        
        if patterns_df.empty:
            logger.warning("No patterns found for evaluation")
            return None
        
        logger.info(f"Evaluating {len(patterns_df)} patterns")
        
        # Evaluate patterns
        evaluation_df = self.evaluate_all_patterns(patterns_df)
        
        if evaluation_df.empty:
            logger.warning("No evaluation results")
            return None
        
        # Calculate statistics
        stats = self.calculate_statistics(evaluation_df)
        
        # Save results
        self.save_evaluation_to_azure(evaluation_df, stats)
        
        # Print summary
        logger.info("\n=== EVALUATION SUMMARY ===")
        logger.info(f"Total patterns evaluated: {len(patterns_df)}")
        
        for horizon in ['5min', '15min', '30min', '1hour', '2hour', 'eod']:
            if f'{horizon}_win_rate' in stats:
                logger.info(f"\n{horizon.upper()} Performance:")
                logger.info(f"  Win Rate: {stats[f'{horizon}_win_rate']}%")
                logger.info(f"  Avg Price Change: {stats[f'{horizon}_avg_price_change']}%")
                logger.info(f"  Avg Favorable Move: {stats[f'{horizon}_avg_favorable']}%")
                logger.info(f"  Avg Adverse Move: {stats[f'{horizon}_avg_adverse']}%")
        
        if 'pattern_performance' in stats:
            logger.info("\n=== PATTERN PERFORMANCE (1-hour horizon) ===")
            print(stats['pattern_performance'].to_string(index=False))
        
        return stats

def main():
    """Main function for GitHub Actions"""
    evaluator = PatternEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()