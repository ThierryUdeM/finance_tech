#!/usr/bin/env python3
"""
Pattern Signal Tracker - Saves trading signals to Azure and evaluates performance
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import logging

# Add the script directory to Python path
sys.path.append('/home/thierrygc/script/')
import pattern_scanner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternSignalTracker:
    def __init__(self):
        # Load environment variables
        load_dotenv('config/.env')
        
        # Azure configuration
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_key = os.getenv('AZURE_STORAGE_KEY')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        if not all([self.storage_account, self.storage_key, self.container_name]):
            raise ValueError("Azure storage credentials not found in environment variables")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.storage_key
        )
        
    def scan_patterns(self, ticker, interval='5m', period='1d'):
        """Scan for patterns and generate signals"""
        logger.info(f"Scanning patterns for {ticker}")
        
        # Download data
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data retrieved for {ticker}")
            return None
        
        # Clean data
        cleaned_data = pattern_scanner.clean_yf_data(data)
        
        # Define patterns to check
        patterns = {
            "Head and Shoulders": pattern_scanner.find_head_and_shoulders,
            "Inverse Head and Shoulders": pattern_scanner.find_inverse_head_and_shoulders,
            "Double Top": pattern_scanner.find_double_top,
            "Double Bottom": pattern_scanner.find_double_bottom
        }
        
        signals = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for pattern_name, find_func in patterns.items():
            result = find_func(cleaned_data)
            
            if result is not None:
                # Calculate trading signals
                signal_data = pattern_scanner.calculate_trading_signals(cleaned_data, pattern_name, result)
                
                signal = {
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'pattern': pattern_name,
                    'action': signal_data['action'],
                    'current_price': signal_data['current_price'],
                    'entry_price': signal_data['entry'],
                    'stop_loss': signal_data['stop_loss'],
                    'target_price': signal_data['target'],
                    'risk_reward': signal_data['risk_reward'],
                    'distance_to_entry': signal_data['distance_to_entry'],
                    'status': 'pending',  # pending, triggered, hit_target, hit_stop, expired
                    'pattern_data': result  # Store pattern points for reference
                }
                
                signals.append(signal)
                logger.info(f"Found {pattern_name} pattern for {ticker}")
        
        return signals
    
    def save_signals_to_azure(self, signals, ticker):
        """Save signals to Azure blob storage"""
        if not signals:
            logger.info(f"No signals to save for {ticker}")
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals/{ticker}/{ticker}_signals_{timestamp}.json"
        
        # Convert to JSON
        json_data = json.dumps(signals, indent=2)
        
        # Upload to Azure
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            blob_client.upload_blob(json_data, overwrite=True)
            logger.info(f"Saved {len(signals)} signals to {filename}")
        except Exception as e:
            logger.error(f"Error saving to Azure: {e}")
    
    def evaluate_signals(self, ticker, hours_back=24):
        """Evaluate past signals to calculate win rate"""
        logger.info(f"Evaluating signals for {ticker} from last {hours_back} hours")
        
        # List all signal files for the ticker
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blobs = container_client.list_blobs(name_starts_with=f"signals/{ticker}/")
        
        all_signals = []
        for blob in blobs:
            # Download and parse signal files
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob.name
            )
            content = blob_client.download_blob().readall()
            signals = json.loads(content)
            all_signals.extend(signals)
        
        if not all_signals:
            logger.info("No signals found to evaluate")
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_signals)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter signals from the specified time period
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        df = df[df['timestamp'] >= cutoff_time]
        
        if df.empty:
            logger.info(f"No signals found in the last {hours_back} hours")
            return None
        
        # Get current price data
        current_data = yf.download(ticker, period='5d', interval='5m', progress=False)
        
        # Evaluate each signal
        results = []
        for idx, signal in df.iterrows():
            result = self._evaluate_single_signal(signal, current_data)
            results.append(result)
        
        # Calculate statistics
        eval_df = pd.DataFrame(results)
        
        # Win rate calculation
        completed_signals = eval_df[eval_df['status'].isin(['hit_target', 'hit_stop'])]
        if len(completed_signals) > 0:
            win_rate = (len(completed_signals[completed_signals['status'] == 'hit_target']) / 
                       len(completed_signals)) * 100
        else:
            win_rate = 0
        
        stats = {
            'ticker': ticker,
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_signals': len(eval_df),
            'pending_signals': len(eval_df[eval_df['status'] == 'pending']),
            'triggered_signals': len(eval_df[eval_df['status'] == 'triggered']),
            'hit_target': len(eval_df[eval_df['status'] == 'hit_target']),
            'hit_stop': len(eval_df[eval_df['status'] == 'hit_stop']),
            'expired': len(eval_df[eval_df['status'] == 'expired']),
            'win_rate': round(win_rate, 2),
            'signals_evaluated': results
        }
        
        # Save evaluation results
        self._save_evaluation_results(stats, ticker)
        
        return stats
    
    def _evaluate_single_signal(self, signal, price_data):
        """Evaluate a single signal against price data"""
        signal_time = pd.to_datetime(signal['timestamp'])
        
        # Get price data after signal time
        future_prices = price_data[price_data.index > signal_time]
        
        if future_prices.empty:
            return {**signal, 'status': 'pending', 'outcome': 'no_data'}
        
        # Check if signal was triggered
        if signal['action'] == 'BUY':
            # For buy signals, check if price went above entry
            triggered_mask = future_prices['High'] >= signal['entry_price']
            if triggered_mask.any():
                trigger_time = future_prices[triggered_mask].index[0]
                
                # Check outcome after trigger
                post_trigger = future_prices[future_prices.index >= trigger_time]
                
                # Check if hit target
                if (post_trigger['High'] >= signal['target_price']).any():
                    return {**signal, 'status': 'hit_target', 'outcome': 'win'}
                
                # Check if hit stop loss
                if (post_trigger['Low'] <= signal['stop_loss']).any():
                    return {**signal, 'status': 'hit_stop', 'outcome': 'loss'}
                
                return {**signal, 'status': 'triggered', 'outcome': 'open'}
        
        else:  # SELL signal
            # For sell signals, check if price went below entry
            triggered_mask = future_prices['Low'] <= signal['entry_price']
            if triggered_mask.any():
                trigger_time = future_prices[triggered_mask].index[0]
                
                # Check outcome after trigger
                post_trigger = future_prices[future_prices.index >= trigger_time]
                
                # Check if hit target
                if (post_trigger['Low'] <= signal['target_price']).any():
                    return {**signal, 'status': 'hit_target', 'outcome': 'win'}
                
                # Check if hit stop loss
                if (post_trigger['High'] >= signal['stop_loss']).any():
                    return {**signal, 'status': 'hit_stop', 'outcome': 'loss'}
                
                return {**signal, 'status': 'triggered', 'outcome': 'open'}
        
        # Signal not triggered yet
        # Check if expired (more than 24 hours old)
        if (datetime.now() - signal_time).total_seconds() > 86400:
            return {**signal, 'status': 'expired', 'outcome': 'expired'}
        
        return {**signal, 'status': 'pending', 'outcome': 'waiting'}
    
    def _save_evaluation_results(self, stats, ticker):
        """Save evaluation results to Azure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluations/{ticker}/{ticker}_evaluation_{timestamp}.json"
        
        json_data = json.dumps(stats, indent=2)
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=filename
            )
            blob_client.upload_blob(json_data, overwrite=True)
            logger.info(f"Saved evaluation results to {filename}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

def main():
    """Main function for GitHub Actions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pattern Signal Tracker')
    parser.add_argument('action', choices=['scan', 'evaluate'], 
                       help='Action to perform: scan for new signals or evaluate past signals')
    parser.add_argument('--ticker', default='NVDA', help='Stock ticker to analyze')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Hours to look back for evaluation (default: 24)')
    
    args = parser.parse_args()
    
    tracker = PatternSignalTracker()
    
    if args.action == 'scan':
        # Scan for patterns and save signals
        signals = tracker.scan_patterns(args.ticker)
        if signals:
            tracker.save_signals_to_azure(signals, args.ticker)
            print(f"Found and saved {len(signals)} signals for {args.ticker}")
            for signal in signals:
                print(f"- {signal['pattern']}: {signal['action']} at ${signal['entry_price']}")
        else:
            print(f"No patterns found for {args.ticker}")
    
    elif args.action == 'evaluate':
        # Evaluate past signals
        stats = tracker.evaluate_signals(args.ticker, args.hours)
        if stats:
            print(f"\nEvaluation Results for {args.ticker}:")
            print(f"Total Signals: {stats['total_signals']}")
            print(f"Win Rate: {stats['win_rate']}%")
            print(f"Hits Target: {stats['hit_target']}")
            print(f"Hits Stop: {stats['hit_stop']}")
            print(f"Still Open: {stats['triggered_signals']}")
            print(f"Pending: {stats['pending_signals']}")

if __name__ == "__main__":
    main()