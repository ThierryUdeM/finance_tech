#!/usr/bin/env python3
"""
Optimized Multi-Ticker YOLO Predictor with Consolidated Azure Storage
Stores all predictions and evaluations in consolidated files to minimize storage operations
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import yfinance as yf
import mplfinance as mpf
import pandas as pd
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'tickers': {
        'BTC-USD': {
            'name': 'Bitcoin',
            'evaluation_threshold': 0.5
        },
        'NVDA': {
            'name': 'NVIDIA',
            'evaluation_threshold': 0.3
        },
        'AC.TO': {
            'name': 'Air Canada',
            'evaluation_threshold': 0.5
        }
    },
    'intervals': ['15m', '1h', '4h', '1d'],
    'model_path': '../ChartScanAI/weights/custom_yolov8.pt',
    'confidence_threshold': 0.3,
    'max_history_hours': 168  # Keep 7 days of hourly data
}

class OptimizedMultiTickerPredictor:
    def __init__(self):
        """Initialize the predictor with Azure connection"""
        load_dotenv('config/.env')
        
        # Azure connection
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        access_key = os.getenv('ACCESS_KEY')
        container_name = os.getenv('CONTAINER_NAME')
        
        if not all([storage_account_name, access_key, container_name]):
            raise ValueError("Azure credentials not found in environment")
        
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={storage_account_name};"
            f"AccountKey={access_key};"
            f"EndpointSuffix=core.windows.net"
        )
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        
        # Ensure container exists
        try:
            self.blob_service_client.create_container(self.container_name)
        except:
            pass
        
        # Load YOLO model
        self.model = YOLO(CONFIG['model_path'])
        
        # Create temp directory for charts
        os.makedirs('temp_charts', exist_ok=True)
        
    def load_consolidated_data(self, blob_name: str) -> Dict:
        """Load consolidated JSON data from Azure"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            data = blob_client.download_blob().readall()
            return json.loads(data)
        except:
            # Return empty structure if file doesn't exist
            return {}
    
    def save_consolidated_data(self, data: Dict, blob_name: str):
        """Save consolidated JSON data to Azure"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            json_data = json.dumps(data, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)
            logger.info(f"Saved consolidated data to: {blob_name}")
        except Exception as e:
            logger.error(f"Error saving to Azure: {e}")
    
    def cleanup_old_data(self, data: Dict, max_hours: int = 168):
        """Remove data older than max_hours from consolidated structure"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:00:00')
        
        cleaned_data = {}
        for timestamp, entries in data.items():
            if timestamp >= cutoff_str:
                cleaned_data[timestamp] = entries
        
        return cleaned_data
    
    def fetch_ticker_data(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch ticker data for given interval"""
        try:
            stock = yf.Ticker(ticker)
            period_map = {
                '15m': '5d',
                '1h': '2wk',
                '4h': '3mo',
                '1d': '6mo'
            }
            
            period = period_map.get(interval, '1mo')
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker} at {interval}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker} data: {e}")
            return None
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for ticker"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        except:
            return None
    
    def generate_chart(self, data: pd.DataFrame, ticker: str, interval: str) -> str:
        """Generate candlestick chart"""
        chart_path = f'temp_charts/{ticker}_{interval}.png'
        
        try:
            # Create chart with minimal style
            mpf.plot(
                data[-100:],  # Last 100 candles
                type='candle',
                style='charles',
                title=f'{ticker} - {interval}',
                volume=True,
                savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight')
            )
            return chart_path
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    
    def run_detection(self, chart_path: str) -> Dict:
        """Run YOLO detection on chart"""
        try:
            results = self.model(chart_path, conf=CONFIG['confidence_threshold'])
            
            if len(results) == 0:
                return {'buy_signals': 0, 'sell_signals': 0, 'patterns': []}
            
            result = results[0]
            buy_signals = 0
            sell_signals = 0
            patterns = []
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    if 'buy' in class_name.lower():
                        buy_signals += 1
                    elif 'sell' in class_name.lower():
                        sell_signals += 1
                    
                    patterns.append({
                        'pattern': class_name,
                        'confidence': confidence
                    })
            
            return {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'patterns': patterns
            }
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return {'buy_signals': 0, 'sell_signals': 0, 'patterns': []}
    
    def run_all_predictions(self) -> Dict:
        """Run predictions for all tickers and intervals"""
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:00:00')
        
        # Load existing predictions
        predictions_blob = f"consolidated/predictions_latest.json"
        all_predictions = self.load_consolidated_data(predictions_blob)
        
        # Clean up old data
        all_predictions = self.cleanup_old_data(all_predictions, CONFIG['max_history_hours'])
        
        # Initialize structure for current hour
        current_predictions = {}
        
        for ticker in CONFIG['tickers']:
            logger.info(f"Processing {ticker}...")
            ticker_results = {
                'ticker': ticker,
                'timestamp': timestamp_str,
                'current_price': self.get_current_price(ticker),
                'intervals': {},
                'total_buy_signals': 0,
                'total_sell_signals': 0
            }
            
            for interval in CONFIG['intervals']:
                # Fetch data
                data = self.fetch_ticker_data(ticker, interval)
                if data is None:
                    continue
                
                # Generate chart
                chart_path = self.generate_chart(data, ticker, interval)
                if not chart_path:
                    continue
                
                # Run detection
                detection = self.run_detection(chart_path)
                
                # Store interval results
                ticker_results['intervals'][interval] = {
                    'buy_signals': detection['buy_signals'],
                    'sell_signals': detection['sell_signals'],
                    'patterns': detection['patterns'],
                    'signal': 'BUY' if detection['buy_signals'] > detection['sell_signals'] else 
                             'SELL' if detection['sell_signals'] > detection['buy_signals'] else 'HOLD',
                    'avg_confidence': sum(p['confidence'] for p in detection['patterns']) / len(detection['patterns']) 
                                     if detection['patterns'] else 0
                }
                
                ticker_results['total_buy_signals'] += detection['buy_signals']
                ticker_results['total_sell_signals'] += detection['sell_signals']
                
                # Save chart to Azure (still individual files for charts)
                if os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        chart_blob = f"charts/{ticker}/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}/{interval}.png"
                        blob_client = self.blob_service_client.get_blob_client(
                            container=self.container_name,
                            blob=chart_blob
                        )
                        blob_client.upload_blob(f, overwrite=True)
                    os.remove(chart_path)
            
            # Determine overall recommendation
            if ticker_results['total_buy_signals'] > ticker_results['total_sell_signals']:
                ticker_results['recommendation'] = 'BUY'
            elif ticker_results['total_sell_signals'] > ticker_results['total_buy_signals']:
                ticker_results['recommendation'] = 'SELL'
            else:
                ticker_results['recommendation'] = 'HOLD'
            
            current_predictions[ticker] = ticker_results
            logger.info(f"{ticker} complete: {ticker_results['recommendation']}")
        
        # Update consolidated predictions
        all_predictions[timestamp_str] = current_predictions
        self.save_consolidated_data(all_predictions, predictions_blob)
        
        # Also save a "latest" file for quick access
        latest_blob = f"consolidated/predictions_current.json"
        self.save_consolidated_data(current_predictions, latest_blob)
        
        return current_predictions
    
    def evaluate_all_predictions(self):
        """Evaluate past predictions for all tickers"""
        # Load predictions and evaluations
        predictions_blob = f"consolidated/predictions_latest.json"
        evaluations_blob = f"consolidated/evaluations_latest.json"
        
        all_predictions = self.load_consolidated_data(predictions_blob)
        all_evaluations = self.load_consolidated_data(evaluations_blob)
        
        # Clean up old evaluations
        all_evaluations = self.cleanup_old_data(all_evaluations, CONFIG['max_history_hours'])
        
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        current_evaluations = {}
        
        # Evaluate predictions from 1 hour ago
        eval_time = timestamp - timedelta(hours=1)
        eval_key = eval_time.strftime('%Y-%m-%d %H:00:00')
        
        if eval_key in all_predictions:
            past_predictions = all_predictions[eval_key]
            
            for ticker, pred_data in past_predictions.items():
                current_price = self.get_current_price(ticker)
                if not current_price or not pred_data.get('current_price'):
                    continue
                
                past_price = pred_data['current_price']
                price_change_pct = ((current_price - past_price) / past_price) * 100
                
                threshold = CONFIG['tickers'][ticker]['evaluation_threshold']
                
                # Determine if prediction was correct
                recommendation = pred_data.get('recommendation', 'HOLD')
                if recommendation == 'BUY':
                    success = price_change_pct > threshold
                elif recommendation == 'SELL':
                    success = price_change_pct < -threshold
                else:
                    success = abs(price_change_pct) <= threshold
                
                current_evaluations[ticker] = {
                    'timestamp': timestamp_str,
                    'evaluation_time': eval_key,
                    'past_price': past_price,
                    'current_price': current_price,
                    'price_change_pct': round(price_change_pct, 3),
                    'recommendation': recommendation,
                    'success': success,
                    'threshold': threshold
                }
                
                logger.info(f"{ticker} evaluation: {'SUCCESS' if success else 'FAIL'} "
                          f"(Change: {price_change_pct:.2f}%)")
        
        # Update consolidated evaluations
        all_evaluations[timestamp_str] = current_evaluations
        self.save_consolidated_data(all_evaluations, evaluations_blob)
        
        # Generate performance summary
        self.generate_performance_summary(all_evaluations)
        
        return current_evaluations
    
    def generate_performance_summary(self, all_evaluations: Dict):
        """Generate performance summary from all evaluations"""
        summary = {
            'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers': {}
        }
        
        # Calculate metrics per ticker
        for ticker in CONFIG['tickers']:
            ticker_evals = []
            
            # Collect all evaluations for this ticker
            for timestamp, evals in all_evaluations.items():
                if ticker in evals:
                    ticker_evals.append(evals[ticker])
            
            if ticker_evals:
                total = len(ticker_evals)
                successes = sum(1 for e in ticker_evals if e.get('success', False))
                
                summary['tickers'][ticker] = {
                    'total_evaluations': total,
                    'successful': successes,
                    'failed': total - successes,
                    'win_rate': round((successes / total) * 100, 2) if total > 0 else 0,
                    'avg_price_change': round(
                        sum(e.get('price_change_pct', 0) for e in ticker_evals) / total, 3
                    ) if total > 0 else 0
                }
        
        # Save summary
        summary_blob = f"consolidated/performance_summary.json"
        self.save_consolidated_data(summary, summary_blob)
        
        return summary
    
    def cleanup_temp_files(self):
        """Clean up temporary chart files"""
        try:
            for file in os.listdir('temp_charts'):
                os.remove(os.path.join('temp_charts', file))
            os.rmdir('temp_charts')
        except:
            pass


def main():
    """Main execution function"""
    logger.info("Starting optimized multi-ticker predictions...")
    
    predictor = OptimizedMultiTickerPredictor()
    
    try:
        # Run predictions
        predictions = predictor.run_all_predictions()
        logger.info(f"Completed predictions for {len(predictions)} tickers")
        
        # Run evaluations
        evaluations = predictor.evaluate_all_predictions()
        logger.info(f"Completed evaluations for {len(evaluations)} tickers")
        
        # Print summary
        print("\n=== PREDICTION SUMMARY ===")
        for ticker, data in predictions.items():
            print(f"\n{ticker}:")
            print(f"  Recommendation: {data['recommendation']}")
            print(f"  Buy Signals: {data['total_buy_signals']}")
            print(f"  Sell Signals: {data['total_sell_signals']}")
        
        print("\n=== EVALUATION SUMMARY ===")
        for ticker, eval_data in evaluations.items():
            print(f"\n{ticker}:")
            print(f"  Success: {eval_data['success']}")
            print(f"  Price Change: {eval_data['price_change_pct']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        predictor.cleanup_temp_files()


if __name__ == "__main__":
    main()