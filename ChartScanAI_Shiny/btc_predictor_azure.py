#!/usr/bin/env python3
"""
Bitcoin YOLO Predictor with Azure Storage
Runs predictions and stores results in Azure Blob Storage
Evaluates past predictions for performance tracking
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
    'ticker': 'BTC-USD',
    'intervals': ['15m', '1h', '4h', '1d'],
    'model_path': '../ChartScanAI/weights/custom_yolov8.pt',
    'confidence_threshold': 0.3,
    'evaluation_threshold': 0.5  # Price change % for success
}

class BTCPredictor:
    def __init__(self):
        """Initialize the predictor with Azure connection"""
        load_dotenv('config/.env')
        
        # Azure connection using individual credentials
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        access_key = os.getenv('ACCESS_KEY')
        container_name = os.getenv('CONTAINER_NAME')
        
        if not all([storage_account_name, access_key, container_name]):
            raise ValueError("Azure credentials not found in environment")
        
        # Create connection string from components
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
            logger.info(f"Created container: {self.container_name}")
        except Exception as e:
            logger.info(f"Container already exists or error: {e}")
        
        # Load YOLO model
        self.model = YOLO(CONFIG['model_path'])
        
    def fetch_btc_data(self, interval: str) -> Optional[pd.DataFrame]:
        """Fetch Bitcoin data for given interval"""
        try:
            btc = yf.Ticker(CONFIG['ticker'])
            
            # Determine period based on interval - Pure Intraday Signals
            # Use minimum period that provides sufficient recent data for pattern recognition
            period_map = {
                '15m': '1d',   # ~16 hours of 15-min data for intraday patterns
                '1h': '1d',    # ~16 hours of hourly data for intraday patterns  
                '4h': '1d',    # ~16 hours of 4-hour data for daily patterns
                '1d': '5d'     # 5 days of daily data for weekly patterns
            }
            
            period = period_map.get(interval, '1mo')
            data = btc.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data for interval {interval}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {interval}: {e}")
            return None
    
    def generate_chart(self, data: pd.DataFrame, interval: str, save_path: str) -> bool:
        """Generate candlestick chart"""
        try:
            mpf.plot(
                data,
                type='candle',
                style='yahoo',
                title=f'BTC-USD - {interval}',
                volume=True,
                savefig=save_path,
                figsize=(12, 8),
                warn_too_much_data=800
            )
            return True
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return False
    
    def run_yolo_detection(self, image_path: str) -> Dict:
        """Run YOLO detection on chart"""
        try:
            results = self.model(image_path, conf=CONFIG['confidence_threshold'])
            
            detections = []
            if len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        detections.append({
                            'class': class_name,
                            'confidence': conf
                        })
            
            # Count signals
            buy_count = sum(1 for d in detections if d['class'] == 'Buy')
            sell_count = sum(1 for d in detections if d['class'] == 'Sell')
            
            return {
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'detections': detections,
                'avg_confidence': sum(d['confidence'] for d in detections) / len(detections) if detections else 0
            }
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return {'buy_signals': 0, 'sell_signals': 0, 'detections': [], 'avg_confidence': 0}
    
    def get_current_price(self) -> Optional[float]:
        """Get current BTC price"""
        try:
            btc = yf.Ticker(CONFIG['ticker'])
            return btc.info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    def save_to_azure(self, data: Dict, blob_name: str):
        """Save JSON data to Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            json_data = json.dumps(data, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)
            logger.info(f"Saved to Azure: {blob_name}")
            
        except Exception as e:
            logger.error(f"Error saving to Azure: {e}")
    
    def load_from_azure(self, blob_name: str) -> Optional[Dict]:
        """Load JSON data from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            data = blob_client.download_blob().readall()
            return json.loads(data)
            
        except Exception as e:
            logger.info(f"Could not load {blob_name}: {e}")
            return None
    
    def evaluate_past_predictions(self, hours_ago: int = 1):
        """Evaluate predictions from specified hours ago"""
        evaluation_time = datetime.utcnow() - timedelta(hours=hours_ago)
        blob_name = f"predictions/{evaluation_time.strftime('%Y-%m-%d')}/{evaluation_time.strftime('%H')}.json"
        
        past_data = self.load_from_azure(blob_name)
        if not past_data:
            logger.info(f"No predictions found from {hours_ago} hour(s) ago")
            return None
        
        current_price = self.get_current_price()
        if not current_price:
            return None
        
        past_price = past_data.get('price')
        if not past_price:
            return None
        
        price_change_pct = ((current_price - past_price) / past_price) * 100
        recommendation = past_data.get('recommendation', 'HOLD')
        
        # Determine if prediction was correct
        was_correct = False
        if recommendation in ['BUY', 'STRONG BUY'] and price_change_pct > CONFIG['evaluation_threshold']:
            was_correct = True
        elif recommendation in ['SELL', 'STRONG SELL'] and price_change_pct < -CONFIG['evaluation_threshold']:
            was_correct = True
        elif recommendation == 'HOLD' and abs(price_change_pct) <= CONFIG['evaluation_threshold']:
            was_correct = True
        
        evaluation = {
            'timestamp': datetime.utcnow().isoformat(),
            'ticker': 'BTC-USD',
            'prediction_time': past_data.get('timestamp'),
            'past_price': past_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'recommendation': recommendation,
            'was_correct': was_correct,
            'hours_elapsed': hours_ago
        }
        
        # Save evaluation
        eval_blob_name = f"evaluations/BTC-USD/{datetime.utcnow().strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%H%M%S')}.json"
        self.save_to_azure(evaluation, eval_blob_name)
        
        return evaluation
    
    def run_predictions(self):
        """Run predictions for all intervals"""
        timestamp = datetime.utcnow()
        current_price = self.get_current_price()
        
        if not current_price:
            logger.error("Could not get current price")
            return
        
        all_results = {
            'timestamp': timestamp.isoformat(),
            'price': current_price,
            'intervals': {}
        }
        
        total_buy = 0
        total_sell = 0
        
        # Create temp directory for charts
        os.makedirs('temp_charts', exist_ok=True)
        
        for interval in CONFIG['intervals']:
            logger.info(f"Processing {interval}...")
            
            # Fetch data
            data = self.fetch_btc_data(interval)
            if data is None:
                continue
            
            # Generate chart
            chart_path = f"temp_charts/btc_{interval}.png"
            if not self.generate_chart(data, interval, chart_path):
                continue
            
            # Run detection
            detection_result = self.run_yolo_detection(chart_path)
            
            # Store results
            all_results['intervals'][interval] = {
                'buy_signals': detection_result['buy_signals'],
                'sell_signals': detection_result['sell_signals'],
                'avg_confidence': detection_result['avg_confidence'],
                'signal': 'BUY' if detection_result['buy_signals'] > detection_result['sell_signals'] else
                         'SELL' if detection_result['sell_signals'] > detection_result['buy_signals'] else 'NEUTRAL'
            }
            
            total_buy += detection_result['buy_signals']
            total_sell += detection_result['sell_signals']
            
            # Save chart to Azure
            with open(chart_path, 'rb') as f:
                chart_blob_name = f"charts/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}/{interval}.png"
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=chart_blob_name
                )
                blob_client.upload_blob(f, overwrite=True)
            
            # Clean up local chart
            os.remove(chart_path)
        
        # Overall recommendation
        if total_buy > total_sell * 1.5:
            recommendation = 'STRONG BUY'
        elif total_sell > total_buy * 1.5:
            recommendation = 'STRONG SELL'
        elif total_buy > total_sell:
            recommendation = 'BUY'
        elif total_sell > total_buy:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        all_results['total_buy_signals'] = total_buy
        all_results['total_sell_signals'] = total_sell
        all_results['recommendation'] = recommendation
        
        # Save predictions
        blob_name = f"predictions/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}.json"
        self.save_to_azure(all_results, blob_name)
        
        logger.info(f"Prediction complete: {recommendation} (Buy: {total_buy}, Sell: {total_sell})")
        
        # Clean up
        os.rmdir('temp_charts')
        
        return all_results
    
    def generate_performance_report(self, days: int = 7):
        """Generate performance report for past N days"""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        evaluations = []
        for blob in container_client.list_blobs(name_starts_with='evaluations/'):
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob.name
                )
                data = json.loads(blob_client.download_blob().readall())
                evaluations.append(data)
            except Exception as e:
                logger.error(f"Error loading evaluation: {e}")
        
        if not evaluations:
            logger.info("No evaluations found")
            return
        
        # Filter to last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        evaluations = [e for e in evaluations 
                      if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > cutoff_date]
        
        # Calculate statistics
        total = len(evaluations)
        correct = sum(1 for e in evaluations if e['was_correct'])
        
        buy_evals = [e for e in evaluations if e['recommendation'] in ['BUY', 'STRONG BUY']]
        sell_evals = [e for e in evaluations if e['recommendation'] in ['SELL', 'STRONG SELL']]
        hold_evals = [e for e in evaluations if e['recommendation'] == 'HOLD']
        
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'period_days': days,
            'total_predictions': total,
            'correct_predictions': correct,
            'overall_accuracy': (correct / total * 100) if total > 0 else 0,
            'buy_accuracy': (sum(1 for e in buy_evals if e['was_correct']) / len(buy_evals) * 100) if buy_evals else 0,
            'sell_accuracy': (sum(1 for e in sell_evals if e['was_correct']) / len(sell_evals) * 100) if sell_evals else 0,
            'hold_accuracy': (sum(1 for e in hold_evals if e['was_correct']) / len(hold_evals) * 100) if hold_evals else 0,
            'buy_count': len(buy_evals),
            'sell_count': len(sell_evals),
            'hold_count': len(hold_evals)
        }
        
        # Save report
        report_blob = f"reports/performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_to_azure(report, report_blob)
        
        logger.info(f"Performance Report ({days} days):")
        logger.info(f"  Overall Accuracy: {report['overall_accuracy']:.1f}% ({correct}/{total})")
        logger.info(f"  Buy Accuracy: {report['buy_accuracy']:.1f}% ({report['buy_count']} total)")
        logger.info(f"  Sell Accuracy: {report['sell_accuracy']:.1f}% ({report['sell_count']} total)")
        logger.info(f"  Hold Accuracy: {report['hold_accuracy']:.1f}% ({report['hold_count']} total)")
        
        return report


def main():
    """Main execution function"""
    predictor = BTCPredictor()
    
    # Run predictions
    logger.info("Running BTC predictions...")
    results = predictor.run_predictions()
    
    # Evaluate past predictions (1 hour ago)
    logger.info("Evaluating predictions from 1 hour ago...")
    evaluation = predictor.evaluate_past_predictions(hours_ago=1)
    if evaluation:
        logger.info(f"Past prediction was {'CORRECT' if evaluation['was_correct'] else 'INCORRECT'}")
        logger.info(f"Price changed by {evaluation['price_change_pct']:.2f}%")
    
    # Generate weekly performance report on Sundays
    if datetime.utcnow().weekday() == 6:  # Sunday
        logger.info("Generating weekly performance report...")
        predictor.generate_performance_report(days=7)
    
    # Output for GitHub Actions
    if results:
        print(f"::set-output name=recommendation::{results['recommendation']}")
        print(f"::set-output name=buy_signals::{results['total_buy_signals']}")
        print(f"::set-output name=sell_signals::{results['total_sell_signals']}")


if __name__ == "__main__":
    main()