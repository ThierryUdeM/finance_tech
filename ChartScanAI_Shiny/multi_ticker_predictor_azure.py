#!/usr/bin/env python3
"""
Multi-Ticker YOLO Predictor with Azure Storage
Runs predictions for multiple stocks and stores results in Azure Blob Storage
Evaluates past predictions for performance tracking
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
            'evaluation_threshold': 0.5  # Price change % for success
        },
        'NVDA': {
            'name': 'NVIDIA',
            'evaluation_threshold': 0.3  # Lower threshold for stocks
        },
        'AC.TO': {
            'name': 'Air Canada',
            'evaluation_threshold': 0.5
        }
    },
    'intervals': ['15m', '1h', '4h', '1d'],
    'model_path': '../ChartScanAI/weights/custom_yolov8.pt',
    'confidence_threshold': 0.3
}

class MultiTickerPredictor:
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
        
    def fetch_ticker_data(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch ticker data for given interval"""
        try:
            stock = yf.Ticker(ticker)
            
            # Determine period based on interval
            period_map = {
                '15m': '5d',
                '1h': '2wk',
                '4h': '3mo',
                '1d': '6mo'
            }
            
            period = period_map.get(interval, '1mo')
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data for {ticker} interval {interval}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} {interval}: {e}")
            return None
    
    def generate_chart(self, data: pd.DataFrame, ticker: str, interval: str, save_path: str) -> bool:
        """Generate candlestick chart"""
        try:
            ticker_info = CONFIG['tickers'][ticker]
            mpf.plot(
                data,
                type='candle',
                style='yahoo',
                title=f'{ticker_info["name"]} ({ticker}) - {interval}',
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
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current ticker price"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
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
    
    def evaluate_past_predictions(self, ticker: str, hours_ago: int = 1):
        """Evaluate predictions from specified hours ago"""
        evaluation_time = datetime.utcnow() - timedelta(hours=hours_ago)
        blob_name = f"predictions/{ticker}/{evaluation_time.strftime('%Y-%m-%d')}/{evaluation_time.strftime('%H')}.json"
        
        past_data = self.load_from_azure(blob_name)
        if not past_data:
            logger.info(f"No predictions found for {ticker} from {hours_ago} hour(s) ago")
            return None
        
        current_price = self.get_current_price(ticker)
        if not current_price:
            return None
        
        past_price = past_data.get('price')
        if not past_price:
            return None
        
        price_change_pct = ((current_price - past_price) / past_price) * 100
        recommendation = past_data.get('recommendation', 'HOLD')
        
        # Determine if prediction was correct
        threshold = CONFIG['tickers'][ticker]['evaluation_threshold']
        was_correct = False
        if recommendation in ['BUY', 'STRONG BUY'] and price_change_pct > threshold:
            was_correct = True
        elif recommendation in ['SELL', 'STRONG SELL'] and price_change_pct < -threshold:
            was_correct = True
        elif recommendation == 'HOLD' and abs(price_change_pct) <= threshold:
            was_correct = True
        
        evaluation = {
            'timestamp': datetime.utcnow().isoformat(),
            'ticker': ticker,
            'prediction_time': past_data.get('timestamp'),
            'past_price': past_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'recommendation': recommendation,
            'was_correct': was_correct,
            'hours_elapsed': hours_ago,
            'threshold_used': threshold
        }
        
        # Save evaluation
        eval_blob_name = f"evaluations/{ticker}/{datetime.utcnow().strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%H%M%S')}.json"
        self.save_to_azure(evaluation, eval_blob_name)
        
        return evaluation
    
    def run_predictions_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Run predictions for a single ticker"""
        timestamp = datetime.utcnow()
        current_price = self.get_current_price(ticker)
        
        if not current_price:
            logger.error(f"Could not get current price for {ticker}")
            return None
        
        ticker_info = CONFIG['tickers'][ticker]
        
        all_results = {
            'timestamp': timestamp.isoformat(),
            'ticker': ticker,
            'name': ticker_info['name'],
            'price': current_price,
            'intervals': {}
        }
        
        total_buy = 0
        total_sell = 0
        
        # Create temp directory for charts
        os.makedirs('temp_charts', exist_ok=True)
        
        for interval in CONFIG['intervals']:
            logger.info(f"Processing {ticker} {interval}...")
            
            # Fetch data
            data = self.fetch_ticker_data(ticker, interval)
            if data is None:
                continue
            
            # Generate chart
            chart_path = f"temp_charts/{ticker.replace('.', '_')}_{interval}.png"
            if not self.generate_chart(data, ticker, interval, chart_path):
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
                chart_blob_name = f"charts/{ticker}/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}/{interval}.png"
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
        blob_name = f"predictions/{ticker}/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}.json"
        self.save_to_azure(all_results, blob_name)
        
        logger.info(f"{ticker} prediction complete: {recommendation} (Buy: {total_buy}, Sell: {total_sell})")
        
        return all_results
    
    def run_all_predictions(self) -> Dict[str, Dict]:
        """Run predictions for all configured tickers"""
        results = {}
        
        for ticker in CONFIG['tickers']:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running predictions for {ticker}")
                logger.info(f"{'='*50}")
                
                result = self.run_predictions_for_ticker(ticker)
                if result:
                    results[ticker] = result
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        # Save combined summary
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'tickers_analyzed': len(results),
            'recommendations': {
                ticker: result['recommendation'] 
                for ticker, result in results.items()
            }
        }
        
        summary_blob = f"summaries/{datetime.utcnow().strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%H')}.json"
        self.save_to_azure(summary, summary_blob)
        
        # Clean up temp directory
        try:
            os.rmdir('temp_charts')
        except:
            pass
        
        return results
    
    def generate_performance_report(self, days: int = 7):
        """Generate performance report for past N days for all tickers"""
        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        reports = {}
        
        for ticker in CONFIG['tickers']:
            evaluations = []
            prefix = f'evaluations/{ticker}/'
            
            for blob in container_client.list_blobs(name_starts_with=prefix):
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
                logger.info(f"No evaluations found for {ticker}")
                continue
            
            # Filter to last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            evaluations = [e for e in evaluations 
                          if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > cutoff_date]
            
            if not evaluations:
                continue
            
            # Calculate statistics
            total = len(evaluations)
            correct = sum(1 for e in evaluations if e['was_correct'])
            
            buy_evals = [e for e in evaluations if e['recommendation'] in ['BUY', 'STRONG BUY']]
            sell_evals = [e for e in evaluations if e['recommendation'] in ['SELL', 'STRONG SELL']]
            hold_evals = [e for e in evaluations if e['recommendation'] == 'HOLD']
            
            ticker_report = {
                'ticker': ticker,
                'name': CONFIG['tickers'][ticker]['name'],
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
            
            reports[ticker] = ticker_report
            
            logger.info(f"\n{ticker} Performance Report ({days} days):")
            logger.info(f"  Overall Accuracy: {ticker_report['overall_accuracy']:.1f}% ({correct}/{total})")
            logger.info(f"  Buy Accuracy: {ticker_report['buy_accuracy']:.1f}% ({ticker_report['buy_count']} total)")
            logger.info(f"  Sell Accuracy: {ticker_report['sell_accuracy']:.1f}% ({ticker_report['sell_count']} total)")
            logger.info(f"  Hold Accuracy: {ticker_report['hold_accuracy']:.1f}% ({ticker_report['hold_count']} total)")
        
        # Save combined report
        combined_report = {
            'generated_at': datetime.utcnow().isoformat(),
            'period_days': days,
            'ticker_reports': reports
        }
        
        report_blob = f"reports/performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_to_azure(combined_report, report_blob)
        
        return combined_report


def main():
    """Main execution function"""
    predictor = MultiTickerPredictor()
    
    # Run predictions for all tickers
    logger.info("Running multi-ticker predictions...")
    results = predictor.run_all_predictions()
    
    # Evaluate past predictions (1 hour ago) for each ticker
    logger.info("\nEvaluating predictions from 1 hour ago...")
    for ticker in CONFIG['tickers']:
        evaluation = predictor.evaluate_past_predictions(ticker, hours_ago=1)
        if evaluation:
            logger.info(f"{ticker}: Past prediction was {'CORRECT' if evaluation['was_correct'] else 'INCORRECT'}")
            logger.info(f"  Price changed by {evaluation['price_change_pct']:.2f}%")
    
    # Generate weekly performance report on Sundays
    if datetime.utcnow().weekday() == 6:  # Sunday
        logger.info("\nGenerating weekly performance report...")
        predictor.generate_performance_report(days=7)
    
    # Output for GitHub Actions
    if results:
        for ticker, result in results.items():
            print(f"::set-output name={ticker.replace('.', '_').replace('-', '_')}_recommendation::{result['recommendation']}")
            print(f"::set-output name={ticker.replace('.', '_').replace('-', '_')}_buy_signals::{result['total_buy_signals']}")
            print(f"::set-output name={ticker.replace('.', '_').replace('-', '_')}_sell_signals::{result['total_sell_signals']}")


if __name__ == "__main__":
    main()