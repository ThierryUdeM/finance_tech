#!/usr/bin/env python3
"""
Multi-Ticker YOLO Predictor with Volume Confirmation
Enhanced version with comprehensive volume analysis for better signal quality
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
import numpy as np
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Add path for volume confirmation module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'signal', 'directional_analysis'))

try:
    from volume_confirmation import VolumeConfirmation
    VOLUME_MODULE_AVAILABLE = True
except ImportError:
    print("Warning: Volume confirmation module not available")
    VOLUME_MODULE_AVAILABLE = False

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
    'confidence_threshold': 0.3,
    'volume_analysis': {
        'enabled': True,
        'lookback_periods': 20,
        'min_volume_score': 0.3,
        'volume_weight': 0.3  # How much volume affects final confidence
    }
}

class VolumeEnhancedMultiTickerPredictor:
    def __init__(self):
        """Initialize the predictor with Azure connection and volume analyzer"""
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
        
        # Initialize volume analyzer if available
        if VOLUME_MODULE_AVAILABLE and CONFIG['volume_analysis']['enabled']:
            self.volume_analyzer = VolumeConfirmation(CONFIG['volume_analysis']['lookback_periods'])
            logger.info("Volume confirmation module initialized")
        else:
            self.volume_analyzer = None
            logger.info("Running without volume confirmation")
        
    def fetch_ticker_data(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch ticker data for given interval with volume"""
        try:
            stock = yf.Ticker(ticker)
            
            # Fixed period mapping for intraday signals
            period_map = {
                '15m': '1d',   # ~16 hours for intraday patterns
                '1h': '1d',    # ~16 hours for intraday patterns  
                '4h': '1d',    # ~16 hours for daily patterns
                '1d': '5d'     # 5 days for weekly patterns
            }
            
            period = period_map.get(interval, '1mo')
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data for {ticker} interval {interval}")
                return None
            
            # Ensure we have volume data
            if 'Volume' not in data.columns:
                logger.warning(f"No volume data for {ticker}")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} {interval}: {e}")
            return None
    
    def analyze_volume_patterns(self, data: pd.DataFrame, interval: str) -> Dict:
        """Analyze volume patterns for the given data"""
        if self.volume_analyzer is None or len(data) < 20:
            return {
                'volume_score': 0.5,
                'volume_quality': 'unknown',
                'volume_metrics': {}
            }
        
        try:
            # Calculate volume metrics
            data_with_metrics = self.volume_analyzer.calculate_volume_metrics(data)
            
            # Get recent volume characteristics
            recent_volume_ratio = data_with_metrics['volume_ratio'].iloc[-3:].mean()
            volume_trend = data_with_metrics['Volume'].iloc[-5:].mean() > data_with_metrics['Volume'].iloc[:5].mean()
            obv_signal = data_with_metrics['obv'].iloc[-1] > data_with_metrics['obv_sma'].iloc[-1]
            
            # Calculate volume score based on multiple factors
            volume_score = 0.5  # Base score
            
            # High recent volume is positive
            if recent_volume_ratio > 1.5:
                volume_score += 0.2
            elif recent_volume_ratio > 1.2:
                volume_score += 0.1
            elif recent_volume_ratio < 0.5:
                volume_score -= 0.2
            
            # Volume trend alignment
            if volume_trend:
                volume_score += 0.15
            
            # OBV confirmation
            if obv_signal:
                volume_score += 0.15
            
            # Ensure score is between 0 and 1
            volume_score = max(0, min(1, volume_score))
            
            # Determine quality
            if volume_score > 0.7:
                quality = 'high'
            elif volume_score < 0.3:
                quality = 'low'
            else:
                quality = 'medium'
            
            return {
                'volume_score': round(volume_score, 3),
                'volume_quality': quality,
                'volume_metrics': {
                    'recent_volume_ratio': round(recent_volume_ratio, 2),
                    'volume_trend': 'increasing' if volume_trend else 'decreasing',
                    'obv_signal': 'bullish' if obv_signal else 'bearish',
                    'current_volume': int(data['Volume'].iloc[-1]),
                    'avg_volume': int(data_with_metrics['avg_volume'].iloc[-1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {
                'volume_score': 0.5,
                'volume_quality': 'unknown',
                'volume_metrics': {}
            }
    
    def generate_enhanced_chart(self, data: pd.DataFrame, ticker: str, interval: str, save_path: str, volume_analysis: Dict) -> bool:
        """Generate candlestick chart with volume indicators"""
        try:
            ticker_info = CONFIG['tickers'][ticker]
            
            # Add volume analysis to title
            quality_indicator = {
                'high': '🟢',
                'medium': '🟡',
                'low': '🔴',
                'unknown': '⚪'
            }.get(volume_analysis.get('volume_quality', 'unknown'), '⚪')
            
            title = f'{ticker_info["name"]} ({ticker}) - {interval} {quality_indicator}'
            
            # Create the plot with volume
            mpf.plot(
                data,
                type='candle',
                style='yahoo',
                title=title,
                volume=True,
                savefig=save_path,
                figsize=(12, 8),
                warn_too_much_data=800
            )
            return True
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return False
    
    def run_yolo_detection_with_volume(self, image_path: str, volume_score: float) -> Dict:
        """Run YOLO detection and adjust confidence based on volume"""
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
                        
                        # Adjust confidence based on volume score
                        volume_weight = CONFIG['volume_analysis']['volume_weight']
                        adjusted_conf = conf * (1 - volume_weight) + (conf * volume_score) * volume_weight
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'volume_adjusted_confidence': adjusted_conf
                        })
            
            # Count signals with volume-adjusted confidence
            buy_count = sum(1 for d in detections if d['class'] == 'Buy' and d['volume_adjusted_confidence'] > CONFIG['confidence_threshold'])
            sell_count = sum(1 for d in detections if d['class'] == 'Sell' and d['volume_adjusted_confidence'] > CONFIG['confidence_threshold'])
            
            # Calculate average confidences
            avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
            avg_adjusted_confidence = sum(d['volume_adjusted_confidence'] for d in detections) / len(detections) if detections else 0
            
            return {
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'detections': detections,
                'avg_confidence': avg_confidence,
                'avg_volume_adjusted_confidence': avg_adjusted_confidence,
                'volume_impact': avg_adjusted_confidence - avg_confidence if detections else 0
            }
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return {
                'buy_signals': 0, 
                'sell_signals': 0, 
                'detections': [], 
                'avg_confidence': 0,
                'avg_volume_adjusted_confidence': 0,
                'volume_impact': 0
            }
    
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
    
    def run_predictions_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Run predictions for a single ticker with volume analysis"""
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
            'intervals': {},
            'volume_analysis_enabled': self.volume_analyzer is not None
        }
        
        total_buy = 0
        total_sell = 0
        weighted_volume_score = 0
        volume_scores = []
        
        # Create temp directory for charts
        os.makedirs('temp_charts', exist_ok=True)
        
        for interval in CONFIG['intervals']:
            logger.info(f"Processing {ticker} {interval}...")
            
            # Fetch data
            data = self.fetch_ticker_data(ticker, interval)
            if data is None:
                continue
            
            # Analyze volume patterns
            volume_analysis = self.analyze_volume_patterns(data, interval)
            volume_scores.append(volume_analysis['volume_score'])
            
            # Generate chart
            chart_path = f"temp_charts/{ticker.replace('.', '_')}_{interval}.png"
            if not self.generate_enhanced_chart(data, ticker, interval, chart_path, volume_analysis):
                continue
            
            # Run detection with volume adjustment
            detection_result = self.run_yolo_detection_with_volume(chart_path, volume_analysis['volume_score'])
            
            # Store results with volume information
            all_results['intervals'][interval] = {
                'buy_signals': detection_result['buy_signals'],
                'sell_signals': detection_result['sell_signals'],
                'avg_confidence': detection_result['avg_confidence'],
                'avg_volume_adjusted_confidence': detection_result['avg_volume_adjusted_confidence'],
                'volume_score': volume_analysis['volume_score'],
                'volume_quality': volume_analysis['volume_quality'],
                'volume_metrics': volume_analysis['volume_metrics'],
                'signal': 'BUY' if detection_result['buy_signals'] > detection_result['sell_signals'] else
                         'SELL' if detection_result['sell_signals'] > detection_result['buy_signals'] else 'NEUTRAL',
                'data_points': len(data),
                'period_analyzed': f"{data.index[0].strftime('%Y-%m-%d %H:%M')} to {data.index[-1].strftime('%Y-%m-%d %H:%M')}"
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
        
        # Calculate overall volume score
        if volume_scores:
            weighted_volume_score = np.mean(volume_scores)
        else:
            weighted_volume_score = 0.5
        
        # Overall recommendation with volume adjustment
        volume_factor = 1.0
        if weighted_volume_score < CONFIG['volume_analysis']['min_volume_score']:
            volume_factor = 0.7  # Reduce signal strength for low volume
        elif weighted_volume_score > 0.7:
            volume_factor = 1.2  # Boost signal strength for high volume
        
        adjusted_buy = total_buy * volume_factor
        adjusted_sell = total_sell * volume_factor
        
        if adjusted_buy > adjusted_sell * 1.5:
            recommendation = 'STRONG BUY'
        elif adjusted_sell > adjusted_buy * 1.5:
            recommendation = 'STRONG SELL'
        elif adjusted_buy > adjusted_sell:
            recommendation = 'BUY'
        elif adjusted_sell > adjusted_buy:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        # Add volume warning if needed
        if weighted_volume_score < CONFIG['volume_analysis']['min_volume_score']:
            recommendation += ' (Low Volume Warning)'
        
        all_results['total_buy_signals'] = total_buy
        all_results['total_sell_signals'] = total_sell
        all_results['recommendation'] = recommendation
        all_results['overall_volume_score'] = round(weighted_volume_score, 3)
        all_results['overall_volume_quality'] = 'high' if weighted_volume_score > 0.7 else ('low' if weighted_volume_score < 0.3 else 'medium')
        
        # Save predictions
        blob_name = f"predictions/{ticker}/{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H')}.json"
        self.save_to_azure(all_results, blob_name)
        
        logger.info(f"{ticker} prediction complete: {recommendation} (Buy: {total_buy}, Sell: {total_sell}, Volume Score: {weighted_volume_score:.3f})")
        
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
        
        # Save combined summary with volume information
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'tickers_analyzed': len(results),
            'volume_analysis_enabled': self.volume_analyzer is not None,
            'recommendations': {
                ticker: {
                    'signal': result['recommendation'],
                    'volume_score': result['overall_volume_score'],
                    'volume_quality': result['overall_volume_quality']
                }
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
            'threshold_used': threshold,
            'volume_score': past_data.get('overall_volume_score', 'N/A'),
            'volume_quality': past_data.get('overall_volume_quality', 'N/A')
        }
        
        # Save evaluation
        eval_blob_name = f"evaluations/{ticker}/{datetime.utcnow().strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%H%M%S')}.json"
        self.save_to_azure(evaluation, eval_blob_name)
        
        return evaluation
    
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
            
            # Calculate volume statistics
            volume_scores = [e.get('volume_score', 0.5) for e in evaluations if isinstance(e.get('volume_score'), (int, float))]
            avg_volume_score = np.mean(volume_scores) if volume_scores else None
            
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
                'hold_count': len(hold_evals),
                'avg_volume_score': round(avg_volume_score, 3) if avg_volume_score is not None else 'N/A'
            }
            
            reports[ticker] = ticker_report
            
            logger.info(f"\n{ticker} Performance Report ({days} days):")
            logger.info(f"  Overall Accuracy: {ticker_report['overall_accuracy']:.1f}% ({correct}/{total})")
            logger.info(f"  Buy Accuracy: {ticker_report['buy_accuracy']:.1f}% ({ticker_report['buy_count']} total)")
            logger.info(f"  Sell Accuracy: {ticker_report['sell_accuracy']:.1f}% ({ticker_report['sell_count']} total)")
            logger.info(f"  Hold Accuracy: {ticker_report['hold_accuracy']:.1f}% ({ticker_report['hold_count']} total)")
            if ticker_report['avg_volume_score'] != 'N/A':
                logger.info(f"  Avg Volume Score: {ticker_report['avg_volume_score']}")
        
        # Save combined report
        combined_report = {
            'generated_at': datetime.utcnow().isoformat(),
            'period_days': days,
            'volume_analysis_enabled': True,
            'ticker_reports': reports
        }
        
        report_blob = f"reports/performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_to_azure(combined_report, report_blob)
        
        return combined_report


def main():
    """Main execution function"""
    predictor = VolumeEnhancedMultiTickerPredictor()
    
    # Run predictions for all tickers
    logger.info("Running multi-ticker predictions with volume analysis...")
    results = predictor.run_all_predictions()
    
    # Output summary
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY WITH VOLUME ANALYSIS")
    logger.info("="*60)
    
    for ticker, result in results.items():
        logger.info(f"\n{ticker}:")
        logger.info(f"  Recommendation: {result['recommendation']}")
        logger.info(f"  Volume Score: {result['overall_volume_score']}")
        logger.info(f"  Volume Quality: {result['overall_volume_quality']}")
        logger.info(f"  Buy Signals: {result['total_buy_signals']}")
        logger.info(f"  Sell Signals: {result['total_sell_signals']}")
    
    # Output for GitHub Actions
    if results:
        # Create a combined recommendation
        strong_buys = sum(1 for r in results.values() if 'STRONG BUY' in r['recommendation'])
        buys = sum(1 for r in results.values() if r['recommendation'] == 'BUY')
        sells = sum(1 for r in results.values() if r['recommendation'] == 'SELL')
        strong_sells = sum(1 for r in results.values() if 'STRONG SELL' in r['recommendation'])
        
        print(f"::set-output name=strong_buy_count::{strong_buys}")
        print(f"::set-output name=buy_count::{buys}")
        print(f"::set-output name=sell_count::{sells}")
        print(f"::set-output name=strong_sell_count::{strong_sells}")


if __name__ == "__main__":
    main()