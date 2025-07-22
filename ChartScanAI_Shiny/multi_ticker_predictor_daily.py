#!/usr/bin/env python3
"""
Multi-Ticker YOLO Predictor - Daily Chart Version
Designed for swing trading (2-5 day holding periods) using daily charts
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

# Configuration for daily charts
CONFIG = {
    'tickers': {
        'BTC-USD': {
            'name': 'Bitcoin',
            'evaluation_threshold': 2.0,  # Higher threshold for daily moves
            'holding_days': 3
        },
        'NVDA': {
            'name': 'NVIDIA',
            'evaluation_threshold': 1.5,  # Stocks typically have smaller daily moves
            'holding_days': 3
        },
        'AC.TO': {
            'name': 'Air Canada',
            'evaluation_threshold': 2.0,
            'holding_days': 4
        }
    },
    'chart_periods': {
        '1d': {
            'lookback_days': 90,   # 3 months of daily data
            'min_bars': 60,        # Minimum bars for pattern detection
            'description': 'Daily patterns for swing trading'
        },
        '1wk': {
            'lookback_days': 365,  # 1 year of weekly data
            'min_bars': 30,        # Minimum weeks
            'description': 'Weekly patterns for position trading'
        }
    },
    'model_path': '../ChartScanAI/weights/custom_yolov8.pt',
    'confidence_threshold': 0.4,  # Higher threshold for daily signals
    'volume_analysis': {
        'enabled': True,
        'lookback_periods': 50,  # More periods for daily
        'min_volume_score': 0.4,
        'volume_weight': 0.35    # Volume more important for daily
    }
}

class DailyYOLOPredictor:
    def __init__(self):
        """Initialize the daily predictor with Azure connection"""
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
        
        # Load YOLO model
        self.model = YOLO(CONFIG['model_path'])
        
        # Initialize volume analyzer
        if VOLUME_MODULE_AVAILABLE and CONFIG['volume_analysis']['enabled']:
            self.volume_analyzer = VolumeConfirmation(CONFIG['volume_analysis']['lookback_periods'])
            logger.info("Volume confirmation module initialized for daily analysis")
        else:
            self.volume_analyzer = None
            logger.info("Running without volume confirmation")
    
    def fetch_daily_data(self, ticker: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Fetch daily or weekly data for pattern analysis"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get appropriate lookback period
            lookback_days = CONFIG['chart_periods'][interval]['lookback_days']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Fetch data
            data = stock.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty or len(data) < CONFIG['chart_periods'][interval]['min_bars']:
                logger.warning(f"Insufficient data for {ticker} {interval}: {len(data)} bars")
                return None
            
            logger.info(f"Fetched {len(data)} {interval} bars for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} {interval}: {e}")
            return None
    
    def analyze_daily_volume_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns specific to daily charts"""
        if self.volume_analyzer is None or len(data) < 20:
            return {
                'volume_score': 0.5,
                'volume_quality': 'unknown',
                'volume_breakout': False,
                'volume_metrics': {}
            }
        
        try:
            # Calculate volume metrics
            data_with_metrics = self.volume_analyzer.calculate_volume_metrics(data)
            
            # Daily-specific volume analysis
            recent_volume = data_with_metrics['Volume'].iloc[-5:].mean()
            avg_volume = data_with_metrics['avg_volume'].iloc[-1]
            volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Check for volume breakout (important for daily patterns)
            volume_breakout = (
                data_with_metrics['Volume'].iloc[-1] > avg_volume * 1.5 and
                data_with_metrics['Volume'].iloc[-1] > data_with_metrics['Volume'].iloc[-2]
            )
            
            # OBV trend for daily
            obv_trend = data_with_metrics['obv'].iloc[-5:].mean() > data_with_metrics['obv'].iloc[-20:-5].mean()
            
            # Calculate daily volume score
            volume_score = 0.5
            
            if volume_spike > 1.5:
                volume_score += 0.25
            elif volume_spike > 1.2:
                volume_score += 0.15
            elif volume_spike < 0.5:
                volume_score -= 0.2
            
            if volume_breakout:
                volume_score += 0.2
            
            if obv_trend:
                volume_score += 0.15
            
            volume_score = max(0, min(1, volume_score))
            
            quality = 'high' if volume_score > 0.7 else ('low' if volume_score < 0.4 else 'medium')
            
            return {
                'volume_score': round(volume_score, 3),
                'volume_quality': quality,
                'volume_breakout': volume_breakout,
                'volume_metrics': {
                    'volume_spike': round(volume_spike, 2),
                    'recent_daily_volume': int(recent_volume),
                    'avg_daily_volume': int(avg_volume),
                    'obv_trend': 'bullish' if obv_trend else 'bearish',
                    'days_above_average': sum(data_with_metrics['Volume'].iloc[-5:] > avg_volume)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in daily volume analysis: {e}")
            return {
                'volume_score': 0.5,
                'volume_quality': 'unknown',
                'volume_breakout': False,
                'volume_metrics': {}
            }
    
    def generate_daily_chart(self, data: pd.DataFrame, ticker: str, interval: str, save_path: str, volume_analysis: Dict) -> bool:
        """Generate daily/weekly candlestick chart optimized for pattern detection"""
        try:
            ticker_info = CONFIG['tickers'][ticker]
            
            # Volume quality indicator
            quality_indicator = {
                'high': '🟢',
                'medium': '🟡',
                'low': '🔴',
                'unknown': '⚪'
            }.get(volume_analysis.get('volume_quality', 'unknown'), '⚪')
            
            # Add breakout indicator if present
            breakout_indicator = ' 🚀' if volume_analysis.get('volume_breakout', False) else ''
            
            title = f'{ticker_info["name"]} ({ticker}) - {interval} {quality_indicator}{breakout_indicator}'
            
            # Create enhanced daily chart
            mpf.plot(
                data,
                type='candle',
                style='yahoo',
                title=title,
                volume=True,
                savefig=save_path,
                figsize=(14, 10),  # Larger size for daily patterns
                mav=(10, 20, 50),  # Add moving averages for daily
                warn_too_much_data=1000
            )
            return True
            
        except Exception as e:
            logger.error(f"Error generating daily chart: {e}")
            return False
    
    def calculate_swing_targets(self, current_price: float, signal_type: str, ticker: str) -> Dict:
        """Calculate stop loss and take profit for swing trading"""
        ticker_info = CONFIG['tickers'][ticker]
        holding_days = ticker_info.get('holding_days', 3)
        
        # Daily ATR-based stops (wider than intraday)
        if signal_type == 'BUY':
            # Wider stops for swing trading
            stop_loss = current_price * 0.97  # 3% stop
            take_profit_1 = current_price * 1.05  # 5% target
            take_profit_2 = current_price * 1.10  # 10% stretch target
        else:  # SELL
            stop_loss = current_price * 1.03
            take_profit_1 = current_price * 0.95
            take_profit_2 = current_price * 0.90
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit_1': round(take_profit_1, 2),
            'take_profit_2': round(take_profit_2, 2),
            'holding_days': holding_days,
            'risk_reward_ratio': round(abs(take_profit_1 - current_price) / abs(stop_loss - current_price), 2)
        }
    
    def run_daily_detection(self, image_path: str, volume_score: float) -> Dict:
        """Run YOLO detection optimized for daily patterns"""
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
                        
                        # Volume-adjusted confidence for daily patterns
                        volume_weight = CONFIG['volume_analysis']['volume_weight']
                        adjusted_conf = conf * (1 - volume_weight) + (conf * volume_score) * volume_weight
                        
                        # Higher threshold for daily signals
                        if adjusted_conf > CONFIG['confidence_threshold']:
                            detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'volume_adjusted_confidence': adjusted_conf,
                                'strength': 'Strong' if adjusted_conf > 0.7 else 'Moderate'
                            })
            
            # Aggregate signals for daily timeframe
            buy_signals = [d for d in detections if d['class'] == 'Buy']
            sell_signals = [d for d in detections if d['class'] == 'Sell']
            
            # Determine primary signal
            if buy_signals and not sell_signals:
                primary_signal = 'BUY'
            elif sell_signals and not buy_signals:
                primary_signal = 'SELL'
            elif buy_signals and sell_signals:
                # Compare average confidences
                avg_buy = sum(d['volume_adjusted_confidence'] for d in buy_signals) / len(buy_signals)
                avg_sell = sum(d['volume_adjusted_confidence'] for d in sell_signals) / len(sell_signals)
                primary_signal = 'BUY' if avg_buy > avg_sell else 'SELL'
            else:
                primary_signal = 'NEUTRAL'
            
            return {
                'primary_signal': primary_signal,
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals),
                'detections': detections,
                'signal_strength': 'Strong' if len(detections) > 2 else ('Moderate' if len(detections) > 0 else 'Weak')
            }
            
        except Exception as e:
            logger.error(f"Error in daily detection: {e}")
            return {
                'primary_signal': 'NEUTRAL',
                'buy_count': 0,
                'sell_count': 0,
                'detections': [],
                'signal_strength': 'None'
            }
    
    def run_daily_analysis(self, ticker: str) -> Optional[Dict]:
        """Run complete daily analysis for a ticker"""
        timestamp = datetime.utcnow()
        
        # Get current price
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
            if not current_price:
                logger.error(f"Could not get current price for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
            return None
        
        ticker_info = CONFIG['tickers'][ticker]
        
        results = {
            'timestamp': timestamp.isoformat(),
            'analysis_type': 'daily_swing_trading',
            'ticker': ticker,
            'name': ticker_info['name'],
            'current_price': current_price,
            'charts_analyzed': {},
            'recommendation': None,
            'targets': None
        }
        
        # Create temp directory
        os.makedirs('temp_charts', exist_ok=True)
        
        signals_collected = []
        volume_scores = []
        
        # Analyze each timeframe
        for interval, config in CONFIG['chart_periods'].items():
            logger.info(f"Analyzing {ticker} {interval} chart...")
            
            # Fetch data
            data = self.fetch_daily_data(ticker, interval)
            if data is None:
                continue
            
            # Volume analysis
            volume_analysis = self.analyze_daily_volume_patterns(data)
            volume_scores.append(volume_analysis['volume_score'])
            
            # Generate chart
            chart_path = f"temp_charts/{ticker.replace('.', '_')}_{interval}_daily.png"
            if not self.generate_daily_chart(data, ticker, interval, chart_path, volume_analysis):
                continue
            
            # Run detection
            detection = self.run_daily_detection(chart_path, volume_analysis['volume_score'])
            
            # Store results
            results['charts_analyzed'][interval] = {
                'description': config['description'],
                'bars_analyzed': len(data),
                'signal': detection['primary_signal'],
                'signal_strength': detection['signal_strength'],
                'pattern_count': len(detection['detections']),
                'volume_analysis': volume_analysis,
                'period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
            }
            
            if detection['primary_signal'] != 'NEUTRAL':
                signals_collected.append({
                    'interval': interval,
                    'signal': detection['primary_signal'],
                    'strength': detection['signal_strength'],
                    'volume_score': volume_analysis['volume_score']
                })
            
            # Save chart to Azure
            with open(chart_path, 'rb') as f:
                chart_blob_name = f"yolo_daily/{ticker}/{timestamp.strftime('%Y-%m-%d')}/{interval}_chart.png"
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=chart_blob_name
                )
                blob_client.upload_blob(f, overwrite=True)
            
            os.remove(chart_path)
        
        # Generate recommendation based on daily analysis
        if not signals_collected:
            results['recommendation'] = 'HOLD'
            results['recommendation_strength'] = 'No clear patterns detected'
        else:
            # Weight signals by timeframe and volume
            buy_score = sum(1.5 if s['interval'] == '1d' else 1.0 
                          for s in signals_collected if s['signal'] == 'BUY')
            sell_score = sum(1.5 if s['interval'] == '1d' else 1.0 
                           for s in signals_collected if s['signal'] == 'SELL')
            
            avg_volume_score = np.mean(volume_scores) if volume_scores else 0.5
            
            if buy_score > sell_score * 1.5 and avg_volume_score > 0.5:
                results['recommendation'] = 'STRONG BUY'
                results['targets'] = self.calculate_swing_targets(current_price, 'BUY', ticker)
            elif sell_score > buy_score * 1.5 and avg_volume_score > 0.5:
                results['recommendation'] = 'STRONG SELL' 
                results['targets'] = self.calculate_swing_targets(current_price, 'SELL', ticker)
            elif buy_score > sell_score:
                results['recommendation'] = 'BUY'
                results['targets'] = self.calculate_swing_targets(current_price, 'BUY', ticker)
            elif sell_score > buy_score:
                results['recommendation'] = 'SELL'
                results['targets'] = self.calculate_swing_targets(current_price, 'SELL', ticker)
            else:
                results['recommendation'] = 'HOLD'
            
            # Add volume warning
            if avg_volume_score < CONFIG['volume_analysis']['min_volume_score']:
                results['recommendation'] += ' (Low Volume Warning)'
            
            results['recommendation_strength'] = f"Based on {len(signals_collected)} pattern(s)"
            results['average_volume_score'] = round(avg_volume_score, 3)
        
        # Save to Azure
        blob_name = f"yolo_daily/{ticker}/{timestamp.strftime('%Y-%m-%d')}/analysis.json"
        self.save_to_azure(results, blob_name)
        
        # Also save to standard predictions folder for Shiny app
        shiny_blob_name = f"predictions/{ticker}/{timestamp.strftime('%Y-%m-%d')}/daily.json"
        self.save_to_azure(results, shiny_blob_name)
        
        logger.info(f"{ticker} daily analysis complete: {results['recommendation']}")
        
        return results
    
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
    
    def run_all_tickers(self):
        """Run daily analysis for all configured tickers"""
        results = {}
        
        for ticker in CONFIG['tickers']:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running daily analysis for {ticker}")
                logger.info(f"{'='*60}")
                
                result = self.run_daily_analysis(ticker)
                if result:
                    results[ticker] = result
                    
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue
        
        # Save summary
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'analysis_type': 'daily_swing_trading',
            'tickers_analyzed': len(results),
            'recommendations': {
                ticker: {
                    'recommendation': result['recommendation'],
                    'price': result['current_price'],
                    'targets': result.get('targets')
                }
                for ticker, result in results.items()
            }
        }
        
        summary_blob = f"yolo_daily/summary/{datetime.utcnow().strftime('%Y-%m-%d')}.json"
        self.save_to_azure(summary, summary_blob)
        
        logger.info("\n" + "="*60)
        logger.info("Daily Analysis Summary:")
        for ticker, result in results.items():
            logger.info(f"{ticker}: {result['recommendation']} @ ${result['current_price']}")
        logger.info("="*60)
        
        return results


def main():
    """Main execution function"""
    try:
        predictor = DailyYOLOPredictor()
        predictor.run_all_tickers()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()