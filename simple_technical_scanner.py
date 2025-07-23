#!/usr/bin/env python3
"""
Simple Technical Indicator Scanner for Intraday Trading
Uses momentum indicators optimized for 15-minute timeframes
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

# Load Azure credentials
load_dotenv('config/.env')

class SimpleTechnicalScanner:
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
        
        # Technical indicator parameters (optimized for 15-min)
        self.params = {
            'sma_fast': 10,
            'sma_slow': 30,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'volume_sma': 20,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2
        }
    
    def calculate_indicators(self, data):
        """Calculate all technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_fast'] = df['Close'].rolling(window=self.params['sma_fast']).mean()
        df['SMA_slow'] = df['Close'].rolling(window=self.params['sma_slow']).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'], self.params['rsi_period'])
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close'],
            self.params['stoch_k_period'], self.params['stoch_d_period']
        )
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['Close'].rolling(window=self.params['bb_period']).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * self.params['bb_std'])
        df['BB_lower'] = df['BB_middle'] - (bb_std * self.params['bb_std'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=self.params['volume_sma']).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # ATR for volatility
        df['ATR'] = self.calculate_atr(df, self.params['atr_period'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Price momentum
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def get_dynamic_thresholds(self, df):
        """Adjust thresholds based on current volatility"""
        current_atr = df['ATR'].iloc[-1]
        avg_atr = df['ATR'].mean()
        
        if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr == 0:
            volatility_ratio = 1.0
        else:
            volatility_ratio = current_atr / avg_atr
        
        # Adjust RSI thresholds based on volatility
        if volatility_ratio > 1.5:  # High volatility
            rsi_oversold = 35
            rsi_overbought = 65
        elif volatility_ratio < 0.7:  # Low volatility
            rsi_oversold = 25
            rsi_overbought = 75
        else:  # Normal volatility
            rsi_oversold = self.params['rsi_oversold']
            rsi_overbought = self.params['rsi_overbought']
        
        return {
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volatility_ratio': volatility_ratio
        }
    
    def generate_signals(self, df):
        """Generate trading signals from indicators"""
        signals = []
        
        # Get dynamic thresholds
        thresholds = self.get_dynamic_thresholds(df)
        
        # Current values
        current_idx = -1
        current_time = df.index[current_idx]
        
        # Initialize signal components
        signal_components = {
            'ma_cross': 0,
            'rsi': 0,
            'stoch': 0,
            'bb': 0,
            'macd': 0,
            'volume': 0,
            'momentum': 0
        }
        
        # 1. Moving Average Crossover
        if df['SMA_fast'].iloc[current_idx] > df['SMA_slow'].iloc[current_idx]:
            if df['SMA_fast'].iloc[current_idx-1] <= df['SMA_slow'].iloc[current_idx-1]:
                signal_components['ma_cross'] = 2  # Fresh bullish cross
            else:
                signal_components['ma_cross'] = 1  # Continued bullish
        elif df['SMA_fast'].iloc[current_idx] < df['SMA_slow'].iloc[current_idx]:
            if df['SMA_fast'].iloc[current_idx-1] >= df['SMA_slow'].iloc[current_idx-1]:
                signal_components['ma_cross'] = -2  # Fresh bearish cross
            else:
                signal_components['ma_cross'] = -1  # Continued bearish
        
        # 2. RSI with dynamic thresholds
        current_rsi = df['RSI'].iloc[current_idx]
        if not pd.isna(current_rsi):
            if current_rsi < thresholds['rsi_oversold']:
                signal_components['rsi'] = 1
            elif current_rsi > thresholds['rsi_overbought']:
                signal_components['rsi'] = -1
        
        # 3. Stochastic
        current_stoch_k = df['STOCH_K'].iloc[current_idx]
        current_stoch_d = df['STOCH_D'].iloc[current_idx]
        if not pd.isna(current_stoch_k) and not pd.isna(current_stoch_d):
            if current_stoch_k < self.params['stoch_oversold'] and current_stoch_k > current_stoch_d:
                signal_components['stoch'] = 1
            elif current_stoch_k > self.params['stoch_overbought'] and current_stoch_k < current_stoch_d:
                signal_components['stoch'] = -1
        
        # 4. Bollinger Bands
        current_price = df['Close'].iloc[current_idx]
        bb_lower = df['BB_lower'].iloc[current_idx]
        bb_upper = df['BB_upper'].iloc[current_idx]
        if not pd.isna(bb_lower) and not pd.isna(bb_upper):
            if current_price <= bb_lower:
                signal_components['bb'] = 1
            elif current_price >= bb_upper:
                signal_components['bb'] = -1
        
        # 5. MACD
        macd_hist = df['MACD_histogram'].iloc[current_idx]
        macd_hist_prev = df['MACD_histogram'].iloc[current_idx-1]
        if not pd.isna(macd_hist) and not pd.isna(macd_hist_prev):
            if macd_hist > 0 and macd_hist_prev <= 0:
                signal_components['macd'] = 1
            elif macd_hist < 0 and macd_hist_prev >= 0:
                signal_components['macd'] = -1
        
        # 6. Volume confirmation
        volume_ratio = df['Volume_ratio'].iloc[current_idx]
        if not pd.isna(volume_ratio):
            if volume_ratio > 1.5:
                signal_components['volume'] = 1
            elif volume_ratio < 0.5:
                signal_components['volume'] = -1
        
        # 7. Momentum (ROC)
        current_roc = df['ROC'].iloc[current_idx]
        if not pd.isna(current_roc):
            if current_roc > 1.0:
                signal_components['momentum'] = 1
            elif current_roc < -1.0:
                signal_components['momentum'] = -1
        
        # Calculate composite signal
        weights = {
            'ma_cross': 2.0,
            'rsi': 1.5,
            'stoch': 1.0,
            'bb': 1.0,
            'macd': 1.5,
            'volume': 0.5,
            'momentum': 1.0
        }
        
        weighted_sum = sum(signal_components[key] * weights[key] for key in signal_components)
        total_weight = sum(weights.values())
        
        # Determine signal
        if weighted_sum >= 4:
            signal_type = 'BUY'
            confidence = min(weighted_sum / (total_weight * 2), 0.95)
        elif weighted_sum <= -4:
            signal_type = 'SELL'
            confidence = min(abs(weighted_sum) / (total_weight * 2), 0.95)
        else:
            signal_type = 'HOLD'
            confidence = 0.5 - abs(weighted_sum) / (total_weight * 4)
        
        # Calculate stop loss and take profit
        atr = df['ATR'].iloc[current_idx]
        if not pd.isna(atr):
            if signal_type == 'BUY':
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (3 * atr)
            elif signal_type == 'SELL':
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (3 * atr)
            else:
                stop_loss = None
                take_profit = None
        else:
            stop_loss = None
            take_profit = None
        
        signal = {
            'timestamp': current_time,
            'signal': signal_type,
            'confidence': round(confidence, 3),
            'price': round(current_price, 2),
            'components': signal_components,
            'weighted_score': round(weighted_sum, 2),
            'indicators': {
                'rsi': round(current_rsi, 2) if not pd.isna(current_rsi) else None,
                'stoch_k': round(current_stoch_k, 2) if not pd.isna(current_stoch_k) else None,
                'macd_hist': round(macd_hist, 4) if not pd.isna(macd_hist) else None,
                'volume_ratio': round(volume_ratio, 2) if not pd.isna(volume_ratio) else None,
                'atr': round(atr, 2) if not pd.isna(atr) else None,
                'volatility_ratio': round(thresholds['volatility_ratio'], 2)
            },
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit': round(take_profit, 2) if take_profit else None
        }
        
        signals.append(signal)
        return signals
    
    def scan_ticker(self, ticker, interval='15m'):
        """Scan a ticker for technical signals"""
        logger.info(f"Scanning {ticker} with {interval} interval")
        
        # Get data (2 days for 15-min to ensure we have enough history)
        period = '5d' if interval == '15m' else '1mo'
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data retrieved for {ticker}")
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        # Ensure we have enough data
        min_periods = max(self.params.values()) + 10
        if len(df) < min_periods:
            logger.warning(f"Not enough data for {ticker}. Need at least {min_periods} periods.")
            return None
        
        # Generate signals
        signals = self.generate_signals(df)
        
        # Add ticker and additional info
        for signal in signals:
            signal['ticker'] = ticker
            signal['interval'] = interval
        
        logger.info(f"Generated signal for {ticker}: {signals[0]['signal']} "
                   f"with {signals[0]['confidence']:.1%} confidence")
        
        return signals[0] if signals else None
    
    def load_or_create_evaluation_file(self):
        """Load existing evaluation file or create new one"""
        blob_name = "same_day_technical/technical_evaluations.json"
        
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
    
    def save_evaluation(self, signal_data):
        """Append evaluation to the cumulative file"""
        evaluations = self.load_or_create_evaluation_file()
        
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'ticker': signal_data['ticker'],
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'price': signal_data['price'],
            'indicators': signal_data['indicators'],
            'components': signal_data['components']
        }
        
        evaluations.append(evaluation)
        
        # Keep only last 1000 evaluations
        if len(evaluations) > 1000:
            evaluations = evaluations[-1000:]
        
        # Save back to Azure
        blob_name = "same_day_technical/technical_evaluations.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(evaluations, indent=2, default=str)
        blob_client.upload_blob(json_data, overwrite=True)
        logger.info(f"Saved evaluation. Total evaluations: {len(evaluations)}")
    
    def prepare_signal_for_evaluation(self, signal_data):
        """Prepare signal data with metadata for evaluation storage"""
        # Add metadata
        signal_data['scan_time'] = datetime.now().isoformat()
        signal_data['expiry_time'] = (datetime.now() + timedelta(minutes=15)).isoformat()
        
        # Calculate signal strength for UI
        if signal_data['signal'] == 'HOLD':
            signal_data['strength'] = 'Neutral'
        elif signal_data['confidence'] >= 0.8:
            signal_data['strength'] = 'Strong'
        elif signal_data['confidence'] >= 0.6:
            signal_data['strength'] = 'Moderate'
        else:
            signal_data['strength'] = 'Weak'
        
        return signal_data
    
    def run_scan(self, tickers=['NVDA']):
        """Run technical scan for specified tickers"""
        all_signals = []
        
        for ticker in tickers:
            try:
                signal = self.scan_ticker(ticker)
                if signal:
                    # Prepare signal with metadata
                    prepared_signal = self.prepare_signal_for_evaluation(signal)
                    all_signals.append(prepared_signal)
                    
                    # Save to unified evaluation file only
                    self.save_evaluation(prepared_signal)
                    
                    logger.info(f"Saved signal for {ticker}: {prepared_signal['signal']} "
                               f"({prepared_signal['strength']}) at ${prepared_signal['price']}")
                    
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        logger.info(f"Scan complete. Generated {len(all_signals)} signals")
        return all_signals
    
    def get_performance_summary(self):
        """Get recent performance summary"""
        evaluations = self.load_or_create_evaluation_file()
        
        if len(evaluations) < 10:
            return None
        
        # Get last 100 evaluations or all if less
        recent = evaluations[-100:]
        
        # Calculate metrics
        buy_signals = sum(1 for e in recent if e['signal'] == 'BUY')
        sell_signals = sum(1 for e in recent if e['signal'] == 'SELL')
        hold_signals = sum(1 for e in recent if e['signal'] == 'HOLD')
        avg_confidence = sum(e['confidence'] for e in recent) / len(recent)
        
        # Signal distribution by hour (for intraday patterns)
        hour_distribution = {}
        for e in recent:
            hour = datetime.fromisoformat(e['timestamp']).hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        summary = {
            'total_signals': len(recent),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_confidence': round(avg_confidence, 3),
            'hour_distribution': hour_distribution,
            'last_updated': datetime.now().isoformat()
        }
        
        # Save summary
        blob_name = "same_day_technical/performance_summary.json"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        json_data = json.dumps(summary, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        
        return summary


def main():
    """Main function for GitHub Actions"""
    scanner = SimpleTechnicalScanner()
    
    # Get ticker from environment or default
    ticker = os.getenv('TICKER', 'NVDA')
    tickers = [ticker] if ticker != 'ALL' else ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Run scan
    signals = scanner.run_scan(tickers)
    
    # Update performance summary
    summary = scanner.get_performance_summary()
    if summary:
        logger.info(f"Performance: {summary['buy_signals']} buys, "
                   f"{summary['sell_signals']} sells, "
                   f"{summary['hold_signals']} holds "
                   f"(avg confidence: {summary['avg_confidence']:.1%})")
    
    logger.info("Simple technical scan completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())